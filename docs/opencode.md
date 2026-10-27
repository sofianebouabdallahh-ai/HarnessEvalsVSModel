# `opencode` adapter

OpenCode (sst/opencode) **v1.14.18**, the exact version Harness² pins as its executor, run as a
standalone binary (`.tools/opencode-1.14.18/opencode`, override with `HM_OPENCODE_BIN`). It is the
research-baseline column and the base of the `harness2` column.

## How an episode runs

```
opencode run --format json --pure --dangerously-skip-permissions \
    --model <provider>/<model> --dir <fresh tmp workspace> --title hm [--agent hm] "<task prompt>"
```

with the whole configuration passed in `OPENCODE_CONFIG_CONTENT` (nothing is read from the user's
machine):

| Setting | Value | Why |
|---|---|---|
| `mcp.hm` | `{type: remote, url: <env-server>}` | benchmark tools (D1); OpenCode exposes them as `hm_<tool>` |
| `tools.<builtin>` | `false` for bash, edit, write, read, glob, grep, list, patch, multiedit, webfetch, websearch, task, todowrite, todoread, skill, question | D8: tool set held constant |
| `permission.<builtin>` | `deny` | belt and braces |
| `OPENCODE_PERMISSION` | external_directory fence to the workspace | Harness²'s own fence, mirrored |
| `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` | 64000 | Harness²'s ceiling, mirrored |
| `OPENCODE_CONFIG_DIR` | empty temp dir | no user config |
| `OPENCODE_DISABLE_PROJECT_CONFIG=1` | | no project AGENTS.md / `.opencode` (the `harness2` column re-enables this: the harness *is* project config) |
| `--pure` | | no external plugins |
| `XDG_*_HOME` | `.tools/opencode-home` | OpenCode's db/cache live in a platform-owned home |
| `share: disabled`, `autoupdate: false` | | no outbound traffic beyond the model endpoint |

Verified empirically: with the config above the model is offered exactly the MCP tools (`n_tools == 4`
on the toy benchmark) and no built-in fires.

## What is recorded

- The env-server log is the record for MCP tool calls. Any OpenCode built-in that still fires (or
  a call to a non-existent tool, which OpenCode reports as tool `invalid`) is recorded as an
  `opencode:<tool>` ToolCall/ToolResult pair so it is never invisible.
- `step_finish` events give per-step `tokens {input, output, reasoning, cache{read, write}}` and
  `cost`; they become `LLMCallStep`s (reasoning tokens counted as output) and
  `adapter_extra.harness_reported_usd`.
- `session_id` is kept; `OpenCodeAdapter.continue_session()` resumes it with `--session` for
  Phase 2's user-sim loop.

## Fairness notes (manifest)

- OpenCode's default provider prompt is not retrievable from the CLI: `system_prompt_sha256` is
  null with a reason; the binary version is recorded.
- **No turn cap exists in OpenCode.** `max_turns` is logged, not enforced; the env-server tool
  budget and the wall clock bound the episode. The offline probe showed what that means: a confused
  model looped 1,465 steps before the wall clock cut it.
- `equalized_prompt` mode installs the spec's prompt as a custom primary agent `hm` and runs with
  `--agent hm`, which *replaces* the default prompt.
- Sampling: provider default (D12).

## Offline verification

`hm/models/openai_compat.py` is a tiny OpenAI-compatible `/v1/chat/completions` server (streaming
and non-streaming, OpenAI tool-call format) in front of the zero-cost `MockModel`. The adapter's
provider `mock` points OpenCode's `@ai-sdk/openai-compatible` provider at it
(`HM_OPENAI_COMPAT_URL`). `tests/test_opencode_adapters.py` runs the real binary end to end on toy
tasks this way, in both modes. OpenCode fetches `@ai-sdk/openai-compatible` into its cache on
first use (one-time network access); the cache lives under `.tools/opencode-home`.

The mock model calls tools by the names it is served (`hm_kv_get`), as a real model would; the
env-server canonicalises them back (`kv_get`).

## Live run (needs `ANTHROPIC_API_KEY`)

```
hm run --spec specs/claude-baseline.yaml -H opencode -s dojo/workspace -s dojo/workspace_attacked --n 5
```

The plan login must not be used for this column.
