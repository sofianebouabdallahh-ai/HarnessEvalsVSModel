"""`opencode`: OpenCode (sst/opencode) at the exact version Harness² pins (v1.14.18), run as
`opencode run --format json`. The research baseline column and the base of the `harness2` column.

What is the harness here: OpenCode's default provider system prompt, its loop, compaction,
retries, and whatever its default agent does between turns. What is held constant: the model
string, the benchmark tools (served by the env-server over MCP), the limits.

Fairness notes recorded in the manifest:
- OpenCode's default system prompt is not retrievable from the CLI; `system_prompt_sha256` is
  null with a reason and the binary version is recorded. In `equalized_prompt` mode the spec's
  prompt becomes a custom agent `hm` (`--agent hm`) whose `prompt` REPLACES the default.
- No user config, project config, plugins or auth store is loaded: config comes only from
  `OPENCODE_CONFIG_CONTENT`, `OPENCODE_CONFIG_DIR` points at an empty dir, project config is
  disabled, `--pure` skips external plugins, and XDG dirs point at a platform-owned home.
- Built-in tools are disabled (`tools: {<name>: false}` + `permission: deny`), decision D8; any
  built-in that still fires is recorded as an `opencode:<tool>` step so it is never invisible.
- OpenCode has no turn cap: `max_turns` is logged, not enforced; the env-server tool budget and
  the wall clock bound the episode.
- Sampling: provider default (D12).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from hm.adapters.base import AdapterRun, EnvHandle
from hm.envserver.server import SERVER_NAME, canonical_tool_name
from hm.types import (
    AdapterManifest,
    AgentSpec,
    LLMCallStep,
    Task,
    ToolCallStep,
    ToolResultStep,
    sha256_json,
    sha256_text,
)

DEFAULT_BIN = "/Users/sarahch/Documents/Repos/HarnessEvalsVSModel/.tools/opencode-1.14.18/opencode"
DEFAULT_HOME = "/Users/sarahch/Documents/Repos/HarnessEvalsVSModel/.tools/opencode-home"
EQUALIZED_AGENT = "hm"

# OpenCode built-ins as of 1.14.18 (tool ids are lower-case in config).
BUILTIN_TOOLS = [
    "bash", "edit", "write", "read", "glob", "grep", "list", "patch", "multiedit", "webfetch",
    "websearch", "task", "todowrite", "todoread", "skill", "question",
]  # fmt: skip
PERMISSION_KEYS = ["bash", "edit", "read", "glob", "grep", "list", "webfetch", "websearch", "task",
                   "todowrite", "todoread", "skill"]  # fmt: skip
OUTPUT_TOKEN_MAX = 64_000  # Harness² ceiling, mirrored


def opencode_bin() -> str:
    return os.environ.get("HM_OPENCODE_BIN", DEFAULT_BIN)


def opencode_home() -> Path:
    return Path(os.environ.get("HM_OPENCODE_HOME", DEFAULT_HOME))


def opencode_version() -> str | None:
    try:
        return subprocess.run(
            [opencode_bin(), "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def provider_config(spec: AgentSpec) -> dict[str, Any]:
    """`provider.<id>` block + the `provider/model` string OpenCode expects."""
    p = spec.model.provider
    if p == "anthropic":
        return {
            "anthropic": {"options": {"apiKey": "{env:ANTHROPIC_API_KEY}"}}
        }, f"anthropic/{spec.model.name}"
    if p in ("mock", "openai_compat"):
        base_url = os.environ.get("HM_OPENAI_COMPAT_URL")
        if not base_url:
            raise RuntimeError(
                "provider 'mock' under opencode needs HM_OPENAI_COMPAT_URL (see hm.models.openai_compat)"
            )
        return (
            {
                "hm_mock": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "hm mock",
                    "options": {"baseURL": base_url, "apiKey": "none"},
                    "models": {
                        spec.model.name: {
                            "name": spec.model.name,
                            "limit": {"context": 200_000, "output": 8192},
                        }
                    },
                }
            },
            f"hm_mock/{spec.model.name}",
        )
    raise ValueError(f"opencode adapter: unsupported provider {p!r}")


def build_config(
    spec: AgentSpec, env_url: str, harness_dir_extra: dict[str, Any] | None = None
) -> tuple[dict, str, str | None]:
    """Returns (config dict, provider/model string, agent name or None)."""
    providers, model_ref = provider_config(spec)
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": providers,
        "mcp": {SERVER_NAME: {"type": "remote", "url": env_url, "enabled": True}},
        "tools": {t: False for t in BUILTIN_TOOLS},
        "permission": {k: "deny" for k in PERMISSION_KEYS},
        "share": "disabled",
        "autoupdate": False,
    }
    agent = None
    if spec.system_prompt is not None:
        config["agent"] = {
            EQUALIZED_AGENT: {
                "mode": "primary",
                "prompt": spec.system_prompt,
                "description": "equalized prompt",
            }
        }
        agent = EQUALIZED_AGENT
    if harness_dir_extra:
        config.update(harness_dir_extra)
    return config, model_ref, agent


def run_env(workspace: Path, config: dict[str, Any], cfg_dir: Path) -> dict[str, str]:
    home = opencode_home()
    e = {k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "PATH", "TMPDIR", "LANG", "LC_"))}
    # HOME is a platform-owned directory, not the user's: OpenCode 1.14 loads skills from
    # ~/.claude/skills and ~/.agents/skills and global AGENTS.md/CLAUDE.md from the home dir.
    (home / "home").mkdir(parents=True, exist_ok=True)
    e.update(
        {
            "HOME": str(home / "home"),
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_CONFIG_DIR": str(cfg_dir),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            # Harness²'s permission fence: only the workspace is reachable.
            "OPENCODE_PERMISSION": json.dumps(
                {"external_directory": {"*": "deny", str(workspace): "allow", f"{workspace}/**": "allow"}}
            ),
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(OUTPUT_TOKEN_MAX),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_STATE_HOME": str(home / "state"),
        }
    )
    return e


def parse_export(raw: str) -> dict[str, Any] | None:
    """`opencode export` prints a preamble line, then the session JSON (`info` + `messages`)."""
    i = raw.find("{")
    if i < 0:
        return None
    try:
        d = json.loads(raw[i:])
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("messages"), list) else None


def export_session(session_id: str, env: dict[str, str]) -> dict[str, Any] | None:
    """The session as OpenCode stored it. The JSON event stream of `opencode run` is not a complete
    record: on 1.14.18 the process can exit 0 right after a later step starts, before that step's
    text part is printed (seen on 7 of 8 real-model episodes in the first key smoke, while the
    database held the finished reply). The export is therefore the authoritative transcript."""
    try:
        p = subprocess.run(
            [opencode_bin(), "export", session_id], env=env, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_export(p.stdout) if p.returncode == 0 else None


def _finish_to_stop_reason(reason: str) -> str:
    """OpenCode's (Vercel AI SDK) finish reasons -> the Messages API vocabulary the scorers use.
    `content-filter` is the provider's refusal stop; `length` is `max_tokens`."""
    if reason == "tool-calls":
        return "tool_use"
    if reason == "content-filter":
        return "refusal"
    if reason == "length":
        return "max_tokens"
    return "end_turn" if reason in ("stop", "end-turn", "") else reason


class OpenCodeAdapter:
    name = "opencode"

    def __init__(self) -> None:
        self._bin_version = opencode_version()
        self.version = f"opencode-{self._bin_version or 'unknown'}"
        self._native_tools_seen: set[str] = set()

    # ------------------------------------------------------------------ manifest

    def prepare(self, spec: AgentSpec, task: Task) -> AdapterManifest:
        sp = spec.system_prompt
        return AdapterManifest(
            system_prompt_sha256=sha256_text(sp) if sp is not None else None,
            system_prompt_reason=None
            if sp is not None
            else "OpenCode default provider prompt; not retrievable from the CLI, binary version recorded",
            injected_retries=0,
            extra_tools=sorted(self._native_tools_seen),
            disabled_native_tools=BUILTIN_TOOLS,
            notes="OpenCode run --format json, transcript from `opencode export` (the stream can end "
            "before the last reply is printed); HOME isolated, no user/project config, skills or plugins; "
            "built-ins disabled; "
            "benchmark tools over MCP (remote); max_turns logged, not enforced (no turn cap in OpenCode); "
            + (
                "equalized prompt installed as custom agent 'hm' (replaces default); "
                if sp is not None
                else ""
            )
            + "sampling: provider default",
            extra={"binary_version": self._bin_version, "harness2_pin": "v1.14.18"},
        )

    # ------------------------------------------------------------------ run

    def run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        return self._run(spec, task, seed, env, prompt=task.prompt, session=None, workspace_setup=None)

    def continue_session(
        self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle, session_id: str, message: str
    ) -> AdapterRun:
        """Resume an OpenCode session with a follow-up message (Phase 2 user-sim loops)."""
        return self._run(spec, task, seed, env, prompt=message, session=session_id, workspace_setup=None)

    def _run(
        self,
        spec: AgentSpec,
        task: Task,
        seed: int,
        env: EnvHandle,
        *,
        prompt: str,
        session: str | None,
        workspace_setup,
        config_extra: dict[str, Any] | None = None,
    ) -> AdapterRun:
        out = AdapterRun()
        with (
            tempfile.TemporaryDirectory(prefix="hm-oc-ws-") as ws_s,
            tempfile.TemporaryDirectory(prefix="hm-oc-cfg-") as cfg_s,
        ):
            workspace, cfg_dir = Path(ws_s), Path(cfg_s)
            if workspace_setup:
                out.extra.update(workspace_setup(workspace) or {})
            config, model_ref, agent = build_config(spec, env.url, config_extra)
            argv = [opencode_bin(), "run", "--format", "json", "--pure", "--dangerously-skip-permissions",
                    "--model", model_ref, "--dir", str(workspace), "--title", "hm"]  # fmt: skip
            if agent:
                argv += ["--agent", agent]
            if session:
                argv += ["--session", session]
            argv.append(prompt)
            out.extra["argv"] = argv[:-1] + ["<prompt>"]
            t0 = time.time()
            try:
                p = subprocess.run(
                    argv, env=self._env(workspace, config, cfg_dir), capture_output=True, text=True,
                    timeout=spec.limits.wall_clock_s,
                )  # fmt: skip
            except subprocess.TimeoutExpired as e:
                out.error = "wall clock exceeded"
                self._parse_events(
                    (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), out, t0
                )
                return out
            except FileNotFoundError:
                out.error = f"opencode binary not found at {opencode_bin()}"
                return out
            self._parse_events(p.stdout, out, t0)
            if p.returncode != 0 and not out.error:
                out.error = f"opencode exit {p.returncode}: {p.stderr.strip()[-300:]}"
            if sid := out.extra.get("session_id"):
                self._reconcile_with_export(export_session(sid, self._env(workspace, config, cfg_dir)), out)
        return out

    @staticmethod
    def _env(workspace: Path, config: dict[str, Any], cfg_dir: Path) -> dict[str, str]:
        return run_env(workspace, config, cfg_dir)

    def _tool_part(self, part: dict[str, Any], ts: float, out: AdapterRun) -> None:
        """A `tool` part (from the stream or the export). MCP tools are not recorded here — the
        env-server log is their record. `invalid` = the model named a tool OpenCode does not have;
        kept as an errored `opencode:invalid` call (the attempted name is in its args), which the
        fairness scorer treats like any native tool: declared after the first sighting."""
        name = part.get("tool") or "tool"
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        canon = canonical_tool_name(name)
        if name.startswith(f"{SERVER_NAME}_") and canon != name:
            return
        self._native_tools_seen.add(name.split(":")[0])
        args = state.get("input") or {}
        times = state.get("time") or {}
        t_start = (times.get("start") or 0) / 1000 or ts
        t_end = (times.get("end") or 0) / 1000 or ts
        err = state.get("error") if state.get("status") == "error" else None
        call_id = part.get("callID")
        args_d = dict(args) if isinstance(args, dict) else {"_": args}
        out.llm_steps.append(
            (
                t_start,
                ToolCallStep(
                    i=0, name=f"opencode:{name}", args=args_d, args_sha256=sha256_json(args), call_id=call_id
                ),
            )
        )
        out.llm_steps.append(
            (
                t_end,
                ToolResultStep(
                    i=0,
                    call_id=call_id,
                    name=f"opencode:{name}",
                    content=str(state.get("output") or err or ""),
                    latency_ms=int(max(0, t_end - t_start) * 1000),
                    error=str(err) if err else None,
                ),
            )
        )

    def _reconcile_with_export(self, export: dict[str, Any] | None, out: AdapterRun) -> None:
        """Make the export the record: append the assistant steps the stream never printed, take the
        final text from the last assistant text part and the cost from the stored messages."""
        if not export:
            return
        assistants = [m for m in export["messages"] if (m.get("info") or {}).get("role") == "assistant"]
        n_stream = sum(1 for _, st in out.llm_steps if isinstance(st, LLMCallStep))
        added = 0
        last_text: str | None = None
        cost = 0.0
        for idx, m in enumerate(assistants):
            info, parts = m.get("info") or {}, m.get("parts") or []
            cost += float(info.get("cost") or 0.0)
            texts = [
                p.get("text") or ""
                for p in parts
                if p.get("type") == "text" and (p.get("text") or "").strip()
            ]
            if texts:
                last_text = texts[-1]
            if idx < n_stream:
                continue
            tm = info.get("time") or {}
            t_created = (tm.get("created") or 0) / 1000 or time.time()
            t_done = (tm.get("completed") or 0) / 1000 or t_created
            for p in parts:
                if p.get("type") == "tool":
                    self._tool_part(p, t_created, out)
            tok = info.get("tokens") or {}
            cache = tok.get("cache") or {}
            out.llm_steps.append(
                (
                    t_created,
                    LLMCallStep(
                        i=0,
                        input_tokens=int(tok.get("input") or 0),
                        output_tokens=int(tok.get("output") or 0) + int(tok.get("reasoning") or 0),
                        cache_read_tokens=int(cache.get("read") or 0),
                        cache_write_tokens=int(cache.get("write") or 0),
                        latency_ms=int(max(0, t_done - t_created) * 1000),
                        stop_reason=_finish_to_stop_reason(str(info.get("finish") or "")),
                        text="\n".join(texts) or None,
                    ),
                )
            )
            added += 1
        if last_text is not None:
            out.final_output = last_text
        if assistants:
            out.extra["harness_reported_usd"] = cost
        out.extra["opencode_export_messages"] = len(assistants)
        if added:
            out.extra["opencode_stream_truncated"] = added  # assistant steps recovered from the export

    def _parse_events(self, stdout: str, out: AdapterRun, t0: float) -> None:
        """OpenCode `--format json` JSONL: step_start / text / tool_use / step_finish / error."""
        last_text: str | None = None
        step_texts: list[str] = []
        t_step = t0
        n_steps = 0
        for ln in stdout.splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ts = (ev.get("timestamp") or 0) / 1000 or time.time()
            part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
            etype = ev.get("type")
            if etype == "step_start":
                t_step = ts
                step_texts = []
            elif etype == "text":
                txt = part.get("text") or ""
                if txt.strip():
                    step_texts.append(txt)
                    last_text = txt
            elif etype == "tool_use":
                self._tool_part(part, ts, out)
            elif etype == "step_finish":
                n_steps += 1
                tok = part.get("tokens") or {}
                cache = tok.get("cache") or {}
                reason = part.get("reason") or ""
                out.llm_steps.append(
                    (
                        t_step,
                        LLMCallStep(
                            i=0,
                            input_tokens=int(tok.get("input") or 0),
                            output_tokens=int(tok.get("output") or 0) + int(tok.get("reasoning") or 0),
                            cache_read_tokens=int(cache.get("read") or 0),
                            cache_write_tokens=int(cache.get("write") or 0),
                            latency_ms=int(max(0, ts - t_step) * 1000),
                            stop_reason=_finish_to_stop_reason(reason),
                            text="\n".join(step_texts) or None,
                        ),
                    )
                )
                out.extra["harness_reported_usd"] = (out.extra.get("harness_reported_usd") or 0.0) + float(
                    part.get("cost") or 0.0
                )
            elif etype == "error":
                err = ev.get("error") or {}
                msg = (err.get("data") or {}).get("message") or err.get("message") or json.dumps(err)[:200]
                out.error = f"opencode error: {err.get('name', 'Error')}: {msg}"[:300]
            if sid := ev.get("sessionID"):
                out.extra["session_id"] = sid
        out.final_output = last_text
        out.extra["opencode_steps"] = n_steps

    def teardown(self) -> None:
        pass
