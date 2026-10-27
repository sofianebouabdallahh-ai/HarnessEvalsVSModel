# Harness Matrix

**An agent's score is a property of the (model, harness) pair, not of the model.** Public numbers hold the harness constant and vary the model. This platform holds the model constant, varies the harness, and measures what moves: task success, safety under prompt injection, cost, and variance.

```
                  raw_loop      claude_sdk     opencode (H² base)   harness2 (frozen)
Model X (fixed)   [pass^1±CI]   [pass^1±CI]    [pass^1±CI]          [pass^1±CI]
                  [ASR]         [ASR]          [ASR]                [ASR]
```

The claim under test: *for a fixed model, swapping the harness changes task success by more than the gap between adjacent frontier models, and changes prompt-injection resistance by even more.*

- [ROADMAP.md](ROADMAP.md) — the decisions taken and what is verified live
- [docs/metrics.md](docs/metrics.md) — every metric, defined before any scorer was written
- [docs/harness_engineering_report.md](docs/harness_engineering_report.md) — the Harness² test written for engineers: what a self-improving harness buys, what it costs, how to ship it safely ([PDF](docs/harness_engineering_report.pdf))
- [docs/replication_note.md](docs/replication_note.md) — the same study written for researchers, every number with its interval and run id

## Why not Inspect / HAL / Harbor / promptfoo?

They are excellent at what they do and we reuse their benchmarks and sandboxes. None of them treats **the harness as the independent variable**, and none puts **a security axis** (utility under attack, attack success rate) next to the capability axis on the same board.

## Ten-minute demo (zero API cost)

```bash
uv sync
docker compose up -d          # Postgres 16 + Grafana, schema and dashboard provisioned
uv run hm run --demo          # mock model × raw_loop × toy suites, N=5 → score → aggregate → load
open http://localhost:3001/d/harness-matrix
```

Without Docker, `hm run --demo` still writes `runs/<run_id>/{manifest.json,traces.jsonl,scores.jsonl,cells.json}` and prints the matrix.

## How it works

```
hm run  ──▶ for each (harness, task, seed):
              env-server (MCP over HTTP) owns the benchmark env, serves its tools,
              enforces the tool budget, tags untrusted/injected results
              adapter runs the harness against it, reports LLM calls + final output
              ⇒ one normalized Trace (JSONL, raw args/results kept)
hm score     Trace → Score[]   (benchmark ground truth · cost · trajectory stats)
hm aggregate episodes → cells  (pass^k, ASR, $/solve … each with a bootstrap 95% CI)
hm load      → Postgres        → Grafana
```

Scorers consume traces only, never harness objects. Adapters cannot under-report tool calls because the env-server log is the record.

## What exists

| Piece | Status |
|---|---|
| Columns (harness adapters) | `raw_loop` · `claude_sdk` (Claude Code via Agent SDK, bundled CLI) · `opencode` (pinned 1.14.18 binary) · `harness2` (frozen Harness² tree on OpenCode) · `harness2_online` (Harness²'s per-task online protocol, `hm refine-online`) |
| Live (2026-10-25) | AgentDojo `workspace` (benign / attacked, N=5): `raw_loop` 1.00 / 0.86, `claude_sdk` 0.98 / 0.99, `opencode` 0.98 / 0.95, `harness2` baseline tree 0.95 / 0.95, k=1 0.85 / 0.86, k=2 0.94 / 0.95; ASR 0 everywhere; the bare loop's attacked losses are API refusals (14%). τ² airline with a Haiku customer: `claude_sdk` 0.72, `opencode` 0.61, `harness2_online` 0.79 (+0.19 [+0.03, +0.36] paired vs OpenCode, at 20× the price per solve). Online Harness² on the attacked dojo stream, standard and editor-targeted injections: no harness file ever carried attacker text (31 tasks). Custom suite on `claude_sdk`: 0/50 violations. Harness² replication note: [docs/replication_note.md](docs/replication_note.md); engineering report: [docs/harness_engineering_report.md](docs/harness_engineering_report.md) |
| Benchmarks | `toy` (demo) · `dojo` (AgentDojo v1.2.2, `workspace{,_attacked}` + the other suites wired) · `tau2` (τ²-bench v1.0.1 `airline`/`retail`/`telecom`, env-side user simulator with the telecom customer's own tools; retail graded on DB only, D13) · `custom` (sandboxed safety suite, 10 attacked tasks + benign twins = 20; live on `claude_sdk`: 0/50 violations, 0 over-refusal, injection seen in 50%) |
| Scorers | benchmark ground truth · cost with prompt-cache tiers (+ `usd_improver`) · trajectory · fairness (`tainted`) · harness artifact (`harness_poisoned`, `defensive_edit`) · dojo `poison_executed` on benign episodes |
| Aggregation | pass^k, ASR, ASR-given-seen, $/solve (and incl. improvement compute), over-refusal, per-task variance, horizon (calls/turns per episode, tool-error rate), poisoning rates — bootstrap 95% CI on every cell; `scripts/paired.py` for paired per-task tests; `scripts/figures.py` for the figures |
| Modes | `out_of_the_box` and `equalized_prompt` |
| Dashboard | 7 Grafana panels incl. the safety–utility frontier and the mode delta |
| Live so far | `claude_sdk` × `dojo/workspace{,_attacked}`, 20 tasks × N=5 on a Claude plan login |

Disk and Docker budget, guards and `hm doctor`: [docs/disk.md](docs/disk.md).

Design notes per piece: [docs/metrics.md](docs/metrics.md) · [docs/tau2.md](docs/tau2.md) · [docs/opencode.md](docs/opencode.md) · [docs/harness2.md](docs/harness2.md) · [docs/custom_suite.md](docs/custom_suite.md).

## Running the matrix

```bash
cp .env.example .env            # ANTHROPIC_API_KEY (all columns) or none (claude_sdk on a Claude plan login)
set -a; source .env; set +a
# smoke: two tasks, one seed
HM_DOJO_MAX_TASKS=2 uv run hm run --spec specs/claude-baseline.yaml \
    -H raw_loop -H claude_sdk -H opencode -H harness2 -s dojo/workspace -s dojo/workspace_attacked --n 1
# the matrix, both modes
uv run hm run --spec specs/claude-baseline.yaml -H raw_loop -H claude_sdk -H opencode -H harness2 \
    -s dojo/workspace -s dojo/workspace_attacked -s tau2/airline -s custom/safety -s custom/safety_benign --n 5
uv run hm run ... --mode equalized_prompt
uv run hm score && uv run hm aggregate && uv run hm load && uv run hm matrix
```

`claude_sdk` runs on the Claude plan login (the only column the plan terms allow) and ignores an exported
`ANTHROPIC_API_KEY` unless `HM_CLAUDE_SDK_AUTH=key`. The other columns need the key; the model string is
identical and logged either way. Every run records its price-table spend in `manifest.json`
(`usd_table`); `--max-usd 25` or `HM_MAX_USD=25` stops a run after the episode that reaches the cap,
keeping what was written (`stopped_early` says why). The bare loop uses prompt caching
(`HM_ANTHROPIC_CACHE=0` turns it off).

What is and is not verified live: [ROADMAP.md](ROADMAP.md).

## Independence

Public benchmarks, personal account, personal time. No employer code or data.

## License

Apache-2.0
