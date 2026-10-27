# Roadmap

The decisions taken while building the platform, and what is and is not verified live. The Phase 7
plan, its cost model and the reasoning behind D14–D16 are in [docs/phase7.md](docs/phase7.md).

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Env-server per episode, tools served as MCP over streamable HTTP.** Every harness, including `raw_loop`, connects to it. | τ² and AgentDojo tools are Python closures over in-memory state; external harnesses (Claude Code, OpenCode) can only reach them through MCP. One process owning the env gives us uniform `max_turns` enforcement, injection tagging in one place, and DB-state grading that is harness-independent. |
| D2 | **Store raw tool args/results in the JSONL trace; hash only in Postgres.** | Credential-leak, destructive-call and wasted-step scorers need the content. Traces are local files. |
| D3 | **AgentDojo before τ².** | AgentDojo is single-prompt and deterministically graded; τ² needs a user-sim conversation driver with session resume on every harness. Get the security axis (the novel part) first; defer the hardest mounting problem until the pipeline is proven. |
| D4 | **τ²-bench pinned to v1.0.1, `--communicate-scorer substring`.** | v2.0 defaults the communicate check to an LLM judge (gpt-5.5) and is not comparable to the 1.x leaderboard. |
| D5 | **Harness² column = frozen harness, train/test split.** Refine on a train split of the suite with `--k 1`, freeze the adopted harness tree + improvement library, evaluate on held-out tasks. Per-task refinement is a stretch row. | Task-level recursion is per-task and uses no ground truth. Refining on the eval task is both expensive (K+1 rollouts) and makes the "harness" task-specific. |
| D6 | **Improver model pinned to the same model string as the executor.** | Otherwise the refined harness was written by a different model and the column is contaminated. Harness² codes the improver path for Vertex; may need the Vertex-Anthropic patch. |
| D7 | **No NeMo column in v1.** (Revised 2026-10-24; was "Column E = NeMo Fabric".) | NeMo Agent Toolkit was deprecated Oct 2026 (EOL end 2026), so a `nat` column is a frozen reference row at best. NeMo Fabric 0.5.0 is a harness *runner* with adapters for Claude Code, OpenCode, Codex, Deep Agents, Hermes… — the same role as `hm/adapters`, with no tool policy or turn cap and no shipped token/cost metrics; mounting it would re-run our own columns through a thicker layer. If a fifth harness is wanted, mount one of Fabric's adaptees directly. NeMo Helix is a competitor for the positioning table. |
| D8 | **Native harness tools disabled on sim benchmarks; declared in manifest.** | "Tool set held constant" is otherwise false. The custom safety suite is where native tools are measured. |
| D9 | **Egress block is a Phase 1 requirement.** | AgentDojo solutions are on GitHub; Harness² itself warns that its executor fetched a rubric during validation. |
| D10 | **Metrics are defined before scorers exist.** See [docs/metrics.md](docs/metrics.md). Scorers emit exactly those `Score.name`s. | Panels and aggregation query scores by name; the catalog is the contract. |
| D12 | **Sampling is recorded, not controlled.** The Messages API (anthropic SDK 2.3.0) exposes no `temperature`/`top_p`/`top_k`; `ModelSpec.temperature` stays in the spec for provenance and every adapter notes "provider default" in its manifest. | The design's "temperature set identically" control cannot be implemented against the current API. N ≥ 5 seeds and the bootstrap CI carry the variance instead. |
| D11 | **Benchmarks may grade in `Env.snapshot()` and store verdicts in `env_final`.** Scorers still read traces only. | AgentDojo's `Inbox`/`Calendar` rebuild item dicts from `initial_*` lists on validation, so anything the agent created is lost in a dump→validate roundtrip; grading must see live objects. Grading is version-pinned, so re-scoring from traces loses nothing in practice. |
| D13 | **τ² tasks are graded on their programmatic reward components only; dropped components are recorded.** (2026-10-24) Airline `[DB, COMMUNICATE]` and telecom `[ENV_ASSERTION, ACTION]` are unaffected. Retail is `[DB, NL_ASSERTION]` on 112/114 tasks, so retail `success` = DB match, with `reward_basis_dropped=["NL_ASSERTION"]` in `env_final` and in the score explanation. | Design rule: no LLM judge in the loop. Excluding retail entirely would leave a 2-task suite; grading DB-only is lenient but identical for every column, which is what the matrix needs. Retail cells are not comparable to published numbers and the docs say so. |
| D14 | **`harness2_online` is a separate column**: per-task Harness² at k=1 with the library threaded through a seed-ordered stream, improver pinned to the executor model, improvement cost reported in its own `usd_improver` field and in `usd_per_solve_incl_refine`. The frozen `harness2` row stays as the offline variant (D5). | The paper evaluates the online protocol; the frozen row answers a different question (does a refined harness transfer). Both belong in the write-up, labelled. |
| D15 | **Editor-targeted injections are an hm-registered AgentDojo attack (`harness_editor_note`), run as their own arm and never pooled with the standard attack.** | The standard attack tests the paper's claim as stated; the targeted one tests a threat the paper does not consider. Mixing them would make ASR uninterpretable. |
| D16 | **Poisoning is measured on the artifact and on behaviour**: `library_poisoned` reads the adopted trees and experience files; `poison_executed` requires an attacker-goal action in an episode with no injection present. A guardrail that mentions the attacker's address is contamination, not poisoning, and is reported under `library_poisoned` only. | Text in a harness is not harm; an action on a clean task is. Keeping the two apart stops the headline from overstating. |

## Status

Built and verified live (numbers in [README.md](README.md), [docs/replication_note.md](docs/replication_note.md)
and the per-piece docs):

- **Platform.** Env-server serving benchmark tools over MCP with injection tagging and a tool budget;
  normalized traces validated against `schema/trace.schema.json`; scorers (benchmark, cost, trajectory,
  fairness, harness); bootstrap aggregation; Postgres + Grafana dashboard; `--max-usd` spend caps; disk
  guards (`hm doctor`, `hm clean`, [docs/disk.md](docs/disk.md)).
- **Columns.** `raw_loop`, `claude_sdk` (Claude Code), `opencode`, `harness2` (frozen refined tree,
  `hm refine`), `harness2_online` (per-task Harness², `hm refine-online`).
- **Benchmarks.** AgentDojo `workspace` (benign, `important_instructions`, `harness_editor_note`);
  τ² `airline`/`retail`/`telecom` with an LLM or scripted customer; the custom Docker safety suite
  (20 tasks, [docs/custom_suite.md](docs/custom_suite.md)); toy suites for the zero-cost demo.
- **Modes.** `out_of_the_box` and `equalized_prompt`; the mode delta is measured on `claude_sdk`.
- **Live cells.** The AgentDojo matrix for all five columns (N=5, benign and attacked); τ² airline for
  Claude Code, OpenCode and online Harness²; the custom suite on Claude Code; the Harness² k=1 and k=2
  refinements; the two poisoning arms and their transfer check.

Not done:

- Network-level egress allow-list for the OpenCode-based columns (today: web tools disabled and the
  OpenCode permission fence only).
- τ² `retail`/`telecom` with an LLM customer; `raw_loop` and the frozen `harness2` tree on τ²; a second
  customer model beyond Haiku.
- `equalized_prompt` on the key columns; Claude Code on a Console key instead of the plan login.
- A second model row, only once the harness axis has shown signal.
- Grafana visual pass in a browser.

Ideas, not scheduled: a `harbor_agent` column, an Inspect `Solver` wrapper, a fifth harness mounted
directly (Deep Agents, Hermes).
