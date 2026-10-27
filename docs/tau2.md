# τ²-bench mount

Pinned to **v1.0.1** (git rev `fc0055dc`, via `uv.lock`). τ² 2.0 switched the communicate check to an LLM judge and is not comparable to the 1.x leaderboard, so 1.0.1 stays (decision D4). Suites: `tau2/airline`, `tau2/retail`, `tau2/telecom`; the first `HM_TAU2_MAX_TASKS` (20) tasks of each domain's default split whose `reward_basis` is fully programmatic (DB / COMMUNICATE / ENV_ASSERTION / ACTION). In airline v1.0.1 that is every task (`reward_basis = [DB, COMMUNICATE]`; the `nl_assertions` present on tasks are diagnostic and are not evaluated).

## Data

τ² keeps its data (domain DBs, policies, tasks, user-simulator guidelines; ~730 MB unpacked) in the repository, not in the wheel. `hm/benchmarks/tau2.py` honours `TAU2_DATA_DIR`; otherwise it downloads the v1.0.1 release tarball (~90 MB) once into `.cache/` and extracts only `data/`.

## `send_message_to_user`: how a turn-based benchmark reaches any harness

τ²'s orchestrator alternates agent and customer turns and executes the agent's tool calls itself. Claude Code, OpenCode and friends execute their own tools and have no notion of a counterpart that talks back. Rather than teaching each adapter a conversation protocol (session resume, turn splicing), the env-server owns the whole τ² side — environment, transcript, user simulator — and exposes one extra tool next to the domain tools:

```
send_message_to_user(message: str) -> str   # the customer's reply; contains ###STOP### when they hang up
```

Properties:
- It is an **environment** tool: identical schema and behaviour for every harness, listed in `env_final["environment_tools_extra"]`. Adapters do not know τ² exists.
- The agent's greeting is τ²'s fixed first message (`"Hi! How can I help you today?"`), as in τ²'s own orchestrator. The customer's opening utterance is generated when the env is built and appended to the task prompt through `Env.prompt_addendum()`, so within an episode every harness starts from byte-identical input.
- The transcript is built with τ²'s own message types (`AssistantMessage`, `UserMessage`, `ToolMessage` from `Environment.get_response`), so τ²'s DB-state evaluator can replay it strictly.
- τ²'s limits are mirrored: `DEFAULT_MAX_STEPS` (200 messages) and `DEFAULT_MAX_ERRORS` (10 tool errors) terminate the conversation; after termination every tool call is refused with an error.
- If the harness returns without the customer having said `###STOP###`, its final text is delivered to the customer once more; the episode then ends as `USER_STOP` or `AGENT_STOP`. Both are gradable in τ² (an agent may end a conversation). An episode that never spoke or acted ends as `AGENT_ERROR` and scores 0, as in τ².

Deviation from τ²'s baseline agent: τ² agents receive the policy as a system prompt and may send either text or a tool call per turn. Here the policy and the protocol are in the task prompt (identical for all harnesses) and a text turn is a tool call. The `equalized_prompt` mode measures how much of any gap is the system prompt.

## Grading (decision D11)

`Tau2Env.snapshot(final_output)` runs τ²'s `evaluate_simulation(…, EvaluationType.ALL)` on the live environment and stores the verdicts in `env_final["grades"]`:

| score | from |
|---|---|
| `db_match` | `RewardInfo.db_check.db_match` — golden actions replayed on a fresh env vs. the agent's transcript replayed on a fresh env, agent and user DB hashes both equal |
| `communicate_match` | all `communicate_checks.met` — case-insensitive substring, commas stripped (the v1.0.1 scorer) |
| `success` | `RewardInfo.reward == 1.0` — product over the task's `reward_basis`; additionally requires no harness-level error |

Note τ²'s semantics: a task whose golden actions are all reads (refusal tasks such as airline/0 and airline/1) is passed by an agent that does nothing and talks politely. That is the benchmark's definition, not ours.

## The user-simulator confound

The customer is an LLM. Its model is a hidden variable that moves every cell equally only if it is held fixed, so it is pinned and recorded:

- `HM_TAU2_USER_SIM=llm` (default): τ²'s `UserSimulator` through LiteLLM, model `HM_TAU2_USER_SIM_MODEL` (default `claude-sonnet-5-5`), **`llm_args={}`** because the Messages API no longer accepts `temperature` (D12; τ²'s own default of `{"temperature": 0.0}` would be rejected). Needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
- `HM_TAU2_USER_SIM=mock`: a deterministic scripted customer (reason for call → task instructions → `###STOP###`) for offline tests and the zero-cost demo.

Backend and model string are written to `env_final["user_sim"]` on every episode. The one-swap ablation (same matrix, different user-sim model) is a one-variable change.

## Reward basis per domain (D13)

Only programmatic reward components are graded: `DB`, `ENV_ASSERTION`, `ACTION`, `COMMUNICATE`. Per domain in v1.0.1:

| suite | τ² `reward_basis` | graded here | note |
|---|---|---|---|
| `airline` | `[DB, COMMUNICATE]` (50/50 tasks) | the same | identical to τ²'s own grading |
| `telecom` | `[ENV_ASSERTION, ACTION]` (114/114) | the same | identical to τ²'s own grading; the customer's own tool calls are part of the replay |
| `retail` | `[DB, NL_ASSERTION]` on 112/114 tasks | `[DB]` | the NL assertions need an LLM judge, which this platform refuses on purpose. `success` therefore means *final DB state matches*; `env_final["grades"]["reward_basis_dropped"] == ["NL_ASSERTION"]` and the `success` score's explanation say so on every episode |

The retail definition is more lenient than the τ² leaderboard's and **is not comparable to published retail numbers**. It is applied identically to every harness column, which is what the matrix needs. Any write-up that quotes a retail cell must say "DB-match only".

## Customer tools (telecom)

The telecom customer has 30 tools of their own (`check_status_bar`, `toggle_airplane_mode`, `reseat_sim_card`, …). They are never served to the harness: `Tau2Env.tools()` lists the assistant toolkit plus `send_message_to_user` only. The simulator receives them (`UserSimulator(tools=…)`); when it answers with a tool call the env-server executes it with `requestor="user"`, appends the `ToolMessage` to the transcript and hands it back to the simulator until it produces text — the orchestrator's USER → ENV → USER leg. The transcript therefore contains the customer's actions, and τ²'s `EnvironmentEvaluator` replays them when it checks the env assertions. `env_final["user_sim"]["tools"]` lists what the customer could do. The scripted `mock` customer never uses tools; tests cover the tool leg with a scripted tool-using double.

## Smoke run with the scripted customer (2026-10-25)

Run `20261025T021257Z-d9d8`: `claude_sdk`, five `telecom` + five `retail` tasks, `HM_TAU2_USER_SIM=mock`,
0 errors. Everything on the platform side held: the telecom customer's 30 tools were served to the
simulator and never to the harness, every episode ended with `user_stop` or `agent_stop`, retail
recorded `reward_basis_dropped=["NL_ASSERTION"]` and was graded on `DB`, telecom on `ENV_ASSERTION`.
Success was 0/10 **by construction**: the scripted customer answers with its scenario text and hangs up
on the second turn, so it never confirms an exchange or reports what its phone shows, and τ² policy
forbids acting without confirmation. Claude Code did the right thing in every episode (looked up the
account, proposed the exchange or asked the customer to check airplane mode, made no change without a
yes). The run is kept under `runs/` as a plumbing check and is deliberately **not** loaded into
Postgres; the first real τ² cells need the LLM customer, which needs an API key.

## First real cell — `claude_sdk` × `airline` with an LLM customer (2026-10-25)

Run `20261025T031522Z-d067`: 20 airline tasks × 5 seeds = 100 episodes, `claude_sdk` on the plan login,
customer = τ²'s `UserSimulator` on **`claude-haiku-5-5`** through LiteLLM on the Console key (the only
billed side). Loaded into Postgres.

| metric | value |
|---|---|
| pass^1 | **0.72** [0.56, 0.87] |
| pass^5 | 0.50 [0.34, 0.69] |
| `db_match` / `communicate_match` (means) | 0.76 / 0.90 |
| terminations | 52 `user_stop`, 48 `agent_stop` |
| messages per conversation | mean 22.5, max 64 |
| errors | 1 (`max_turns` = 30 reached on the 64-message conversation) |
| API refusals | 0 |
| agent cost per solve (price table; on the plan, not billed) | $0.07 |

Reading it: the conversations are real — twenty-plus messages, the customer pushes back, half the
time the customer hangs up and half the time the agent closes. Failures concentrate on tasks 7 and 19
(0/5), 16 (1/5), 8 and 15 (2/5); `db_match` is the component that fails, `communicate_match` rarely.
This number is **not comparable to the τ² leaderboard**: the customer is Haiku 5.5 rather than τ²'s
reference simulator, grading is the programmatic subset (D13; airline is unaffected), and N=5 over 20
tasks gives a wide interval. It is comparable to the other columns of this matrix once they run with
the same customer, which is the point.

Against the scripted-customer smoke (0/10 by construction) this is also the first user-simulator swap:
the same harness, same tasks, mock → Haiku, 0.00 → 0.72. A second LLM customer (Sonnet) is the
ablation the spec asks for and waits on budget.

One scorer fix came out of this run: in 5 episodes the model first called the bare name
`send_message_to_user` (the prompt names the tool that way; the MCP name carries a prefix), Claude
Code answered "No such tool available", and the model recovered. The fairness check had counted that
as an undeclared native tool and excluded the episodes; a call the harness rejects as nonexistent no
longer taints.

## Not yet done

- `airline` on `raw_loop` and the frozen `harness2` tree with the same Haiku customer (`opencode` and online Harness² are done, see [replication_note.md](replication_note.md)); `retail`/`telecom` with an LLM customer.
- A second LLM customer (Sonnet) on one column — the user-sim swap beyond mock → Haiku.
