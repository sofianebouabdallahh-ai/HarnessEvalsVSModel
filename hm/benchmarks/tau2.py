"""τ²-bench mount (Sierra), pinned to v1.0.1 (D4): programmatic grading only.

Suites: `tau2/airline`, `tau2/retail`, `tau2/telecom` (first HM_TAU2_MAX_TASKS tasks of the
domain's default split that have a programmatic reward component and whose initial state we can
replay).

Reward basis (D13). Only programmatic components are graded: DB, ENV_ASSERTION, ACTION,
COMMUNICATE. In v1.0.1 airline is [DB, COMMUNICATE] and telecom is [ENV_ASSERTION, ACTION], so
both are graded exactly as τ² grades them. Retail is [DB, NL_ASSERTION] on 112 of 114 tasks; the
NL assertions need an LLM judge, which this platform refuses on purpose, so retail
`success` means *DB state matches* and `env_final["grades"]["reward_basis_dropped"]` records
["NL_ASSERTION"]. That is a more lenient definition than the τ² leaderboard's and is applied
identically to every harness column, which is what the matrix needs; it is not comparable to
published retail numbers and docs/tau2.md says so.

Customer tools (telecom). The τ² telecom customer has tools of their own (airplane mode, SIM,
speed test…). They are never exposed to the harness: the user simulator gets them, its tool calls
go to the environment with requestor="user" and the results go back to the simulator, exactly as
τ²'s orchestrator does, and all of it lands in the transcript so τ²'s env-assertion replay sees it.

How a harness talks to the simulated customer
---------------------------------------------
τ²'s orchestrator alternates agent and user turns and executes the agent's tool calls itself.
External harnesses (Claude Code, OpenCode) execute their own tools and have no notion of a
counterpart that speaks back, so the env-server owns the whole τ² side: the domain environment,
the running transcript, and the LLM user simulator. The domain tools are served over MCP exactly
as τ² hands them to agents, plus ONE environment tool:

    send_message_to_user(message: str) -> str   # the customer's reply, or a reply containing ###STOP###

That tool is part of the *environment*, identical for every harness, and is how a turn-based
benchmark becomes reachable from any harness without harness-specific plumbing. The agent's
greeting is τ²'s fixed first message ("Hi! How can I help you today?"); the customer's opening
utterance is generated when the env is built and appended to the task prompt via
`prompt_addendum()`, so every harness starts from the same bytes for a given episode.

Grading happens in `snapshot()` while the env is live (D11), with τ²'s own evaluator
(`evaluate_simulation`, EvaluationType.ALL): DB-state replay → `db_match`, substring communicate
check → `communicate_match`, product over the task's `reward_basis` → `success`. If the harness
returns without the customer having said ###STOP###, its final text is delivered to the customer
once more; the episode then ends as USER_STOP or AGENT_STOP, both of which τ² grades.

User simulator (the confound): `HM_TAU2_USER_SIM=llm` (default) uses τ²'s
`UserSimulator` through LiteLLM with `HM_TAU2_USER_SIM_MODEL`; `mock` is a deterministic
scripted customer for offline tests. Backend and model string are recorded in `env_final`.
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from hm.benchmarks.base import Benchmark, Env, register
from hm.types import Score, Task, ToolDef, ToolResult, Trace, utcnow

VERSION = "v1.0.1"  # also pinned in uv.lock (git rev fc0055dc)
DATA_URL = f"https://github.com/sierra-research/tau2-bench/archive/refs/tags/{VERSION}.tar.gz"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
MAX_TASKS = int(os.environ.get("HM_TAU2_MAX_TASKS", "20"))
USER_SIM_MODEL = os.environ.get("HM_TAU2_USER_SIM_MODEL", "claude-sonnet-5-5")
SUITES = ["airline", "retail", "telecom"]
SEND_TOOL = "send_message_to_user"


def ensure_data_dir() -> Path:
    """τ² ships its data (DBs, policies, tasks, user-sim guidelines; ~730 MB unpacked) in the git
    repo, not in the wheel. Honour TAU2_DATA_DIR; otherwise fetch the pinned release tarball once."""
    if os.environ.get("TAU2_DATA_DIR"):
        return Path(os.environ["TAU2_DATA_DIR"])
    data = CACHE_DIR / f"tau2-bench-{VERSION.lstrip('v')}" / "data"
    if not (data / "tau2" / "domains").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=600) as r:  # noqa: S310 (pinned github url)
            buf = io.BytesIO(r.read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if "/data/" in m.name or m.name.endswith("/data")]
            tf.extractall(CACHE_DIR, members=members, filter="data")
    return data


# tau2 resolves DATA_DIR at import time, so this must run before the first `import tau2`.
os.environ.setdefault("TAU2_DATA_DIR", str(ensure_data_dir()))

from tau2.config import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason  # noqa: E402
from tau2.data_model.tasks import RewardType  # noqa: E402
from tau2.data_model.tasks import Task as T2Task  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # noqa: E402
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402
from tau2.user.user_simulator_base import STOP  # noqa: E402

try:  # τ² logs every tool response at DEBUG through loguru; keep episode logs readable
    from loguru import logger as _tau2_logger

    _tau2_logger.disable("tau2")
except ImportError:  # pragma: no cover
    pass

PROGRAMMATIC_BASES = {RewardType.DB, RewardType.COMMUNICATE, RewardType.ENV_ASSERTION, RewardType.ACTION}

AGENT_INSTRUCTIONS = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    f"You talk to the customer only through the `{SEND_TOOL}` tool: its result is the customer's "
    "reply. In each turn you can either send a message to the customer or make a tool call.\n"
    f"The conversation is over when the customer's reply contains {STOP}; stop then.\n"
    "Try to be helpful and always follow the policy."
)


# --------------------------------------------------------------------------- user simulators


class ScriptedUser:
    """Deterministic offline customer: states the reason for the call, then the task instructions,
    then hangs up. Exercises the env-server/driver/grading path with no LLM."""

    backend = "mock"
    model = "scripted"

    def __init__(self, task: T2Task):
        ins = task.user_scenario.instructions
        parts = [getattr(ins, "reason_for_call", None), getattr(ins, "known_info", None)]
        first = "\n".join(p for p in parts if p) or str(ins)
        self._script = [
            first,
            getattr(ins, "task_instructions", None) or "That is all.",
            f"Thanks, bye. {STOP}",
        ]
        self._i = 0

    def reply(self, agent_message: AssistantMessage) -> UserMessage:
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return UserMessage(role="user", content=text)


class LLMUser:
    """τ²'s own UserSimulator (LiteLLM). `llm_args` is empty: the Messages API exposes no
    temperature any more (D12), so τ²'s default {"temperature": 0.0} would be rejected.
    `tools` are the domain's *user* tools (telecom); the simulator then answers with tool calls
    whose requestor is "user" and expects the ToolMessage back (see Tau2Env._user_turn)."""

    backend = "llm"

    def __init__(self, task: T2Task, model: str, tools: list | None = None):
        self.model = model
        self._sim = UserSimulator(
            llm=model, instructions=str(task.user_scenario), tools=tools or None, llm_args={}
        )
        self._state = self._sim.get_init_state()

    def reply(self, message: AssistantMessage | ToolMessage | MultiToolMessage) -> UserMessage:
        msg, self._state = self._sim.generate_next_message(message, self._state)
        return msg


def make_user(task: T2Task, tools: list | None = None):
    kind = os.environ.get("HM_TAU2_USER_SIM", "llm")
    if kind == "mock":
        return ScriptedUser(task)
    if kind == "llm":
        if not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            raise RuntimeError(
                "tau2 user simulator needs an API key (ANTHROPIC_API_KEY/OPENAI_API_KEY); "
                "set HM_TAU2_USER_SIM=mock for the offline scripted customer"
            )
        return LLMUser(task, USER_SIM_MODEL, tools)
    raise ValueError(f"HM_TAU2_USER_SIM={kind!r}; expected llm|mock")


# --------------------------------------------------------------------------- env


@lru_cache
def _tasks(domain: str) -> list[T2Task]:
    return list(registry.get_tasks_loader(domain)())


def programmatic_basis(t: T2Task) -> list[RewardType]:
    ec = t.evaluation_criteria
    return [b for b in (ec.reward_basis if ec else []) if b in PROGRAMMATIC_BASES]


def _supported(t: T2Task) -> bool:
    if not programmatic_basis(t):
        return False
    hist = t.initial_state.message_history if t.initial_state and t.initial_state.message_history else []
    if not hist:
        return True
    last = hist[-1]
    return isinstance(last, AssistantMessage | UserMessage) and not last.is_tool_call()


class Tau2Env(Env):
    def __init__(self, domain: str, task_id: str):
        self.domain = domain
        self.task = next(t for t in _tasks(domain) if t.id == task_id)
        self.env = registry.get_env_constructor(domain)()
        init = self.task.initial_state
        history = deepcopy(init.message_history) if init and init.message_history else []
        self.env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=history,
        )
        try:
            self.user_tools = list(self.env.get_user_tools())
        except ValueError:  # domain has no customer-side tools
            self.user_tools = []
        self.user = make_user(self.task, self.user_tools)
        self.trajectory: list = list(history)
        self.terminated: TerminationReason | None = None
        self.errors = 0
        self._n_calls = 0
        self._grades: dict[str, Any] | None = None
        # Opening exchange, as τ²'s orchestrator does it: fixed agent greeting, then the customer.
        if history and isinstance(history[-1], UserMessage):
            self.opening = history[-1].content or ""
        else:
            if not history:
                first = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                self.trajectory.append(first)
            self.opening = self._user_turn(self.trajectory[-1]).content or ""

    # ---- τ² transcript helpers

    def _user_turn(self, agent_message: AssistantMessage) -> UserMessage:
        reply = self.user.reply(agent_message)
        reply.validate()
        self.trajectory.append(reply)
        # The customer may use their own tools (telecom) before answering: USER -> ENV -> USER, as in
        # τ²'s orchestrator. Results carry requestor="user" so the evaluator's replay attributes them.
        while reply.is_tool_call() and self.terminated is None:
            results = [self.env.get_response(tc) for tc in reply.tool_calls]
            self.trajectory.extend(results)
            self._check_steps()
            if self.terminated is not None:
                break
            nxt = MultiToolMessage(role="tool", tool_messages=results) if len(results) > 1 else results[0]
            reply = self.user.reply(nxt)
            reply.validate()
            self.trajectory.append(reply)
        if UserSimulator.is_stop(reply):
            self.terminated = TerminationReason.USER_STOP
        self._check_steps()
        return reply

    def _check_steps(self) -> None:
        if self.terminated is None and len(self.trajectory) >= DEFAULT_MAX_STEPS:
            self.terminated = TerminationReason.MAX_STEPS

    # ---- Env protocol

    def prompt_addendum(self) -> str | None:
        greeting = DEFAULT_FIRST_AGENT_MESSAGE.content
        return f"\n\n<conversation>\nAgent: {greeting}\nCustomer: {self.opening}\n</conversation>"

    def tools(self) -> list[ToolDef]:
        defs = []
        for t in self.env.get_tools():
            fn = t.openai_schema["function"]
            defs.append(
                ToolDef(
                    name=fn["name"], description=fn.get("description") or "", input_schema=fn["parameters"]
                )
            )
        defs.append(
            ToolDef(
                name=SEND_TOOL,
                description="Send a message to the customer and wait for their reply. The result is the "
                f"customer's next message; a reply containing {STOP} means the conversation is over.",
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        )
        return defs

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.terminated is not None:
            return ToolResult(content="", error=f"conversation has ended ({self.terminated.value})")
        self._n_calls += 1
        if name == SEND_TOOL:
            text = str(args.get("message", ""))
            msg = AssistantMessage(role="assistant", content=text)
            msg.validate()
            self.trajectory.append(msg)
            return ToolResult(content=self._user_turn(msg).content or "")
        tc = ToolCall(id=f"call_{self._n_calls}", name=name, arguments=dict(args), requestor="assistant")
        msg = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
        self.trajectory.append(msg)
        tm: ToolMessage = self.env.get_response(tc)
        self.trajectory.append(tm)
        if tm.error:
            self.errors += 1
            if self.errors >= DEFAULT_MAX_ERRORS:
                self.terminated = TerminationReason.TOO_MANY_ERRORS
        self._check_steps()
        return ToolResult(
            content=tm.content or "", error=(tm.content or "tool error")[:200] if tm.error else None
        )

    def _finish(self, final_output: str | None) -> None:
        if self.terminated is not None:
            return
        if final_output and final_output.strip():
            msg = AssistantMessage(role="assistant", content=final_output)
            msg.validate()
            self.trajectory.append(msg)
            self._user_turn(msg)
        if self.terminated is None:
            self.terminated = TerminationReason.AGENT_STOP if self._n_calls else TerminationReason.AGENT_ERROR

    def grade(self, final_output: str | None) -> dict[str, Any]:
        if self._grades is not None:
            return self._grades
        self._finish(final_output)
        now = utcnow().isoformat()
        sim = SimulationRun(
            id="hm", task_id=self.task.id, start_time=now, end_time=now, duration=0.0,
            termination_reason=self.terminated, messages=list(self.trajectory),
        )  # fmt: skip
        prog = programmatic_basis(self.task)
        full = list(self.task.evaluation_criteria.reward_basis) if self.task.evaluation_criteria else []
        dropped = [b.value for b in full if b not in PROGRAMMATIC_BASES]
        task = self.task
        if dropped:  # D13: grade the programmatic components only; τ² multiplies over the basis given
            task = deepcopy(self.task)
            task.evaluation_criteria.reward_basis = prog
        ri: RewardInfo = evaluate_simulation(
            sim, task, EvaluationType.ALL, solo_mode=False, domain=self.domain
        )
        basis = set(prog)
        db = ri.db_check.db_match if ri.db_check is not None else None
        comm = (
            all(c.met for c in ri.communicate_checks)
            if ri.communicate_checks
            else (True if ri.reward_basis else None)
        )
        env_ok = all(a.met for a in ri.env_assertions) if ri.env_assertions is not None else None
        act_ok = all(a.action_match for a in ri.action_checks) if ri.action_checks is not None else None
        self._grades = {
            "success": ri.reward >= 1.0,
            "db_match": db if RewardType.DB in basis else None,
            "communicate_match": comm if RewardType.COMMUNICATE in basis else None,
            "env_assertions_match": env_ok if RewardType.ENV_ASSERTION in basis else None,
            "action_match": act_ok if RewardType.ACTION in basis else None,
            "reward": ri.reward,
            "reward_basis": [b.value for b in prog],
            "reward_basis_dropped": dropped,
            "termination_reason": self.terminated.value,
            "reward_info": ri.model_dump(mode="json"),
        }
        return self._grades

    def snapshot(self, final_output: str | None = None) -> dict[str, Any]:
        grades = self.grade(final_output)
        return {
            "domain": self.domain,
            "task_id": self.task.id,
            "tau2_version": VERSION,
            "grades": grades,
            "user_sim": {
                "backend": self.user.backend,
                "model": self.user.model,
                "tools": [t.name for t in self.user_tools],
            },
            "environment_tools_extra": [SEND_TOOL],
            "messages": [m.model_dump(mode="json") for m in self.trajectory],
        }

    def state_hash(self) -> str:
        return self.env.get_db_hash() or ""


class Tau2Benchmark(Benchmark):
    name = "tau2"

    def suites(self) -> list[str]:
        return SUITES

    def tasks(self, suite: str) -> list[Task]:
        policy = registry.get_env_constructor(suite)().get_policy()
        prompt = f"<instructions>\n{AGENT_INSTRUCTIONS}\n</instructions>\n<policy>\n{policy}\n</policy>"
        out = []
        for t in _tasks(suite):
            if not _supported(t):
                continue
            out.append(
                Task(
                    id=f"tau2/{suite}/{t.id}",
                    benchmark=self.name,
                    suite=suite,
                    prompt=prompt,
                    meta={"domain": suite, "tau2_task_id": t.id, "version": VERSION},
                )
            )
            if len(out) >= MAX_TASKS:
                break
        return out

    def make_env(self, task: Task) -> Env:
        return Tau2Env(task.meta["domain"], task.meta["tau2_task_id"])

    def score(self, task: Task, trace: Trace) -> list[Score]:
        g = (trace.env_final or {}).get("grades")
        if g is None:
            return [Score(scorer="tau2", name="success", value=None, explanation="no grades in env_final")]
        label = f"tau2 {VERSION} · {g['termination_reason']}"
        scores = [
            Score(
                scorer="tau2",
                name="success",
                value=float(bool(g["success"]) and trace.error is None),
                label=label,
            )
        ]
        for k in ("db_match", "communicate_match", "env_assertions_match", "action_match"):
            if g.get(k) is not None:
                scores.append(Score(scorer="tau2", name=k, value=float(g[k])))
        if g.get("reward_basis_dropped"):  # D13: visible on every retail score row
            scores[0].explanation = f"graded on {g['reward_basis']}; dropped {g['reward_basis_dropped']}"
        return scores


register(Tau2Benchmark())
