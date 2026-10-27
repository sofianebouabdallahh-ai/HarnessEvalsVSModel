"""τ²-bench mount, offline: scripted customer, tools over MCP, τ² grading semantics from the trace.
First run downloads the pinned v1.0.1 data tarball (~90 MB) into .cache/ unless TAU2_DATA_DIR is set."""

import os

os.environ["HM_TAU2_USER_SIM"] = "mock"

import pytest  # noqa: E402

import hm.benchmarks.tau2 as tau2_mod  # noqa: E402
from hm.benchmarks import get_benchmark  # noqa: E402
from hm.benchmarks.tau2 import SEND_TOOL, Tau2Env  # noqa: E402

ToolCall, ToolMessage, UserMessage, STOP = (
    tau2_mod.ToolCall,
    tau2_mod.ToolMessage,
    tau2_mod.UserMessage,
    tau2_mod.STOP,
)
from hm.envserver import EnvServer, McpToolClient  # noqa: E402
from hm.types import AdapterManifest, Limits, Trace  # noqa: E402

bench = get_benchmark("tau2")
MUTATING = {"book_reservation", "cancel_reservation", "send_certificate"}


def _trace(task, env: Tau2Env, final_output=None, error=None) -> Trace:
    tr = Trace(
        run_id="t", task_id=task.id, benchmark="tau2", suite=task.suite, model="m", harness="h",
        harness_version="0", seed=0, adapter_manifest=AdapterManifest(system_prompt_sha256=None),
        final_output=final_output, env_final=env.snapshot(final_output), env_state_hash=env.state_hash(), error=error,
    )  # fmt: skip
    return Trace.model_validate_json(tr.model_dump_json())  # grades must survive the JSON trace


def _scores(task, env, **kw) -> dict:
    return {s.name: s.value for s in bench.score(task, _trace(task, env, **kw))}


def _task_with_mutating_golden_actions():
    for task in bench.tasks("airline"):
        env = bench.make_env(task)
        if any(
            a.name in MUTATING or a.name.startswith("update_") for a in env.task.evaluation_criteria.actions
        ):
            return task, env
    pytest.skip("no airline task with mutating golden actions in the first HM_TAU2_MAX_TASKS")


def test_suites_tasks_and_prompt():
    assert bench.suites() == ["airline", "retail", "telecom"]
    tasks = bench.tasks("airline")
    assert 0 < len(tasks) <= 20 and all(not t.attacked for t in tasks)
    assert "<policy>" in tasks[0].prompt and SEND_TOOL in tasks[0].prompt and "###STOP###" in tasks[0].prompt


@pytest.mark.asyncio
async def test_tools_and_opening_served_over_mcp():
    task = bench.tasks("airline")[0]
    env = bench.make_env(task)
    assert "<conversation>" in env.prompt_addendum() and "Customer:" in env.prompt_addendum()
    with EnvServer(env, Limits()) as srv:
        async with McpToolClient(srv.url) as c:
            tools = {t.name: t for t in await c.list_tools()}
            assert SEND_TOOL in tools and "get_user_details" in tools and "cancel_reservation" in tools
            assert tools["book_reservation"].input_schema["type"] == "object"
            text, err = await c.call(SEND_TOOL, {"message": "I understand, let me check."})
            assert not err and text  # the customer's second utterance (task instructions)
            text, err = await c.call(SEND_TOOL, {"message": "Anything else?"})
            assert not err and "###STOP###" in text
            text, err = await c.call("get_user_details", {"user_id": "emma_kim_9957"})
            assert err and "conversation has ended" in text
    steps = [s for _, s in srv.log]
    assert steps[0].name == SEND_TOOL and steps[1].content


def test_do_nothing_is_graded_by_tau2_semantics():
    """A task whose golden actions are all reads passes with no agent action (τ² grades end state),
    a task with mutating golden actions does not."""
    read_only = bench.tasks("airline")[0]
    env = bench.make_env(read_only)
    env.call(SEND_TOOL, {"message": "Sorry, I cannot help with that."})
    env.call(SEND_TOOL, {"message": "Goodbye."})
    s = _scores(read_only, env)
    assert s["success"] == 1.0 and s["db_match"] == 1.0 and s["communicate_match"] == 1.0
    assert env.terminated.value == "user_stop"

    task, env = _task_with_mutating_golden_actions()
    env.call(SEND_TOOL, {"message": "Sorry."})
    env.call(SEND_TOOL, {"message": "Goodbye."})
    s = _scores(task, env)
    assert s["db_match"] == 0.0 and s["success"] == 0.0


def test_golden_actions_then_stop_match_db():
    task, env = _task_with_mutating_golden_actions()
    for a in env.task.evaluation_criteria.actions:
        r = env.call(a.name, dict(a.arguments))
        assert r.error is None, r.error
    env.call(SEND_TOOL, {"message": "Done."})
    env.call(SEND_TOOL, {"message": "Done."})
    s = _scores(task, env)
    assert s["db_match"] == 1.0
    assert s["success"] == float(s.get("communicate_match", 1.0) == 1.0)


def test_communicate_substring_check():
    task = next(
        t for t in bench.tasks("airline") if t.meta["tau2_task_id"] == "3"
    )  # communicate_info == ['4']
    env = bench.make_env(task)
    env.call(SEND_TOOL, {"message": "There are 4 of them."})
    env.call(SEND_TOOL, {"message": "Bye."})
    assert _scores(task, env)["communicate_match"] == 1.0
    env2 = bench.make_env(task)
    env2.call(SEND_TOOL, {"message": "I cannot tell you."})
    env2.call(SEND_TOOL, {"message": "Bye."})
    assert _scores(task, env2)["communicate_match"] == 0.0


def test_agent_ending_without_stop_is_agent_stop_and_harness_error_fails():
    task = bench.tasks("airline")[0]
    env = bench.make_env(task)
    env.call("get_user_details", {"user_id": "emma_kim_9957"})
    s = _scores(task, env, final_output="I have looked it up; nothing to do.")
    # final text is delivered once more; scripted customer still had a turn before STOP → agent ended it
    assert env.terminated.value in ("agent_stop", "user_stop") and s["success"] == 1.0
    env3 = bench.make_env(task)
    assert _scores(task, env3, error="crash")["success"] == 0.0


def test_env_final_records_user_sim_and_messages():
    task = bench.tasks("airline")[0]
    env = bench.make_env(task)
    snap = env.snapshot(None)
    assert snap["user_sim"] == {
        "backend": "mock",
        "model": "scripted",
        "tools": [],
    }  # airline: no customer tools
    assert snap["environment_tools_extra"] == [SEND_TOOL] and snap["tau2_version"] == "v1.0.1"
    assert (
        snap["messages"][0]["content"] == "Hi! How can I help you today?"
        and snap["messages"][1]["role"] == "user"
    )


# ---- retail (D13) and telecom (customer tools)


def test_every_suite_loads_and_grades_do_nothing():
    for suite in bench.suites():
        tasks = bench.tasks(suite)
        assert 0 < len(tasks) <= 20, suite
        env = bench.make_env(tasks[0])
        assert "<conversation>" in env.prompt_addendum()
        env.call(SEND_TOOL, {"message": "Sorry."})
        env.call(SEND_TOOL, {"message": "Goodbye."})
        snap = env.snapshot(None)
        g = snap["grades"]
        assert g["termination_reason"] in ("user_stop", "agent_stop"), suite
        assert set(g["reward_basis"]) <= {"DB", "COMMUNICATE", "ENV_ASSERTION", "ACTION"}, suite
        assert isinstance(g["success"], bool)


def test_retail_is_graded_on_db_only_and_says_so():
    tasks = bench.tasks("retail")
    assert len(tasks) == 20  # 112/114 retail tasks carry NL_ASSERTION; D13 keeps them on their DB component
    with_nl = {
        t.id
        for t in tau2_mod._tasks("retail")
        if any(b.value == "NL_ASSERTION" for b in t.evaluation_criteria.reward_basis)
    }
    task = next(t for t in tasks if t.meta["tau2_task_id"] in with_nl)
    env = bench.make_env(task)
    assert env.user_tools == []
    env.call(SEND_TOOL, {"message": "Sorry."})
    env.call(SEND_TOOL, {"message": "Goodbye."})
    s = {sc.name: sc for sc in bench.score(task, _trace(task, env))}
    g = env.snapshot(None)["grades"]
    assert g["reward_basis_dropped"] == ["NL_ASSERTION"] and g["reward_basis"] == ["DB"]
    assert g["reward_info"]["nl_assertions"] is None  # no LLM judge ran
    assert "dropped ['NL_ASSERTION']" in s["success"].explanation
    assert "db_match" in s and "communicate_match" not in s


class _ToolOnceUser:
    """Customer who checks their phone (a user tool) before describing the problem, then hangs up."""

    backend, model = "mock", "scripted-tools"

    def __init__(self):
        self.seen = []

    def reply(self, message):
        self.seen.append(type(message).__name__)
        n = len(self.seen)
        if n == 1:
            return UserMessage(
                role="user",
                content=None,
                tool_calls=[ToolCall(id="u1", name="check_status_bar", arguments={}, requestor="user")],
            )
        if n == 2:
            assert isinstance(message, ToolMessage) and message.requestor == "user"
            return UserMessage(role="user", content=f"My phone shows no data. Status: {message.content[:40]}")
        return UserMessage(role="user", content=f"Thanks, bye. {STOP}")


def test_telecom_customer_tools_go_to_the_env_and_back_to_the_customer(monkeypatch):
    task = bench.tasks("telecom")[0]
    user = _ToolOnceUser()
    monkeypatch.setattr(tau2_mod, "make_user", lambda t, tools=None: user)
    env = bench.make_env(task)
    assert len(env.user_tools) >= 20 and "toggle_airplane_mode" in {t.name for t in env.user_tools}
    agent_tools = {t.name for t in env.tools()}
    assert (
        "toggle_airplane_mode" not in agent_tools and SEND_TOOL in agent_tools
    )  # never exposed to the harness
    kinds = [
        type(m).__name__ + (f":{m.requestor}" if isinstance(m, ToolMessage) else "") for m in env.trajectory
    ]
    # greeting, customer tool call, tool result (requestor=user), customer's opening text
    assert kinds[:4] == ["AssistantMessage", "UserMessage", "ToolMessage:user", "UserMessage"]
    assert "no data" in env.opening and user.seen == ["AssistantMessage", "ToolMessage"]
    env.call(SEND_TOOL, {"message": "Let me look into that."})
    snap = env.snapshot(None)
    g = snap["grades"]
    assert snap["user_sim"]["tools"] and g["termination_reason"] == "user_stop"
    assert set(g["reward_basis"]) <= {"ENV_ASSERTION", "ACTION"} and g["reward_basis_dropped"] == []
    assert g["env_assertions_match"] is not None and g["success"] is False  # nothing was fixed
    # the replayed transcript (with the user's tool call) was accepted by τ²'s env-assertion evaluator
    assert g["reward_info"]["env_assertions"] is not None
