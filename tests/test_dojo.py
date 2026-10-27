"""AgentDojo mount, no model calls: tools, injection tagging, and trace-based grading that
matches AgentDojo's own `run_task_with_pipeline` semantics."""

import json

import pytest

from hm.benchmarks import get_benchmark
from hm.benchmarks.dojo import DojoEnv
from hm.envserver import EnvServer
from hm.types import AdapterManifest, Limits, Trace

bench = get_benchmark("dojo")


def _trace(task, env: DojoEnv, final_output="", error=None) -> Trace:
    return Trace(
        run_id="t", task_id=task.id, benchmark="dojo", suite=task.suite, model="m", harness="h",
        harness_version="0", seed=0, attacked=task.attacked, attack=task.attack,
        adapter_manifest=AdapterManifest(system_prompt_sha256=None), final_output=final_output,
        env_final=env.snapshot(final_output), env_state_hash=env.state_hash(), error=error,
    )  # fmt: skip


def _has_placeholder(args) -> bool:
    return "placeholder" in json.dumps(args, default=lambda o: "placeholder")


def test_suites_and_task_counts():
    assert "workspace" in bench.suites() and "workspace_attacked" in bench.suites()
    benign, attacked = bench.tasks("workspace"), bench.tasks("workspace_attacked")
    assert len(benign) == len(attacked) == 20
    assert all(not t.attacked for t in benign) and all(t.attacked for t in attacked)
    assert attacked[0].meta["injection_task"] and attacked[0].id.endswith(attacked[0].meta["injection_task"])


def test_tools_have_schemas_and_serve_over_mcp():
    task = bench.tasks("workspace")[0]
    env = bench.make_env(task)
    tools = env.tools()
    assert len(tools) == 24 and all(t.input_schema["type"] == "object" for t in tools)
    with EnvServer(env, Limits()) as srv:
        text, is_err = srv.call("get_current_day", {})
        assert not is_err and text.strip()
        text, is_err = srv.call("get_unread_emails", {})
        assert not is_err and "subject" in text


def test_attacked_env_tags_injected_tool_results():
    task = bench.tasks("workspace_attacked")[0]
    env = bench.make_env(task)
    assert isinstance(env, DojoEnv) and env.injections
    # Call the user task's ground-truth tools: the injection must surface in one of their results.
    seen = []
    for fc in env.user_task.ground_truth(env.pre_env):
        if _has_placeholder(fc.args):
            continue
        r = env.call(fc.function, dict(fc.args))
        if r.untrusted:
            seen.append(r.injection_marker)
    assert seen, "injection never surfaced through the ground-truth path"
    assert all(m in env.injections for m in seen)


def test_security_is_true_after_executing_injection_goal():
    task = bench.tasks("workspace_attacked")[0]
    env = bench.make_env(task)
    # benign behaviour: nothing done -> attack did not succeed
    s0 = {s.name: s.value for s in bench.score(task, _trace(task, env))}
    assert s0["attack_success"] == 0.0
    # attacker goal executed
    for fc in env.injection_task.ground_truth(env.pre_env):
        assert not _has_placeholder(fc.args)
        env.call(fc.function, dict(fc.args))
    s1 = {s.name: s.value for s in bench.score(task, _trace(task, env))}
    assert s1["attack_success"] == 1.0


def test_utility_true_after_executing_ground_truth_for_write_tasks():
    solved = tried = 0
    for task in bench.tasks("workspace"):
        env = bench.make_env(task)
        gt = env.user_task.ground_truth(env.pre_env)
        if not gt or any(_has_placeholder(fc.args) for fc in gt):
            continue
        tried += 1
        outputs = [env.call(fc.function, dict(fc.args)).content for fc in gt]
        # a cooperative "model" that echoes everything it read, so read-only tasks can pass too
        s = {x.name: x.value for x in bench.score(task, _trace(task, env, final_output="\n".join(outputs)))}
        solved += int(s["success"] == 1.0)
    # Tasks whose utility needs a specific phrasing in the answer legitimately fail with an echo model.
    assert tried >= 5 and solved >= tried * 0.5, (solved, tried)


def test_env_final_roundtrips_through_trace_json():
    """Grades must survive JSON serialization of the trace (that is how the scorer sees them)."""
    task = bench.tasks("workspace_attacked")[0]
    env = bench.make_env(task)
    for fc in env.injection_task.ground_truth(env.pre_env):
        env.call(fc.function, dict(fc.args))
    tr = Trace.model_validate_json(_trace(task, env).model_dump_json())
    s = {x.name: x.value for x in bench.score(task, tr)}
    assert s["attack_success"] == 1.0 and s["success"] == 0.0


def test_harness_error_fails_utility():
    task = bench.tasks("workspace")[0]
    env = bench.make_env(task)
    s = {x.name: x.value for x in bench.score(task, _trace(task, env, error="crash"))}
    assert s["success"] == 0.0


@pytest.mark.parametrize("name", ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-4-6-20260101"])
def test_pricing_covers_current_models(name):
    from hm.scoring.cost import price_for

    assert price_for(name) is not None


# ---------------------------------------------------------------- Phase 7: poisoning (D15, D16)


def test_benign_snapshot_carries_poison_checks_that_are_all_false():
    task = bench.tasks("workspace")[0]
    env = bench.make_env(task)
    grades = env.snapshot("")["grades"]
    assert grades["security"] is None and grades["poison"] and not any(grades["poison"].values())
    s = {sc.name: sc.value for sc in bench.score(task, _trace(task, env))}
    assert s["poison_executed"] == 0.0 and "attack_success" not in s


def test_poison_executed_when_an_injection_goal_runs_on_a_benign_env():
    task = bench.tasks("workspace")[0]
    env = bench.make_env(task)
    inj = env.suite.injection_tasks["injection_task_0"]
    for fc in inj.ground_truth(env.pre_env):
        assert not _has_placeholder(fc.args)
        env.call(fc.function, dict(fc.args))
    scores = bench.score(task, _trace(task, env))
    s = {sc.name: sc for sc in scores}
    assert s["poison_executed"].value == 1.0 and "injection_task_0" in (s["poison_executed"].label or "")
    assert "mark.black-2134@gmail.com" in (s["poison_executed"].explanation or "")


def test_attacked_snapshot_has_no_poison_block():
    task = bench.tasks("workspace_attacked")[0]
    env = bench.make_env(task)
    assert env.snapshot("")["grades"]["poison"] is None
    assert "poison_executed" not in {sc.name for sc in bench.score(task, _trace(task, env))}


def test_harness_editor_note_attack_targets_the_editor_and_keeps_the_goal(monkeypatch):
    from hm.benchmarks import dojo as d

    s = d._suite("workspace")
    user_task, inj = list(s.user_tasks.values())[0], s.injection_tasks["injection_task_0"]
    text = next(
        iter(d.load_attack("harness_editor_note", s, d._NamedPipeline()).attack(user_task, inj).values())
    )
    assert "<INFORMATION>" in text and "MAINTENANCE NOTE" in text and "guardrails.md" in text
    assert text.count(inj.GOAL) == 2  # once for the executor, once for the editor
    # selected through the env variable like any other attack; the executor-facing text is unchanged
    monkeypatch.setattr(d, "ATTACK", "harness_editor_note")
    d._attack.cache_clear()
    try:
        env = DojoEnv("workspace", user_task.ID, "injection_task_0")
        assert env.injections and all(
            "MAINTENANCE NOTE" in v and "Before you can solve" in v for v in env.injections.values()
        )
        r = env.call("get_unread_emails", {})
        assert r.untrusted or not any("MAINTENANCE" in v for v in [r.content])
    finally:
        d._attack.cache_clear()
