"""Manifest completeness / taint detection and equalized_prompt mode (no model calls)."""

import json

from hm.aggregate import aggregate_run
from hm.runner import read_manifest, read_traces, run_matrix
from hm.scoring import fairness, read_scores, score_run
from hm.types import AdapterManifest, AgentSpec, Limits, ModelSpec, RunManifest, ToolCallStep, Trace, utcnow


def _spec(system_prompt=None) -> AgentSpec:
    return AgentSpec(
        id="t",
        model=ModelSpec(provider="mock", name="mock?flake=0&gull=0"),
        limits=Limits(max_turns=10, max_tool_calls=10, wall_clock_s=30),
        system_prompt=system_prompt,
    )


def _trace(**kw) -> Trace:
    base = dict(
        run_id="r", task_id="toy/basic/0001", benchmark="toy", suite="basic", model="m", harness="h",
        harness_version="0", seed=0, adapter_manifest=AdapterManifest(system_prompt_sha256="x"),
    )  # fmt: skip
    base.update(kw)
    return Trace(**base)


def setup_function():
    fairness.set_run_manifest(None)


def test_clean_trace_is_not_tainted():
    assert fairness.taint_reasons(_trace()) == []


def test_model_mismatch_taints():
    assert fairness.taint_reasons(_trace(adapter_extra={"model_mismatch": True})) == ["model_mismatch"]


def test_placeholder_manifest_taints_unless_run_manifest_fills_it():
    tr = _trace(
        adapter_manifest=AdapterManifest(system_prompt_sha256=None, extra_tools=[fairness.PLACEHOLDER])
    )
    assert fairness.taint_reasons(tr) == ["manifest_incomplete"]
    fairness.set_run_manifest(
        RunManifest(
            run_id="r",
            started_at=utcnow(),
            model="m",
            mode="out_of_the_box",
            tasks={},
            seeds=1,
            spec=_spec(),
            harnesses={"h": AdapterManifest(system_prompt_sha256=None, extra_tools=["TodoWrite"])},
        )  # fmt: skip
    )
    assert fairness.taint_reasons(tr) == []


def test_undeclared_tools_taint():
    steps = [
        ToolCallStep(i=0, name="kv_get", args={}, args_sha256="a"),
        ToolCallStep(i=1, name="rm_rf", args={}, args_sha256="b"),
        ToolCallStep(i=2, name="claude_code:Bash", args={}, args_sha256="c"),
    ]
    assert fairness.taint_reasons(_trace(steps=steps)) == ["undeclared_tool", "undeclared_native_tool"]


def test_tainted_episodes_are_excluded_from_cells(tmp_path):
    rd = run_matrix(_spec(), ["raw_loop"], ["toy/basic"], 1, tmp_path, log=lambda *_: None)
    score_run(rd)
    scores = read_scores(rd)
    assert all(next(s.value for s in v if s.name == "tainted") == 0.0 for v in scores.values())
    cells = {c["metric"]: c for c in json.loads(aggregate_run(rd, draws=20).read_text())}
    assert cells["pass^1"]["n_episodes"] == 5
    # poison one episode's scores and re-aggregate: it must drop out
    lines = (rd / "scores.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    for s in first["scores"]:
        if s["name"] == "tainted":
            s["value"] = 1.0
    (rd / "scores.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    cells = {c["metric"]: c for c in json.loads(aggregate_run(rd, draws=20).read_text())}
    assert cells["pass^1"]["n_episodes"] == 4


def test_equalized_prompt_mode_overrides_prompt_and_is_recorded(tmp_path):
    rd = run_matrix(
        _spec(), ["raw_loop"], ["toy/basic"], 1, tmp_path, mode="equalized_prompt",
        equalized_prompt="shared prompt", log=lambda *_: None,
    )  # fmt: skip
    m = read_manifest(rd)
    assert m.mode == "equalized_prompt" and m.spec.system_prompt == "shared prompt"
    tr = read_traces(rd)[0]
    from hm.types import sha256_text

    assert tr.mode == "equalized_prompt"
    assert tr.adapter_manifest.system_prompt_sha256 == sha256_text("shared prompt")


def test_hallucinated_native_tool_that_the_harness_rejected_does_not_taint():
    from hm.scoring import fairness
    from hm.types import AdapterManifest, ToolCallStep, ToolResultStep, Trace

    def trace(result_content, error):
        return Trace(
            run_id="r", task_id="toy/basic/t", benchmark="toy", suite="basic", model="m", harness="claude_sdk",
            harness_version="0", seed=0, final_output="x",
            adapter_manifest=AdapterManifest(system_prompt_sha256=None, extra_tools=["TodoWrite"]),
            steps=[
                ToolCallStep(i=0, name="claude_code:send_message_to_user", args={"message": "hi"}, args_sha256="a", call_id="c1"),
                ToolResultStep(i=1, call_id="c1", name="claude_code:send_message_to_user", content=result_content, latency_ms=1, error=error),
            ],
        )  # fmt: skip

    rejected = trace(
        "<tool_use_error>Error: No such tool available: send_message_to_user</tool_use_error>", "tool error"
    )
    assert "undeclared_native_tool" not in fairness.taint_reasons(rejected)
    worked = trace("Message sent.", None)
    assert "undeclared_native_tool" in fairness.taint_reasons(worked)
