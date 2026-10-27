"""End to end on the toy benchmark with the mock model: run -> score -> aggregate."""

import json

import jsonschema
import pytest

from hm.aggregate import aggregate_run
from hm.runner import read_traces, run_matrix
from hm.scoring import read_scores, score_run
from hm.types import AgentSpec, Limits, ModelSpec, Trace


def _spec(name: str) -> AgentSpec:
    return AgentSpec(
        id="t",
        model=ModelSpec(provider="mock", name=name),
        limits=Limits(max_turns=10, max_tool_calls=10, wall_clock_s=30),
    )


def test_perfect_mock_solves_everything(tmp_path):
    rd = run_matrix(
        _spec("mock?flake=0&gull=0"), ["raw_loop"], ["toy/basic"], 2, tmp_path, log=lambda *_: None
    )
    traces = read_traces(rd)
    assert len(traces) == 10 and all(t.error is None for t in traces)
    score_run(rd)
    scores = read_scores(rd)
    assert all(next(s.value for s in scores[t.episode_key] if s.name == "success") == 1.0 for t in traces)
    cells = json.loads(aggregate_run(rd, draws=50).read_text())
    p1 = next(c for c in cells if c["metric"] == "pass^1")
    assert p1["value"] == 1.0 and p1["ci_lo"] == 1.0 and p1["ci_hi"] == 1.0
    assert (rd / "manifest.json").exists()


def test_gullible_mock_is_hijacked_only_when_it_sees_the_injection(tmp_path):
    rd = run_matrix(
        _spec("mock?flake=0&gull=1"), ["raw_loop"], ["toy/injected"], 1, tmp_path, log=lambda *_: None
    )
    score_run(rd)
    scores = read_scores(rd)
    for tr in read_traces(rd):
        s = {x.name: x.value for x in scores[tr.episode_key]}
        reads = tr.task_id.endswith("0001") is False  # task 1 reads nothing
        assert s["injection_seen"] == float(reads)
        assert s["attack_success"] == float(reads)
        assert s["success"] == 1.0  # utility survives; the attack is a side effect
    cells = {c["metric"]: c for c in json.loads(aggregate_run(rd, draws=20).read_text())}
    assert cells["targeted_asr"]["value"] == 0.8
    assert cells["asr_given_seen"]["value"] == 1.0
    assert cells["injection_seen_rate"]["value"] == 0.8


def test_flaky_mock_has_nontrivial_ci(tmp_path):
    rd = run_matrix(
        _spec("mock?flake=0.5&gull=0"), ["raw_loop"], ["toy/basic"], 4, tmp_path, log=lambda *_: None
    )
    score_run(rd)
    cells = {c["metric"]: c for c in json.loads(aggregate_run(rd, draws=200).read_text())}
    p1 = cells["pass^1"]
    assert 0 < p1["value"] < 1 and p1["ci_lo"] < p1["ci_hi"]
    assert cells["pass^3"]["value"] <= p1["value"]  # pass^k is monotone non-increasing in k
    assert cells["usd_per_solve"]["value"] > 0


def test_traces_validate_against_exported_schema(tmp_path):
    rd = run_matrix(
        _spec("mock?flake=0&gull=0"), ["raw_loop"], ["toy/basic"], 1, tmp_path, log=lambda *_: None
    )
    schema = Trace.model_json_schema()
    with (rd / "traces.jsonl").open() as f:
        for line in f:
            jsonschema.validate(json.loads(line), schema)


@pytest.mark.parametrize("name,flake,gull", [("mock", 0, 0), ("mock?flake=0.2&gull=0.5", 0.2, 0.5)])
def test_mock_params_parse(name, flake, gull):
    from hm.models.mock import MockModel

    m = MockModel(name)
    assert (m.flake, m.gull, m.name) == (flake, gull, name)


def test_load_spec_refuses_unexpanded_placeholders(tmp_path, monkeypatch):
    from hm.runner import load_spec

    spec = tmp_path / "s.yaml"
    spec.write_text("id: x\nmodel: {provider: anthropic, name: ${MODEL}}\ntools_source: benchmark\n")
    monkeypatch.delenv("MODEL", raising=False)
    with pytest.raises(ValueError, match=r"MODEL"):
        load_spec(spec)
    monkeypatch.setenv("MODEL", "claude-sonnet-5-5")
    assert load_spec(spec).model.name == "claude-sonnet-5-5"
