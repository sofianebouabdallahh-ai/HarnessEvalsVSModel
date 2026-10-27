import pytest

from hm.aggregate import Episode, aggregate, bootstrap_ci, pass_hat_k, pass_k_over_tasks


def test_pass_hat_k_hand_values():
    assert pass_hat_k(3, 5, 1) == 0.6
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_hat_k(4, 5, 5) == 0.0
    assert pass_hat_k(4, 5, 3) == 4 / 10  # C(4,3)/C(5,3)
    assert pass_hat_k(2, 5, 2) == 1 / 10
    assert pass_hat_k(3, 3, 5) is None  # k > n undefined


def test_pass_k_over_tasks_averages():
    tasks = {"a": [True] * 5, "b": [True, True, True, False, False]}
    assert pass_k_over_tasks(tasks, 1) == (1.0 + 0.6) / 2
    assert pass_k_over_tasks(tasks, 5) == 0.5


def test_bootstrap_constant_data_has_zero_width():
    groups = [[1.0, 1.0], [1.0], [1.0, 1.0, 1.0]]
    lo, hi = bootstrap_ci(groups, lambda xs: sum(xs) / len(xs), draws=200)
    assert lo == hi == 1.0


def test_bootstrap_brackets_point_estimate():
    groups = [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
    lo, hi = bootstrap_ci(groups, lambda xs: sum(xs) / len(xs), draws=500)
    assert lo <= 0.5 <= hi and lo < hi


def _ep(task, seed, success, attacked=False, attack_success=None, seen=False, usd=0.01):
    return Episode(
        "h",
        "toy",
        "s",
        "out_of_the_box",
        task,
        seed,
        attacked,
        success,
        attack_success,
        seen,
        usd,
        100,
        10,
        0,
        0,
        False,
    )


def test_aggregate_cells():
    eps = [_ep("t1", s, True) for s in range(5)] + [_ep("t2", s, s < 2) for s in range(5)]
    eps += [_ep("t3", s, True, attacked=True, attack_success=s == 0, seen=s < 3) for s in range(5)]
    cells = {c.metric: c for c in aggregate("r", eps, draws=50)}
    assert cells["pass^1"].value == 12 / 15
    assert cells["pass^5"].value == 2 / 3
    assert cells["benign_utility"].value == 7 / 10
    assert cells["utility_under_attack"].value == 1.0
    assert cells["targeted_asr"].value == 0.2
    assert cells["asr_given_seen"].value == 1 / 3
    assert cells["usd_per_solve"].value == 15 * 0.01 / 12
    assert cells["pass^1"].ci_lo <= cells["pass^1"].value <= cells["pass^1"].ci_hi


def test_policy_violation_rate_counts_only_episodes_that_carry_it():
    from hm.aggregate import CELL_METRICS, Episode

    def ep(attacked, pv):
        return Episode(
            harness="h", benchmark="custom", suite="s", mode="m", task_id="t", seed=0, attacked=attacked,
            success=True, attack_success=None, injection_seen=False, usd=None, tokens=0, wall_ms=0,
            tool_errors=0, tool_errors_recovered=0, tainted=False, policy_violation=pv,
        )  # fmt: skip

    f = CELL_METRICS["policy_violation_rate"]
    assert f([ep(True, True), ep(True, False), ep(False, False), ep(False, None)]) == pytest.approx(1 / 3)
    assert f([ep(True, None)]) is None  # a dojo-style cell without the scorer has no value, not 0


def test_api_refusal_is_scored_and_aggregated():
    from hm.aggregate import CELL_METRICS, to_episode
    from hm.scoring import trajectory
    from hm.types import AdapterManifest, LLMCallStep, Trace

    def trace(stop):
        return Trace(
            run_id="r", task_id="t", benchmark="toy", suite="basic", model="m", harness="h", harness_version="0",
            seed=0, adapter_manifest=AdapterManifest(system_prompt_sha256=None), final_output=None,
            steps=[LLMCallStep(i=0, input_tokens=1, output_tokens=0, latency_ms=1, stop_reason=stop)],
        )  # fmt: skip

    assert {s.name: s.value for s in trajectory.score(trace("content-filter"))}["api_refusal"] == 1.0
    refused, fine = trace("refusal"), trace("end_turn")
    s_ref = {s.name: s.value for s in trajectory.score(refused)}
    s_ok = {s.name: s.value for s in trajectory.score(fine)}
    assert s_ref["api_refusal"] == 1.0 and s_ok["api_refusal"] == 0.0
    eps = [to_episode(refused, trajectory.score(refused)), to_episode(fine, trajectory.score(fine))]
    assert eps[0].api_refusal and not eps[1].api_refusal
    assert CELL_METRICS["api_refusal_rate"](eps) == 0.5


# ---------------------------------------------------------------- Phase 7: horizon, improver cost, poisoning


def _ep7(task, success, **kw):
    base = dict(
        harness="h", benchmark="dojo", suite="workspace", mode="out_of_the_box", task_id=task, seed=0,
        attacked=False, success=success, attack_success=None, injection_seen=False, usd=0.03,
        tokens=1000, wall_ms=10, tool_errors=0, tool_errors_recovered=0, tainted=False,
    )  # fmt: skip
    base.update(kw)
    return Episode(**base)


def test_horizon_and_improver_cost_metrics():
    eps = [
        _ep7("t1", True, tool_calls=4, turns=3, usd_improver=0.5),
        _ep7("t2", False, tool_calls=2, turns=2, tool_errors=1, usd_improver=0.5),
    ]
    c = {x.metric: x.value for x in aggregate("r", eps, draws=10)}
    assert c["tool_calls_per_episode"] == 3.0 and c["turns_per_episode"] == 2.5
    assert c["tool_error_rate"] == pytest.approx(1 / 6)
    assert c["usd_per_solve"] == pytest.approx(0.06)  # Σ usd / solves
    assert c["usd_per_solve_incl_refine"] == pytest.approx(0.06 + 1.0)  # improvement compute counted
    assert c["poison_executed_rate"] is None and c["harness_poisoned_rate"] is None


def test_poison_rates_only_over_episodes_that_carry_the_flag():
    eps = [
        _ep7("t1", True, poison_executed=None, harness_poisoned=True, defensive_edit=False),
        _ep7("t2", True, poison_executed=True, harness_poisoned=False, defensive_edit=True),
        _ep7("t3", True, poison_executed=False),
    ]
    c = {x.metric: x.value for x in aggregate("r", eps, draws=10)}
    assert (
        c["poison_executed_rate"] == 0.5
        and c["harness_poisoned_rate"] == 0.5
        and c["defensive_edit_rate"] == 0.5
    )
    assert c["usd_per_solve_incl_refine"] == c["usd_per_solve"]  # no improver dollars anywhere
