"""Paired comparison of two cells that ran the same tasks and seeds (docs/phase7.md, T5).

    uv run python scripts/paired.py runs/<A> runs/<B> [--suite dojo/workspace] [--harness-a h] [--harness-b h]
                                  [--split .tools/h2-trees/workspace-k1]

Reads `episodes.json` (written by `hm aggregate`) from both runs, pairs episodes on (task, seed),
and reports: mean success of each side, the mean paired difference A−B with a task-resampled
bootstrap 95% CI, and a sign-flip permutation p-value on per-task mean differences. `--split`
reads a frozen tree's `hm_meta.json` and repeats the comparison on its train and held-out tasks,
which is how a Harness² row is reported (held-out first). Horizon statistics (tool calls, turns,
tool-error rate) are printed for both sides.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(run: Path, suite: str | None, harness: str | None) -> dict[tuple[str, int], dict]:
    eps = json.loads((Path(run) / "episodes.json").read_text())
    out = {}
    for e in eps:
        if e.get("tainted"):
            continue
        if suite and f"{e['benchmark']}/{e['suite']}" != suite:
            continue
        if harness and e["harness"] != harness:
            continue
        out[(e["task_id"], int(e["seed"]))] = e
    return out


def per_task_diffs(a: dict, b: dict, tasks: set[str] | None = None) -> dict[str, list[float]]:
    d: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        task, _ = key
        if tasks is not None and task not in tasks:
            continue
        d[task].append(float(a[key]["success"]) - float(b[key]["success"]))
    return d


def bootstrap_mean_ci(groups: list[list[float]], draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [rng.choice(groups) for _ in groups]
        flat = [x for g in sample for x in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def sign_flip_p(task_means: list[float], perms: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation test: under H0 each task's mean difference is symmetric around 0."""
    rng = random.Random(seed)
    observed = abs(sum(task_means))
    hits = 0
    for _ in range(perms):
        s = sum(x if rng.random() < 0.5 else -x for x in task_means)
        if abs(s) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (perms + 1)


def horizon(eps: dict) -> str:
    vals = list(eps.values())
    if not vals:
        return "n=0"
    calls = sum(e.get("tool_calls") or 0 for e in vals)
    errs = sum(e.get("tool_errors") or 0 for e in vals)
    turns = statistics.mean(e.get("turns") or 0 for e in vals)
    return (
        f"n={len(vals)} calls/ep={calls / len(vals):.2f} turns/ep={turns:.2f} "
        f"tool_err={100 * errs / max(calls, 1):.1f}%"
    )


def report(name: str, a: dict, b: dict, tasks: set[str] | None = None) -> None:
    d = per_task_diffs(a, b, tasks)
    if not d:
        print(f"{name}: no paired episodes")
        return
    pairs = [(k, a[k], b[k]) for k in sorted(set(a) & set(b)) if tasks is None or k[0] in tasks]
    ma = statistics.mean(float(x["success"]) for _, x, _ in pairs)
    mb = statistics.mean(float(y["success"]) for _, _, y in pairs)
    groups = list(d.values())
    flat = [x for g in groups for x in g]
    lo, hi = bootstrap_mean_ci(groups)
    p = sign_flip_p([statistics.mean(g) for g in groups])
    print(
        f"{name}: tasks={len(groups)} pairs={len(flat)}  A={ma:.3f}  B={mb:.3f}  "
        f"A-B={statistics.mean(flat):+.3f} [{lo:+.3f}, {hi:+.3f}]  sign-flip p={p:.4f}"
    )
    worst = sorted(d.items(), key=lambda kv: statistics.mean(kv[1]))[:5]
    for task, diffs in worst:
        if statistics.mean(diffs) < 0:
            print(f"    {task}: A-B={statistics.mean(diffs):+.2f} over {len(diffs)} seeds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--suite")
    ap.add_argument("--harness-a")
    ap.add_argument("--harness-b")
    ap.add_argument("--split", help="frozen tree dir with hm_meta.json (train/test task lists)")
    args = ap.parse_args()
    a = load(Path(args.run_a), args.suite, args.harness_a)
    b = load(Path(args.run_b), args.suite, args.harness_b)
    print(f"A {args.run_a}: {horizon(a)}")
    print(f"B {args.run_b}: {horizon(b)}")
    report("all tasks", a, b)
    if args.split:
        meta = json.loads((Path(args.split) / "hm_meta.json").read_text())
        suites = {k[0].rsplit("/", 1)[0] for k in a} | {k[0].rsplit("/", 1)[0] for k in b}

        # tree meta names tasks by their benign id; the attacked suite's ids carry an __injection suffix
        def expand(ids: list[str]) -> set[str]:
            out = set()
            for t in ids:
                base = t.rsplit("/", 1)[-1]
                out |= {k[0] for k in set(a) | set(b) if k[0].rsplit("/", 1)[-1].split("__")[0] == base}
            return out

        report("held-out tasks", a, b, expand(meta.get("test_tasks", [])))
        report("train tasks", a, b, expand(meta.get("train_tasks", [])))
        _ = suites


if __name__ == "__main__":
    main()
