"""Scorers are pure functions Trace -> list[Score]. `hm score` writes scores.jsonl
next to traces.jsonl; traces are never rewritten."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from hm.benchmarks import get_benchmark
from hm.scoring import cost, fairness, harness, trajectory
from hm.types import Score, Trace


def _benchmark_score(trace: Trace) -> list[Score]:
    bench = get_benchmark(trace.benchmark)
    return bench.score(bench.task(trace.task_id), trace)


SCORERS: dict[str, Callable[[Trace], list[Score]]] = {
    "benchmark": _benchmark_score,
    "cost": cost.score,
    "trajectory": trajectory.score,
    "fairness": fairness.score,
    "harness": harness.score,
}


def score_trace(trace: Trace, scorers: list[str] | None = None) -> list[Score]:
    out: list[Score] = []
    for name in scorers or list(SCORERS):
        out.extend(SCORERS[name](trace))
    return out


def score_run(run_dir: Path, scorers: list[str] | None = None) -> Path:
    from hm.runner import read_manifest, read_traces

    fairness.set_run_manifest(read_manifest(run_dir) if (run_dir / "manifest.json").exists() else None)
    path = run_dir / "scores.jsonl"
    with path.open("w") as f:
        for tr in read_traces(run_dir):
            f.write(
                json.dumps(
                    {
                        "episode_key": tr.episode_key,
                        "scores": [s.model_dump() for s in score_trace(tr, scorers)],
                    }
                )
                + "\n"
            )
    return path


def read_scores(run_dir: Path) -> dict[str, list[Score]]:
    out: dict[str, list[Score]] = {}
    with (run_dir / "scores.jsonl").open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["episode_key"]] = [Score.model_validate(s) for s in d["scores"]]
    return out
