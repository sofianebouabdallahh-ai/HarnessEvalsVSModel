"""`hm run`: for each (harness, task, seed) start an env-server, run the adapter, write a Trace.

The results directory is append-only: `traces.jsonl` is only ever appended to, and
`manifest.json` is written once at the start of the run.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Literal

import yaml

from hm.adapters import EnvHandle, get_adapter
from hm.benchmarks import resolve_suite
from hm.envserver import EnvServer
from hm.types import AgentSpec, RunManifest, Step, Task, Trace, utcnow

Mode = Literal["out_of_the_box", "equalized_prompt"]


def load_spec(path: Path) -> AgentSpec:
    raw = os.path.expandvars(path.read_text())
    if unexpanded := sorted(set(re.findall(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", raw))):
        raise ValueError(
            f"{path}: unset environment variable(s) {', '.join(unexpanded)} — export them "
            "(see .env.example) instead of running the matrix against a literal placeholder"
        )
    return AgentSpec.model_validate(yaml.safe_load(raw))


def new_run_id() -> str:
    return f"{utcnow():%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


def latest_run(runs_dir: Path) -> Path:
    runs = sorted(p for p in runs_dir.iterdir() if (p / "manifest.json").exists())
    if not runs:
        raise FileNotFoundError(f"no runs under {runs_dir}")
    return runs[-1]


def _merge_steps(llm: list[tuple[float, Step]], tools: list[tuple[float, Step]]) -> list[Step]:
    merged = [s for _, s in sorted(llm + tools, key=lambda x: x[0])]
    for i, s in enumerate(merged):
        s.i = i
    return merged


def _describe(e: BaseException) -> str:
    """Leaf-most message: async context managers wrap provider errors in ExceptionGroups."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {str(e)[:300]}"


# Provider-side transient failures. An episode that dies of one never started on fair terms,
# so it is re-run (bounded) and the re-run count is recorded on the trace.
TRANSIENT_MARKERS = ("rate limit", "rate_limit", "overloaded", "429", "529", "usage limit", "hit your limit",
                     "connection reset", "timed out", "503")  # fmt: skip
EPISODE_RETRIES = int(os.environ.get("HM_EPISODE_RETRIES", "3"))
RETRY_BACKOFF_S = int(os.environ.get("HM_RETRY_BACKOFF_S", "60"))


def is_transient(error: str | None) -> bool:
    return bool(error) and any(m in error.lower() for m in TRANSIENT_MARKERS)


def _attempt(spec: AgentSpec, adapter, env, task: Task, seed: int):
    addendum = env.prompt_addendum()
    run_task = task.model_copy(update={"prompt": task.prompt + addendum}) if addendum else task
    with EnvServer(env, spec.limits) as srv:
        try:
            out = adapter.run(spec, run_task, seed, EnvHandle(url=srv.url, tools=env.tools()))
        except Exception as e:  # harness crash is an episode failure, not a run failure
            from hm.adapters.base import AdapterRun

            out = AdapterRun(error=_describe(e))
    return out, srv.log


def run_episode(
    spec: AgentSpec, adapter, bench, task: Task, seed: int, run_id: str, mode: Mode, log=lambda *_: None
) -> Trace:
    manifest = adapter.prepare(spec, task)
    t0 = time.time()
    retries = 0
    while True:
        env = bench.make_env(task)
        out, tool_log = _attempt(spec, adapter, env, task, seed)
        if is_transient(out.error) and retries < EPISODE_RETRIES:
            retries += 1
            log(f"    transient error ({out.error[:80]}); re-running episode in {RETRY_BACKOFF_S * retries}s "
                f"[{retries}/{EPISODE_RETRIES}]")  # fmt: skip
            time.sleep(RETRY_BACKOFF_S * retries)
            continue
        break
    error = out.error
    wall_ms = int((time.time() - t0) * 1000)
    trace = Trace(
        run_id=run_id,
        task_id=task.id,
        benchmark=task.benchmark,
        suite=task.suite,
        model=spec.model.name,
        harness=adapter.name,
        harness_version=adapter.version,
        seed=seed,
        mode=mode,
        attacked=task.attacked,
        attack=task.attack,
        adapter_manifest=manifest,
        steps=_merge_steps(out.llm_steps, tool_log),
        final_output=out.final_output,
        env_state_hash=env.state_hash(),
        env_final=env.snapshot(out.final_output),
        error=error,
        adapter_extra=out.extra,
        platform_retries=retries,
    )
    trace.recompute_totals()
    trace.totals.wall_clock_ms = wall_ms
    return trace


def _episode_usd(tr: Trace) -> float:
    """Price-table cost of one episode (0 when the model is unpriced); the harness-reported figure is
    not used here because not every adapter has one."""
    from hm.scoring.cost import usd_for

    return usd_for(tr) or 0.0


def run_matrix(
    spec: AgentSpec,
    harnesses: list[str],
    suites: list[str],
    n: int,
    runs_dir: Path,
    mode: Mode = "out_of_the_box",
    equalized_prompt: str | None = None,
    log=print,
    max_usd: float | None = None,
) -> Path:
    """Run every (harness, task, seed). `max_usd` (default: env HM_MAX_USD, unset = no cap) stops the
    run after the episode that takes the price-table spend to the cap; what was written is kept and
    `stopped_early` says why. The running total is always recorded as `manifest.extra["usd_table"]`."""
    if max_usd is None:
        env_cap = os.environ.get("HM_MAX_USD")
        max_usd = float(env_cap) if env_cap else None
    if mode == "equalized_prompt":
        if not equalized_prompt:
            raise ValueError("equalized_prompt mode needs a shared system prompt")
        spec = spec.model_copy(update={"system_prompt": equalized_prompt})

    from hm import preflight

    resolved = [resolve_suite(s) for s in suites]
    needs_docker = any(getattr(b, "needs_docker", False) for b, _ in resolved)
    guard = preflight.check(runs_dir, need_docker=needs_docker)  # raises DiskGuardError
    log(f"preflight: {preflight.format_report(guard).splitlines()[0]}")

    run_id = new_run_id()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    adapters = {h: get_adapter(h) for h in harnesses}
    first_task = resolved[0][0].tasks(resolved[0][1])[0]
    manifest = RunManifest(
        run_id=run_id,
        started_at=utcnow(),
        model=spec.model.name,
        mode=mode,
        harnesses={h: a.prepare(spec, first_task) for h, a in adapters.items()},
        tasks={f"{b.name}/{s}": len(b.tasks(s)) for b, s in resolved},
        seeds=n,
        spec=spec,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    total = sum(len(b.tasks(s)) for b, s in resolved) * n * len(harnesses)
    done = 0
    spent = 0.0
    stopped: str | None = None
    if max_usd is not None:
        log(f"spend cap: ${max_usd:.2f} (price table, all harnesses in this run)")
    with (run_dir / "traces.jsonl").open("a") as f:
        for adapter in adapters.values():
            try:
                for bench, suite in resolved:
                    for task in bench.tasks(suite):
                        for seed in range(n):
                            if stopped:
                                break
                            if preflight.below_hard_floor(runs_dir):
                                stopped = (
                                    f"host free space under HM_HARD_FLOOR_GB="
                                    f"{preflight.HARD_FLOOR_GB:g} after "
                                    f"{done}/{total} episodes"
                                )
                                log(f"STOPPING RUN: {stopped}")
                                break
                            tr = run_episode(spec, adapter, bench, task, seed, run_id, mode, log)
                            f.write(tr.model_dump_json() + "\n")
                            f.flush()
                            done += 1
                            spent += _episode_usd(tr)
                            log(
                                f"[{done}/{total}] {tr.episode_key} "
                                f"{'ERR ' + tr.error if tr.error else 'ok'} "
                                f"turns={tr.totals.turns} tools={tr.totals.tool_calls} "
                                f"{tr.totals.wall_clock_ms}ms ${spent:.2f}"
                            )
                            if max_usd is not None and spent >= max_usd:
                                stopped = (
                                    f"spend cap ${max_usd:.2f} reached: ${spent:.2f} "
                                    f"after {done}/{total} episodes"
                                )
                                log(f"STOPPING RUN: {stopped}")
            finally:
                adapter.teardown()
                # Adapters learn facts during runs (CLI version, surviving native tools); the run
                # manifest is re-derived so it reflects what was observed, not what was assumed.
                manifest.harnesses[adapter.name] = adapter.prepare(spec, first_task)
                if stopped:
                    manifest.extra["stopped_early"] = stopped
                manifest.extra["episodes_written"] = done
                manifest.extra["usd_table"] = round(spent, 4)
                (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return run_dir


def read_traces(run_dir: Path) -> list[Trace]:
    with (run_dir / "traces.jsonl").open() as f:
        return [Trace.model_validate_json(line) for line in f if line.strip()]


def read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate(json.loads((run_dir / "manifest.json").read_text()))
