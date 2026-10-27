"""`hm refine-online`: Harness²'s own protocol as a matrix column (docs/phase7.md, decision D14).

The paper evaluates *per-task, online* harnessing: every task starts from the base harness, one
Propose–Probe–Compose step (k=1) or a k-wide/k-deep recursion refines it on that task alone, the
composed harness's rollout is what gets delivered, and a library of adopted harnesses plus lessons
is threaded forward to the next task. `hm refine` (decision D5) is the offline cousin: refine on a
train split, freeze, evaluate. This module runs the paper's protocol over a seed-ordered stream of a
suite's tasks and writes a normal run directory, so `hm score`, `hm aggregate` and `hm load` treat
the delivered rollouts as episodes of a column named `harness2_online`.

What is recorded per episode (`adapter_extra["harness2_online"]`): which candidate was delivered and
whether it is the base tree, the tree's digest, how many rollouts the step consumed, the improver's
dollars (Harness²'s own ledger, cross-checked against the audit files) plus the non-delivered
rollouts' dollars — together `usd_improver`, kept apart from `usd` (D14) — and, for the security
arms, whether the delivering tree carries attacker text (`poison`) or an injection guardrail
(`defensive`), see `hm/harness2_poison.py` and D16.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from hm import preflight
from hm.adapters.harness2 import Harness2Adapter, tree_digest
from hm.harness2_poison import WRAPPERS, defensive_edit, dojo_identifiers, scan_library, scan_tree
from hm.harness2_refine import HmBenchmarkAdapter, KeyProposer, _h2, plan
from hm.runner import new_run_id
from hm.types import AdapterManifest, RunManifest, Trace, utcnow

HARNESS = "harness2_online"


def stream_order(task_ids: list[str], seed: int) -> list[str]:
    """The paper processes each benchmark's test tasks as a stream in shuffled order, three
    streams per number. One stream here = one seed = one order."""
    ids = list(task_ids)
    random.Random(seed).shuffle(ids)
    return ids


def rollout_labels(k: int, mode: str) -> list[str]:
    """Every rollout one task can consume: k references, an invariant and a domain candidate per
    width/step, and k composed finals: `agg{i}` in parallel mode, `c{t}` in sequential mode
    (Harness² `recursion.py` labels)."""
    bases = [f"base{i}" for i in range(1, k + 1)]
    cands = [f"{dim}{t}" for t in range(1, k + 1) for dim in ("inv", "dom")]
    finals = [f"{'agg' if mode == 'agg' else 'c'}{t}" for t in range(1, k + 1)]
    return bases + cands + finals


def identifiers_for(adapter: HmBenchmarkAdapter) -> tuple[str, ...]:
    if adapter.bench.name == "dojo":
        return dojo_identifiers(adapter.bench._split(adapter.suite)[0])
    return tuple(WRAPPERS)


def _usd(trace: Trace) -> float:
    from hm.scoring.cost import usd_for  # noqa: PLC0415

    return usd_for(trace) or 0.0


def _read_trace(run_dir: Path | str | None) -> Trace | None:
    if not run_dir:
        return None
    p = Path(run_dir) / "trace.json"
    return Trace.model_validate_json(p.read_text()) if p.is_file() else None


def _audit_costs(audit_dir: Path) -> dict[str, float]:
    out = {}
    if not audit_dir.is_dir():
        return out
    for p in audit_dir.glob("*.meta.json"):
        try:
            out[p.name] = float(json.loads(p.read_text()).get("cost_usd") or 0.0)
        except (OSError, ValueError):
            continue
    return out


def _done_tasks(run_dir: Path) -> set[str]:
    p = run_dir / "traces.jsonl"
    if not p.is_file():
        return set()
    return {json.loads(line)["task_id"] for line in p.read_text().splitlines() if line.strip()}


def run_stream(
    adapter: HmBenchmarkAdapter,
    *,
    k: int = 1,
    mode: str = "agg",
    runs_dir: Path,
    max_usd: float | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    tag_suffix: str = "",
    log=print,
) -> Path | dict[str, Any]:
    """One stream of the suite's tasks through Harness²'s per-task protocol; returns the run dir.

    `max_usd` counts the delivered rollouts *and* the improvement compute (sessions + the other
    rollouts); the stream stops after the task that reaches it and keeps what was written.
    `run_id` resumes an existing run dir: tasks already in its traces are skipped, and Harness²
    resumes the rest from its own summaries."""
    schedule, _h2spec, library_store, _prop, _oc = _h2()
    seed = adapter.seed
    order = stream_order(adapter.tasks("all"), seed)
    # The tag names the Harness² namespace and the library; a stream that must not share cached
    # rollouts with another on the same suite and seed (a different attack, say) gets a suffix.
    tag = f"online_{adapter.proposer_kind}" + (f"_{tag_suffix}" if tag_suffix else "")
    p = plan(adapter, k=k, mode=mode)
    p.update({"column": HARNESS, "stream_seed": seed, "order": order, "tag": tag})
    if dry_run:
        return p
    if not p["improver"]["ready"]:
        raise RuntimeError(
            "improver not configured: pass --improver-model (bare Anthropic id for claude_cli)"
        )
    guard = preflight.check(runs_dir, need_docker=bool(getattr(adapter.bench, "needs_docker", False)))
    log(f"preflight: {preflight.format_report(guard).splitlines()[0]}")

    runs_root = adapter.runs_root
    library = library_store.Library(
        root=runs_root / "library" / f"{tag}-s{seed}" / mode,
        domain=adapter.domain_of(order[0]),
        surface=adapter.substrate.surface(),
    )
    prop = KeyProposer(adapter.spec.channels["proposer"], audit_dir=runs_root / "_audit")
    identifiers = identifiers_for(adapter)
    ns = schedule.Namespace(adapter.spec.name, adapter.substrate.name, mode, k, seed, tag)

    run_id = run_id or new_run_id()
    run_dir = runs_dir / run_id
    resumed = run_dir.is_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    first_task = adapter._tasks[order[0]]
    base_manifest = Harness2Adapter().prepare(adapter.agent_spec, first_task)
    column_manifest = base_manifest.model_copy(
        update={
            "notes": base_manifest.notes
            + "; harness2_online: per-task Propose-Probe-Compose, library threaded forward, "
            "delivered rollout is the episode; improver pinned to the executor model",
            "extra": {
                **base_manifest.extra,
                "online": True,
                "k": k,
                "mode": mode,
                "stream_seed": seed,
                "tag": tag,
                "library_root": str(library.root),
                "library_dir": str(library.dir),
                "improver": p["improver"],
                "base_tree_sha256": p["base_tree_sha256"],
            },
        }
    )
    if resumed and (run_dir / "manifest.json").is_file():
        manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    else:
        manifest = RunManifest(
            run_id=run_id,
            started_at=utcnow(),
            model=adapter.agent_spec.model.name,
            mode="out_of_the_box",
            harnesses={HARNESS: column_manifest},
            tasks={adapter.suite_ref: len(order)},
            seeds=1,
            spec=adapter.agent_spec,
            extra={"stream_seed": seed, "k": k, "recursion_mode": mode, "order": order, "tag": tag},
        )
    manifest.harnesses[HARNESS] = column_manifest
    manifest.extra.pop("stopped_early", None)  # a resumed stream gets a fresh verdict
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    done = _done_tasks(run_dir)
    spent = float(manifest.extra.get("usd_total") or 0.0)
    spent_table = float(manifest.extra.get("usd_table") or 0.0)
    spent_improver = float(manifest.extra.get("usd_improver_total") or 0.0)
    stopped: str | None = None
    if max_usd is not None:
        log(f"spend cap: ${max_usd:.2f} (delivered rollouts + improvement compute)")
    audit_dir = runs_root / "_audit"
    with (run_dir / "traces.jsonl").open("a") as f:
        for i, tid in enumerate(order):
            if tid in done:
                continue
            if stopped:
                break
            if preflight.below_hard_floor(runs_dir):
                stopped = (
                    f"host free space under HM_HARD_FLOOR_GB={preflight.HARD_FLOOR_GB:g} after {i} tasks"
                )
                log(f"STOPPING STREAM: {stopped}")
                break
            library.retain(order[:i])
            entries_before = len(library.entries())
            seen_audit = set(_audit_costs(audit_dir))
            t0 = time.time()
            log(f"[{i + 1}/{len(order)}] {tid} (library entries: {entries_before})")
            res = schedule.run_task(
                tid,
                k,
                adapter=adapter,
                prop=prop,
                runs_root=runs_root,
                modes=(mode,),
                seed=seed,
                tag=tag,
                library=library,
            )
            d = res.to_dict()
            sel = (d.get("selections") or {}).get(mode) or {}
            mode_d = (d.get("modes") or {}).get(mode) or {}
            stages = mode_d.get("stages") or []
            label = sel.get("chosen_label")
            tr = _read_trace(sel.get("chosen_run_dir"))
            fallback = None
            if tr is None:  # the step produced nothing deliverable: the episode is the base rollout, failed
                fallback = _read_trace(adapter._run_dir(ns.run_id(tid, "base1")))
                tr = fallback or Trace(
                    run_id=run_id,
                    task_id=tid,
                    benchmark=adapter.bench.name,
                    suite=adapter.suite,
                    model=adapter.agent_spec.model.name,
                    harness=HARNESS,
                    harness_version="none",
                    seed=seed,
                    attacked=first_task.attacked,
                    adapter_manifest=AdapterManifest(system_prompt_sha256=None),
                )
                errs = "; ".join(d.get("errors") or [])[:200]
                tr.error = f"harness2_online: no deliverable ({d.get('status')}: {errs})"
            # costs: every rollout the step consumed, and every improver session it opened
            rollouts: dict[str, float] = {}
            for lbl in rollout_labels(k, mode):
                rt = _read_trace(adapter._run_dir(ns.run_id(tid, lbl)))
                if rt is not None:
                    rollouts[lbl] = _usd(rt)
            usd_delivered = _usd(tr)
            usd_extra_rollouts = max(0.0, sum(rollouts.values()) - usd_delivered)
            new_audit = {n: c for n, c in _audit_costs(audit_dir).items() if n not in seen_audit}
            usd_sessions = sum(new_audit.values()) or float(sum(st.get("usd") or 0.0 for st in stages))
            usd_improver = usd_sessions + usd_extra_rollouts
            # the delivering tree
            tree_dir = ns.task_dir(runs_root, tid) / "harnesses" / label if label else None
            if tree_dir is None or not tree_dir.is_dir():
                tree_dir = adapter.base_harness()
            poison = scan_tree(tree_dir, identifiers, base=adapter.base_harness())
            defence = defensive_edit(tree_dir, base=adapter.base_harness())
            # the episode
            tr.run_id = run_id
            tr.harness = HARNESS
            tr.seed = seed
            tr.harness_version = f"{tr.harness_version}+online-k{k}{mode}"
            tr.adapter_manifest = tr.adapter_manifest.model_copy(
                update={
                    "extra": {
                        **tr.adapter_manifest.extra,
                        "online": True,
                        "k": k,
                        "mode": mode,
                        "harness_files_sha256": tree_digest(tree_dir),
                    }
                }
            )
            tr.adapter_extra = {
                **tr.adapter_extra,
                "harness2_online": {
                    "task_index": i,
                    "stream_seed": seed,
                    "status": d.get("status"),
                    "errors": d.get("errors") or [],
                    "chosen_label": label,
                    "chosen_tree_is_base": bool(sel.get("chosen_tree_is_base", label is None)),
                    "harness_fingerprint": sel.get("harness_fingerprint"),
                    "chosen_tree_sha256": tree_digest(tree_dir),
                    "delivered_fallback": "base1" if fallback is not None else None,
                    "rollouts": len(rollouts),
                    "rollout_usd": rollouts,
                    "usd_sessions": round(usd_sessions, 6),
                    "usd_extra_rollouts": round(usd_extra_rollouts, 6),
                    "usd_improver": round(usd_improver, 6),
                    "stages": [
                        {
                            k_: st.get(k_)
                            for k_ in ("label", "fell_back", "kept_base", "usd", "cached", "error")
                        }
                        for st in stages
                    ],
                    "library_entries_before": entries_before,
                    "library_entries_after": len(library.entries()),
                    "poison": poison,
                    "defensive": defence,
                    "wall_s": round(time.time() - t0, 1),
                },
            }
            f.write(tr.model_dump_json() + "\n")
            f.flush()
            done.add(tid)
            spent_table += usd_delivered
            spent_improver += usd_improver
            spent = spent_table + spent_improver
            log(
                f"    {'ERR ' + tr.error[:80] if tr.error else 'ok'} chosen={label} "
                f"base={sel.get('chosen_tree_is_base')} "
                f"rollouts={len(rollouts)} sessions=${usd_sessions:.2f} "
                f"poisoned={poison['poisoned']} defensive={defence['defensive']} "
                f"{tr.adapter_extra['harness2_online']['wall_s']}s total=${spent:.2f}"
            )
            manifest.extra.update(
                {
                    "episodes_written": len(done),
                    "usd_table": round(spent_table, 4),
                    "usd_improver_total": round(spent_improver, 4),
                    "usd_total": round(spent, 4),
                }
            )
            (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
            if max_usd is not None and spent >= max_usd:
                stopped = (
                    f"spend cap ${max_usd:.2f} reached: ${spent:.2f} after {len(done)}/{len(order)} tasks"
                )
                log(f"STOPPING STREAM: {stopped}")
    # this stream's domain only: the root is shared by every suite that runs under the same tag and seed
    manifest.extra["library"] = scan_library(library.dir, identifiers)
    if stopped:
        manifest.extra["stopped_early"] = stopped
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return run_dir
