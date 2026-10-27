"""`hm load`: push one run's manifest, episodes, scores and cells into Postgres.
Traces stay on disk; Postgres holds the queryable projection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg

from hm.runner import read_manifest, read_traces
from hm.scoring import read_scores

SCHEMA = Path(__file__).with_name("schema.sql")
DEFAULT_DSN = "postgresql://hm:hm@localhost:5434/hm"


def dsn() -> str:
    return os.environ.get("HM_PG_DSN", DEFAULT_DSN)


def init_db(conn_str: str | None = None) -> None:
    with psycopg.connect(conn_str or dsn()) as conn:
        conn.execute(SCHEMA.read_text())


def load_run(run_dir: Path, conn_str: str | None = None) -> int:
    manifest = read_manifest(run_dir)
    traces = read_traces(run_dir)
    scores = read_scores(run_dir) if (run_dir / "scores.jsonl").exists() else {}
    cells = json.loads((run_dir / "cells.json").read_text()) if (run_dir / "cells.json").exists() else []
    trace_path = str((run_dir / "traces.jsonl").resolve())

    with psycopg.connect(conn_str or dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            """insert into runs (run_id, started_at, model, mode, manifest) values (%s,%s,%s,%s,%s)
               on conflict (run_id) do update set manifest = excluded.manifest""",
            (manifest.run_id, manifest.started_at, manifest.model, manifest.mode, manifest.model_dump_json()),
        )
        for tr in traces:
            s = {sc.name: sc.value for sc in scores.get(tr.episode_key, [])}
            t = tr.totals
            cur.execute(
                """insert into episodes (episode_key, run_id, ts, benchmark, suite, task_id, harness,
                     harness_version, seed, mode, success, attacked, attack_success, refused,
                     injection_seen, turns, tool_calls, tool_errors, input_tokens, output_tokens, usd,
                     wall_ms, tainted, error, trace_path)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   on conflict (episode_key) do update set
                     success = excluded.success, attack_success = excluded.attack_success,
                     refused = excluded.refused, injection_seen = excluded.injection_seen,
                     usd = excluded.usd, tainted = excluded.tainted
                   returning episode_id""",
                (
                    tr.episode_key,
                    tr.run_id,
                    tr.started_at,
                    tr.benchmark,
                    tr.suite,
                    tr.task_id,
                    tr.harness,
                    tr.harness_version,
                    tr.seed,
                    tr.mode,
                    None if "success" not in s else bool(s["success"]),
                    tr.attacked,
                    None if s.get("attack_success") is None else bool(s["attack_success"]),
                    None if s.get("refused") is None else bool(s["refused"]),
                    None if s.get("injection_seen") is None else bool(s["injection_seen"]),
                    t.turns,
                    t.tool_calls,
                    t.tool_errors,
                    t.input_tokens,
                    t.output_tokens,
                    s.get("usd"),
                    t.wall_clock_ms,
                    tr.tainted,
                    tr.error,
                    trace_path,
                ),
            )
            episode_id = cur.fetchone()[0]
            cur.execute("delete from scores where episode_id = %s", (episode_id,))
            for sc in scores.get(tr.episode_key, []):
                cur.execute(
                    "insert into scores (episode_id, scorer, name, value, label, explanation) "
                    "values (%s,%s,%s,%s,%s,%s)",
                    (episode_id, sc.scorer, sc.name, sc.value, sc.label, sc.explanation),
                )
        cur.execute("delete from cells where run_id = %s", (manifest.run_id,))
        for c in cells:
            cur.execute(
                """insert into cells (run_id, harness, benchmark, suite, mode, metric, value, ci_lo, ci_hi,
                     n_episodes, n_tasks) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    c["run_id"],
                    c["harness"],
                    c["benchmark"],
                    c["suite"],
                    c["mode"],
                    c["metric"],
                    c["value"],
                    c["ci_lo"],
                    c["ci_hi"],
                    c["n_episodes"],
                    c["n_tasks"],
                ),
            )
        conn.commit()
    return len(traces)
