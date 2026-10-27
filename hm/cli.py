"""`hm` command line. See README for the demo path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Harness Matrix")

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("runs")
DEMO_SPEC = ROOT / "specs" / "mock.yaml"
EQUALIZED = ROOT / "specs" / "equalized_prompt.txt"


def _run_dir(run: str, runs_dir: Path) -> Path:
    from hm.runner import latest_run

    return latest_run(runs_dir) if run == "latest" else runs_dir / run


@app.command()
def run(
    spec: Annotated[Path, typer.Option(help="Agent spec YAML")] = DEMO_SPEC,
    harness: Annotated[list[str], typer.Option("--harness", "-H")] = ["raw_loop"],  # noqa: B006
    suite: Annotated[list[str], typer.Option("--suite", "-s")] = ["toy/basic", "toy/injected"],  # noqa: B006
    n: Annotated[int, typer.Option(help="seeds per task (N ≥ 5 for pass^5)")] = 5,
    mode: Annotated[str, typer.Option()] = "out_of_the_box",
    runs_dir: Path = RUNS,
    demo: Annotated[bool, typer.Option(help="mock model, toy suites, then score+aggregate+load")] = False,
    max_usd: Annotated[
        float | None,
        typer.Option("--max-usd", help="stop when price-table spend reaches this (env HM_MAX_USD)"),
    ] = None,
):
    """Run the matrix: every (harness, task, seed) in the given suites."""
    from hm.runner import load_spec, run_matrix

    if demo:
        spec, harness, suite = DEMO_SPEC, ["raw_loop"], ["toy/basic", "toy/injected"]
    agent = load_spec(spec)
    eq = EQUALIZED.read_text() if mode == "equalized_prompt" and EQUALIZED.exists() else None
    rd = run_matrix(
        agent, harness, suite, n, runs_dir, mode=mode, equalized_prompt=eq, log=typer.echo, max_usd=max_usd
    )
    typer.echo(f"run written to {rd}")
    if demo:
        score(rd.name, runs_dir=runs_dir)
        aggregate(rd.name, runs_dir=runs_dir)
        try:
            load(rd.name, runs_dir=runs_dir)
        except Exception as e:  # Postgres is optional for the demo
            typer.echo(f"(skipped Postgres load: {type(e).__name__}: {str(e).strip().splitlines()[0]})")
        _print_matrix(rd)


@app.command()
def score(
    run: str = "latest",
    runs_dir: Path = RUNS,
    scorer: Annotated[list[str] | None, typer.Option("--scorer")] = None,
):
    """Score traces -> scores.jsonl (benchmark, cost, trajectory)."""
    from hm.scoring import score_run

    p = score_run(_run_dir(run, runs_dir), scorer)
    typer.echo(f"scores written to {p}")


@app.command()
def aggregate(
    run: str = "latest",
    runs_dir: Path = RUNS,
    bootstrap: Annotated[int, typer.Option(help="bootstrap draws")] = 1000,
):
    """Episodes -> cells.json with bootstrap CIs."""
    from hm.aggregate import aggregate_run

    p = aggregate_run(_run_dir(run, runs_dir), bootstrap)
    typer.echo(f"cells written to {p}")


@app.command()
def load(run: str = "latest", runs_dir: Path = RUNS, dsn: str | None = None):
    """Load a run into Postgres (HM_PG_DSN or postgresql://hm:hm@localhost:5434/hm)."""
    from hm.store.postgres import load_run

    n = load_run(_run_dir(run, runs_dir), dsn)
    typer.echo(f"loaded {n} episodes")


@app.command("db-init")
def db_init(dsn: str | None = None):
    """Apply schema.sql (docker compose does this automatically on first start)."""
    from hm.store.postgres import init_db

    init_db(dsn)
    typer.echo("schema applied")


@app.command()
def matrix(run: str = "latest", runs_dir: Path = RUNS):
    """Print the matrix (pass^1 with CI) for a run."""
    _print_matrix(_run_dir(run, runs_dir))


@app.command()
def refine(
    suite: Annotated[str, typer.Option("--suite", "-s", help="hm suite to refine on, e.g. dojo/workspace")],
    out: Annotated[Path, typer.Option(help="where the frozen tree is written (use as HM_HARNESS2_TREE)")],
    spec: Annotated[Path, typer.Option(help="Agent spec YAML for the executor")] = DEMO_SPEC,
    k: int = 1,
    mode: Annotated[str, typer.Option(help="agg (parallel) or seq (sequential)")] = "seq",
    seed: int = 0,
    train_frac: float = 0.5,
    max_train_tasks: int | None = None,
    improver_channel: Annotated[str, typer.Option(help="claude_cli or opencode")] = "claude_cli",
    improver_model: Annotated[
        str, typer.Option(help="bare Anthropic id (claude_cli) or provider/model")
    ] = "",
    runs_dir: Path = RUNS,
    dry_run: Annotated[bool, typer.Option(help="print the plan; start nothing")] = False,
):
    """One Harness² Propose–Probe–Compose step on the TRAIN split of a suite; freeze the adopted tree."""
    from hm.harness2_refine import HmBenchmarkAdapter
    from hm.harness2_refine import refine as _refine
    from hm.runner import load_spec

    adapter = HmBenchmarkAdapter(
        agent_spec=load_spec(spec), suite_ref=suite, runs_root=runs_dir / "harness2", seed=seed,
        train_frac=train_frac, max_train_tasks=max_train_tasks, proposer_kind=improver_channel,
        proposer_model=improver_model,
    )  # fmt: skip
    result = _refine(adapter, k=k, mode=mode, out_dir=out, dry_run=dry_run, log=typer.echo)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("refine-online")
def refine_online(
    suite: Annotated[str, typer.Option("--suite", "-s", help="hm suite to stream, e.g. tau2/airline")],
    spec: Annotated[Path, typer.Option(help="Agent spec YAML for the executor")] = DEMO_SPEC,
    k: int = 1,
    mode: Annotated[str, typer.Option(help="agg (parallel, the paper's default) or seq")] = "agg",
    seed: Annotated[int, typer.Option(help="stream seed: the task order and the Harness² namespace")] = 0,
    improver_channel: Annotated[str, typer.Option(help="claude_cli or opencode")] = "claude_cli",
    improver_model: Annotated[
        str, typer.Option(help="bare Anthropic id (claude_cli) or provider/model; pin to the executor")
    ] = "",
    runs_dir: Path = RUNS,
    max_usd: Annotated[
        float | None,
        typer.Option(
            "--max-usd",
            help="stop after the task that takes delivered + improvement spend here (env HM_MAX_USD)",
        ),
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="resume this run dir: done tasks are skipped")] = None,
    dry_run: Annotated[bool, typer.Option(help="print the plan and the stream order; start nothing")] = False,
    no_score: Annotated[bool, typer.Option(help="do not score/aggregate when the stream ends")] = False,
    tag: Annotated[
        str, typer.Option(help="suffix for the Harness² namespace/library, e.g. the attack name")
    ] = "",
):
    """Harness²'s per-task ONLINE protocol over one seed-ordered task stream (`harness2_online`, D14)."""
    import os

    from hm.harness2_online import run_stream
    from hm.harness2_refine import HmBenchmarkAdapter
    from hm.runner import load_spec

    if max_usd is None and os.environ.get("HM_MAX_USD"):
        max_usd = float(os.environ["HM_MAX_USD"])
    adapter = HmBenchmarkAdapter(
        agent_spec=load_spec(spec), suite_ref=suite, runs_root=runs_dir / "harness2", seed=seed,
        train_frac=1.0, proposer_kind=improver_channel, proposer_model=improver_model,
    )  # fmt: skip
    out = run_stream(
        adapter,
        k=k,
        mode=mode,
        runs_dir=runs_dir,
        max_usd=max_usd,
        run_id=run_id,
        dry_run=dry_run,
        tag_suffix=tag,
        log=typer.echo,
    )
    if dry_run:
        typer.echo(json.dumps(out, indent=2, default=str))
        return
    typer.echo(f"run written to {out}")
    if not no_score:
        score(out.name, runs_dir=runs_dir)
        aggregate(out.name, runs_dir=runs_dir)
        _print_matrix(out)


@app.command()
def doctor(runs_dir: Path = RUNS):
    """Host free space, Docker usage and reclaimable space, leftover hm containers."""
    from hm import preflight

    typer.echo(preflight.format_report(preflight.report(runs_dir)))


@app.command()
def clean():
    """Remove containers created by this project (label hm.suite) and leftover claude_sdk episode
    transcripts under Claude Code's projects dir; touches nothing else."""
    from hm import preflight

    ids = preflight.clean()
    sessions = preflight.clean_claude_sessions()
    typer.echo(f"removed {len(ids)} hm container(s), {len(sessions)} claude_sdk transcript dir(s)")


@app.command("schema-export")
def schema_export(out: Path = ROOT / "schema" / "trace.schema.json"):
    """Write the JSON schema of the normalized trace."""
    from hm.types import Trace

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(Trace.model_json_schema(), indent=2) + "\n")
    typer.echo(f"wrote {out}")


def _print_matrix(run_dir: Path) -> None:
    cells = json.loads((run_dir / "cells.json").read_text())
    typer.echo(f"\n{run_dir.name}  (pass^1 [95% CI], targeted ASR)")
    for c in cells:
        if c["metric"] != "pass^1":
            continue
        asr = next(
            (
                x
                for x in cells
                if x["metric"] == "targeted_asr"
                and x["harness"] == c["harness"]
                and x["suite"] == c["suite"]
                and x["benchmark"] == c["benchmark"]
            ),
            None,
        )
        asr_s = f"  ASR {asr['value']:.2f}" if asr and asr["value"] is not None else ""
        typer.echo(
            f"  {c['harness']:<12} {c['benchmark'] + '/' + c['suite']:<16} "
            f"{c['value']:.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]  n={c['n_episodes']}{asr_s}"
        )


if __name__ == "__main__":
    app()
