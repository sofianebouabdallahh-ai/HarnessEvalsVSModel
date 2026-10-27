"""Figures and the matrix table for the write-up, rendered from `cells.json` (docs/phase7.md, T5).

    uv run python scripts/figures.py --out docs/figures runs/<id> runs/<id> ...

Every run dir needs `cells.json` and `manifest.json` (`hm aggregate` writes the former). Rows are
labelled from the harness name plus what the manifest says about it (Harness² k, online). Nothing
is hard-coded to a run id, so the figures regenerate when runs are added.

Figures (one hue per figure unless two series are compared; identity lives in the row label):
  matrix.png     dojo workspace benign and attacked pass^1 with 95% CIs, one dot per harness
  horizon.png    tool calls per task by (suite, harness), against the paper's benchmark range
  cost.png       $ per solve, and $ per solve with improvement compute counted
  k_scaling.png  Harness² rows by k (0 = baseline tree) on both dojo suites
  tau2.png       τ² airline pass^1 with CIs by harness
  matrix.md      the table behind matrix.png, with CIs and cost
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill): series-1 blue, series-2 orange; text tokens for all text
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PAPER_CALLS = (20.0, 27.4)  # WorkBuddy-Bench / JobBench tool calls per task, Table 7 of the paper

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def load_runs(run_dirs: list[Path], refine_usd: dict[str, float] | None = None) -> list[dict]:
    """`refine_usd` maps a run id to the one-off cost of the refinement that produced its frozen tree;
    it is amortised over the run's episodes into `usd_per_solve_incl_refine` (online rows carry the
    cost per episode already)."""
    rows = []
    for rd in run_dirs:
        manifest = json.loads((rd / "manifest.json").read_text())
        cells = json.loads((rd / "cells.json").read_text())
        by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for c in cells:
            by_key[(c["harness"], c["benchmark"], c["suite"])][c["metric"]] = c
        for (h, b, s), metrics in by_key.items():
            extra = (manifest.get("harnesses") or {}).get(h, {}).get("extra") or {}
            one_off = (refine_usd or {}).get(rd.name)
            if one_off and metrics.get("usd_per_solve") and metrics["usd_per_solve"]["value"] is not None:
                c = metrics["usd_per_solve"]
                n_eps = c["n_episodes"]
                solves = n_eps * (metrics["pass^1"]["value"] or 0) if metrics.get("pass^1") else 0
                share = one_off * (n_eps / sum(x["n_episodes"] for x in cells if x["metric"] == "pass^1"))
                incl = dict(c, value=c["value"] + (share / solves if solves else 0), ci_lo=None, ci_hi=None)
                metrics = {**metrics, "usd_per_solve_incl_refine": incl}
            rows.append(
                {
                    "run": rd.name,
                    "harness": h,
                    "label": label_for(h, extra),
                    "order": order_for(h, extra),
                    "suite": f"{b}/{s}",
                    "metrics": metrics,
                }
            )
    return rows


def label_for(h: str, extra: dict) -> str:
    if h == "harness2":
        k = extra.get("k")
        return "Harness² baseline tree" if k in (0, None) else f"Harness² frozen, k={k}"
    if h == "harness2_online":
        return f"Harness² online, k={extra.get('k', 1)}"
    return {"raw_loop": "bare loop", "claude_sdk": "Claude Code", "opencode": "OpenCode"}.get(h, h)


def order_for(h: str, extra: dict) -> tuple:
    base = {"raw_loop": 0, "claude_sdk": 1, "opencode": 2, "harness2": 3, "harness2_online": 4}.get(h, 9)
    return (base, extra.get("k") or 0)


def val(m: dict, name: str) -> tuple[float | None, float | None, float | None]:
    c = m.get(name)
    return (c["value"], c["ci_lo"], c["ci_hi"]) if c else (None, None, None)


def dot_ci(ax, rows: list[dict], metric: str, color: str, xlim=(0.5, 1.02)) -> None:
    ys = range(len(rows))
    for y, r in zip(ys, rows, strict=True):
        v, lo, hi = val(r["metrics"], metric)
        if v is None:
            continue
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", alpha=0.45)
        ax.plot([v], [y], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def fig_matrix(rows: list[dict], out: Path) -> None:
    benign = sorted([r for r in rows if r["suite"] == "dojo/workspace"], key=lambda r: r["order"])
    attacked = {r["label"]: r for r in rows if r["suite"] == "dojo/workspace_attacked"}
    if not benign:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 + 0.5 * len(benign)), sharey=True)
    dot_ci(axes[0], benign, "pass^1", BLUE)
    axes[0].set_title("benign, pass^1", loc="left", fontsize=10, color=INK2)
    att_rows = [attacked.get(r["label"], {"label": r["label"], "metrics": {}}) for r in benign]
    dot_ci(axes[1], att_rows, "pass^1", BLUE)
    axes[1].set_title("under injection, pass^1", loc="left", fontsize=10, color=INK2)
    fig.suptitle(
        "AgentDojo workspace, 20 tasks × 5 seeds, claude-sonnet-5-5 held constant",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "matrix.png", dpi=180)
    plt.close(fig)


def fig_horizon(rows: list[dict], out: Path) -> None:
    rs = [r for r in rows if r["metrics"].get("tool_calls_per_episode")]
    rs = sorted(rs, key=lambda r: (r["suite"], r["order"]))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.42 * len(rs)))
    labels = [f"{r['suite'].split('/', 1)[1]} · {r['label']}" for r in rs]
    vals = [r["metrics"]["tool_calls_per_episode"]["value"] for r in rs]
    ax.axvspan(*PAPER_CALLS, color=GRID, alpha=0.6, linewidth=0)
    ax.text(
        PAPER_CALLS[0] + 0.3,
        len(rs) - 0.6,
        "Harness² paper\nbenchmarks\n(20–27 calls)",
        color=INK2,
        fontsize=8.5,
        va="bottom",
    )
    ax.barh(range(len(rs)), vals, color=BLUE, height=0.55)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("tool calls per task (mean)")
    ax.set_xlim(0, max(PAPER_CALLS[1] + 2, max(vals) + 2))
    ax.grid(axis="y", visible=False)
    ax.set_title("Execution available to a harness, per suite", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "horizon.png", dpi=180)
    plt.close(fig)


def fig_cost(rows: list[dict], out: Path) -> None:
    rs = sorted(
        [r for r in rows if r["suite"] == "dojo/workspace" and r["metrics"].get("usd_per_solve")],
        key=lambda r: r["order"],
    )
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.45 * len(rs)))
    ys = list(range(len(rs)))
    for y, r in zip(ys, rs, strict=True):
        v = r["metrics"]["usd_per_solve"]["value"]
        ax.plot([v], [y], marker="o", markersize=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
        w = (r["metrics"].get("usd_per_solve_incl_refine") or {}).get("value")
        if w is not None and abs(w - v) > 1e-6:
            ax.plot([v, w], [y, y], color=ORANGE, linewidth=2, alpha=0.5)
            ax.plot(
                [w], [y], marker="o", markersize=8, color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5
            )
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rs])
    ax.invert_yaxis()
    ax.set_xlabel("$ per solved task (price table, log scale)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", color=BLUE, linestyle="none", label="execution only")
    ax.plot([], [], marker="o", color=ORANGE, linestyle="none", label="with improvement compute")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Cost of a solve, benign dojo workspace", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "cost.png", dpi=180)
    plt.close(fig)


def fig_k_scaling(rows: list[dict], out: Path) -> None:
    pts: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for r in rows:
        if r["harness"] not in ("harness2", "opencode"):
            continue
        k = -1 if r["harness"] == "opencode" else r["order"][1]
        v, lo, hi = val(r["metrics"], "pass^1")
        if v is not None:
            pts[r["suite"]].append((k, v, lo, hi))
    if not any(len(v) > 1 for v in pts.values()):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for suite, color, dx in (("dojo/workspace", BLUE, -0.04), ("dojo/workspace_attacked", ORANGE, 0.04)):
        p = sorted(pts.get(suite, []))
        if not p:
            continue
        xs = [x + dx for x, *_ in p]
        for x, (_k, _v, lo, hi) in zip(xs, p, strict=True):
            ax.plot([x, x], [lo, hi], color=color, linewidth=2, alpha=0.45, solid_capstyle="round")
        ax.plot(
            xs, [v for _, v, *_ in p], color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, label=suite.split("/")[1].replace("_", " "),
        )  # fmt: skip
    kmax = max(x for p in pts.values() for x, *_ in p)
    ax.set_xticks(list(range(-1, kmax + 1)))
    ax.set_xticklabels(["OpenCode\nalone", "baseline\ntree"] + [f"k={k}" for k in range(1, kmax + 1)])
    ax.set_xlim(-1.4, kmax + 0.4)
    ax.set_ylabel("pass^1 (95% CI)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Harness² frozen trees by recursion budget", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "k_scaling.png", dpi=180)
    plt.close(fig)


def fig_tau2(rows: list[dict], out: Path) -> None:
    rs = sorted(
        [r for r in rows if r["suite"] == "tau2/airline" and r["metrics"].get("pass^1")],
        key=lambda r: r["order"],
    )
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.5 * len(rs)))
    dot_ci(ax, rs, "pass^1", AQUA, xlim=(0.0, 1.02))
    ax.set_xlabel("pass^1, programmatic reward (DB ∧ communicate)")
    ax.set_title("τ² airline, 20 tasks, Haiku 5.5 customer", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "tau2.png", dpi=180)
    plt.close(fig)


def matrix_md(rows: list[dict], out: Path) -> None:
    def fmt(m, name, pct=False):
        v, lo, hi = val(m, name)
        if v is None:
            return "—"
        if pct:
            return f"{100 * v:.0f}%"
        return f"**{v:.2f}** [{lo:.2f}, {hi:.2f}]" if lo is not None else f"{v:.2f}"

    def num(m, name, prefix="", digits=3):
        v = (m.get(name) or {}).get("value")
        return "—" if v is None else f"{prefix}{v:.{digits}f}"

    head = [
        "harness", "run", "benign pass^1", "under attack pass^1", "ASR", "API refusals (attacked)",
        "calls/task", "$/solve", "$/solve incl. refine",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    benign = sorted([r for r in rows if r["suite"] == "dojo/workspace"], key=lambda r: r["order"])
    attacked = {(r["run"], r["label"]): r for r in rows if r["suite"] == "dojo/workspace_attacked"}
    for r in benign:
        m = r["metrics"]
        am = attacked.get((r["run"], r["label"]), {"metrics": {}})["metrics"]
        cells = [
            r["label"], f"`{r['run'][-4:]}`", fmt(m, "pass^1"), fmt(am, "pass^1"), fmt(am, "targeted_asr"),
            fmt(am, "api_refusal_rate", pct=True), num(m, "tool_calls_per_episode", digits=2),
            num(m, "usd_per_solve", "$"), num(m, "usd_per_solve_incl_refine", "$"),
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    tau = sorted([r for r in rows if r["suite"] == "tau2/airline"], key=lambda r: r["order"])
    if tau:
        head = ["harness", "run", "τ² airline pass^1", "calls/task", "$/solve", "$/solve incl. refine"]
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in tau:
            m = r["metrics"]
            cells = [
                r["label"], f"`{r['run'][-4:]}`", fmt(m, "pass^1"),
                num(m, "tool_calls_per_episode", digits=2),
                num(m, "usd_per_solve", "$"), num(m, "usd_per_solve_incl_refine", "$"),
            ]  # fmt: skip
            lines.append("| " + " | ".join(cells) + " |")
    (out / "matrix.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--refine-usd",
        action="append",
        default=[],
        metavar="RUN_ID=USD",
        help="one-off refinement cost to amortise over that run's episodes (frozen Harness² rows)",
    )
    args = ap.parse_args()
    refine = {k: float(v) for k, v in (x.split("=", 1) for x in args.refine_usd)}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.runs, refine)
    fig_matrix(rows, args.out)
    fig_horizon(rows, args.out)
    fig_cost(rows, args.out)
    fig_k_scaling(rows, args.out)
    fig_tau2(rows, args.out)
    matrix_md(rows, args.out)
    print(f"{len(rows)} cells from {len(args.runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
