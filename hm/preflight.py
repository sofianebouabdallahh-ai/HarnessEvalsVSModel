"""Disk and Docker guard rails.

A run must never take the host down. Three checks, none of which changes what is measured:
- before a run: host free space >= HM_MIN_FREE_GB (default 5), Docker answers if a suite needs it
- during a run: stop gracefully when host free space drops under HM_HARD_FLOOR_GB (default 2)
- `hm doctor` reports host + Docker usage; `hm clean` removes only containers labelled by this project
- Claude Code writes a session transcript under `$CLAUDE_CONFIG_DIR/projects/<cwd slug>/` for every
  episode cwd the `claude_sdk` adapter creates; the adapter removes it after the episode and
  `hm doctor`/`hm clean` report/remove any left behind (the normalized trace already has everything)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

HM_LABEL = "hm.suite"
SDK_SESSION_PREFIX = "hm-claude-sdk-"  # tempdir prefix of every claude_sdk episode cwd
MIN_FREE_GB = float(os.environ.get("HM_MIN_FREE_GB", "5"))
HARD_FLOOR_GB = float(os.environ.get("HM_HARD_FLOOR_GB", "2"))
DOCKER_TIMEOUT_S = 20
GB = 1024**3


class DiskGuardError(RuntimeError):
    pass


def host_free_gb(path: Path | str = ".") -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / GB


def _docker(args: list[str], timeout: int = DOCKER_TIMEOUT_S) -> str | None:
    try:
        p = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def parse_docker_df(text: str) -> list[dict[str, Any]]:
    """`docker system df --format '{{json .}}'` -> rows with Type, Size, Reclaimable (strings as printed)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            rows.append({k: d.get(k) for k in ("Type", "TotalCount", "Active", "Size", "Reclaimable")})
    return rows


def docker_report() -> dict[str, Any]:
    version = _docker(["version", "--format", "{{.Server.Version}}"])
    if version is None:
        return {"available": False}
    df = _docker(["system", "df", "--format", "{{json .}}"], timeout=60)
    leftovers = _docker(["ps", "-aq", "-f", f"label={HM_LABEL}"]) or ""
    return {
        "available": True,
        "version": version.strip(),
        "usage": parse_docker_df(df) if df else [],
        "hm_containers": [c for c in leftovers.split() if c],
    }


def report(runs_dir: Path | str = ".", want_docker: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "host_free_gb": round(host_free_gb(runs_dir), 2),
        "min_free_gb": MIN_FREE_GB,
        "hard_floor_gb": HARD_FLOOR_GB,
    }
    stale = stale_claude_sessions()
    out["claude_sessions"] = {"count": len(stale), "mb": _dir_mb(stale) if stale else 0.0}
    if want_docker:
        out["docker"] = docker_report()
    return out


def check(runs_dir: Path | str = ".", *, need_docker: bool = False, min_free_gb: float | None = None) -> dict:
    """Raise DiskGuardError when a run must not start. Returns the report otherwise."""
    floor = MIN_FREE_GB if min_free_gb is None else min_free_gb
    rep = report(runs_dir, want_docker=need_docker)
    if rep["host_free_gb"] < floor:
        raise DiskGuardError(
            f"host has {rep['host_free_gb']:.1f} GB free, below HM_MIN_FREE_GB={floor:g}; "
            "free space (see `hm doctor`) or lower the threshold deliberately"
        )
    if need_docker and not rep["docker"]["available"]:
        raise DiskGuardError("a selected suite needs Docker and the engine does not answer within 20s")
    return rep


def below_hard_floor(runs_dir: Path | str = ".") -> bool:
    return host_free_gb(runs_dir) < HARD_FLOOR_GB


def clean() -> list[str]:
    """Remove this project's labelled containers (never anything else)."""
    ids = (_docker(["ps", "-aq", "-f", f"label={HM_LABEL}"]) or "").split()
    if ids:
        _docker(["rm", "-f", *ids], timeout=120)
    return ids


# ---- Claude Code session transcripts (host side)


def claude_projects_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base) if base else Path.home() / ".claude") / "projects"


def claude_session_dirs(cwd: str | Path) -> list[Path]:
    """Where Claude Code keeps the transcript for a working directory: the path with every
    non-alphanumeric character replaced by '-'. The CLI sees the resolved path (`/var` -> `/private/var`
    on macOS), so both spellings are returned."""
    root = claude_projects_dir()
    return [root / re.sub(r"[^A-Za-z0-9]", "-", p) for p in dict.fromkeys((str(cwd), os.path.realpath(cwd)))]


def remove_claude_session(cwd: str | Path) -> list[Path]:
    """Delete the transcript Claude Code wrote for one of our per-episode cwds.
    Touches only directories whose name carries SDK_SESSION_PREFIX."""
    removed = []
    for d in claude_session_dirs(cwd):
        if SDK_SESSION_PREFIX in d.name and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


def stale_claude_sessions() -> list[Path]:
    root = claude_projects_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(f"*{SDK_SESSION_PREFIX}*") if p.is_dir())


def clean_claude_sessions() -> list[Path]:
    """Remove every leftover claude_sdk episode transcript (never any other project directory)."""
    stale = stale_claude_sessions()
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)
    return stale


def _dir_mb(paths: list[Path]) -> float:
    total = 0
    for d in paths:
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return round(total / 1024**2, 1)


def format_report(rep: dict[str, Any]) -> str:
    lines = [
        f"host free: {rep['host_free_gb']} GB  "
        f"(run threshold {rep['min_free_gb']:g} GB, hard floor {rep['hard_floor_gb']:g} GB)"
    ]
    cs = rep.get("claude_sessions")
    if cs and cs["count"]:
        lines.append(
            f"claude_sdk transcripts left in {claude_projects_dir()}: "
            f"{cs['count']} ({cs['mb']} MB)  (run `hm clean`)"
        )
    d = rep.get("docker")
    if d is None:
        return "\n".join(lines)
    if not d["available"]:
        lines.append("docker: engine not answering (wedged or stopped)")
        return "\n".join(lines)
    lines.append(f"docker {d['version']}:")
    for row in d["usage"]:
        lines.append(f"  {row['Type']:<14} {row['Size']:>10}  reclaimable {row['Reclaimable']}")
    n = len(d["hm_containers"])
    lines.append(f"  hm containers: {n}" + ("  (run `hm clean`)" if n else ""))
    return "\n".join(lines)
