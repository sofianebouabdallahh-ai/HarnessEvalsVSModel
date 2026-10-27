"""Fairness scorer: manifest completeness and undeclared deviations (metrics `tainted`).

An episode is *tainted* when the comparison it feeds cannot be trusted:
- `model_mismatch`: the harness ran a different model string than the spec (observed in the trace)
- `manifest_incomplete`: the harness manifest still carries a placeholder (nothing was observed)
- `undeclared_tool`: a tool call reached the env-server under a name the benchmark does not serve
- `undeclared_native_tool`: a harness-native tool was used but is not declared in `extra_tools`.
  A native call the harness itself rejected as nonexistent ("No such tool available: …", OpenCode's
  `invalid`) is a failed hallucination, not a capability, and does not taint.

Tainted episodes are excluded by `hm aggregate`. The run-level manifest (re-derived after the run)
is the reference for declarations, because the first episode's per-trace manifest predates
anything the adapter could observe; `score_run` installs it via `set_run_manifest`.
"""

from __future__ import annotations

import re

from hm.benchmarks import get_benchmark
from hm.types import AdapterManifest, RunManifest, Score, ToolCallStep, ToolResultStep, Trace

PLACEHOLDER = "<observed on first run>"
NATIVE_PREFIXES = ("claude_code:", "opencode:")

_run_manifest: RunManifest | None = None


def set_run_manifest(m: RunManifest | None) -> None:
    global _run_manifest
    _run_manifest = m


def _declarations(trace: Trace) -> AdapterManifest:
    if _run_manifest and trace.harness in _run_manifest.harnesses:
        return _run_manifest.harnesses[trace.harness]
    return trace.adapter_manifest


def taint_reasons(trace: Trace) -> list[str]:
    reasons: list[str] = []
    decl = _declarations(trace)
    if trace.adapter_extra.get("model_mismatch"):
        reasons.append("model_mismatch")
    if PLACEHOLDER in decl.extra_tools:
        reasons.append("manifest_incomplete")
    try:
        served = {t.name for t in get_benchmark(trace.benchmark).make_env_tools(trace.task_id)}
    except (KeyError, AttributeError):
        served = None
    declared_native = set(decl.extra_tools)
    results = {r.call_id: r for r in trace.steps if isinstance(r, ToolResultStep) and r.call_id}
    for s in trace.steps:
        if not isinstance(s, ToolCallStep):
            continue
        if s.name.startswith(NATIVE_PREFIXES):
            bare = s.name.split(":", 1)[1]
            if bare == "invalid" or _rejected_as_nonexistent(results.get(s.call_id)):
                continue
            if bare not in declared_native and "undeclared_native_tool" not in reasons:
                reasons.append("undeclared_native_tool")
        elif served is not None and s.name not in served and "undeclared_tool" not in reasons:
            reasons.append("undeclared_tool")
    return reasons


NO_SUCH_TOOL_RE = re.compile(
    r"no such tool|unknown tool|tool .{0,40}(not found|does not exist|is not available)", re.I
)


def _rejected_as_nonexistent(result: ToolResultStep | None) -> bool:
    return bool(result is not None and result.error and NO_SUCH_TOOL_RE.search(result.content or ""))


def score(trace: Trace) -> list[Score]:
    reasons = taint_reasons(trace)
    return [
        Score(
            scorer="fairness",
            name="tainted",
            value=float(bool(reasons) or trace.tainted),
            label=",".join(reasons) if reasons else (trace.taint_reason or None),
        ),
        Score(scorer="fairness", name="platform_retries", value=float(trace.platform_retries)),
    ]
