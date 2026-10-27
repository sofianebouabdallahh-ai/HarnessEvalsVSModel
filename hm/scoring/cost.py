"""Cost scorer. USD from the dated price table with prompt-cache accounting:
uncached input × p_in, cache reads × 0.1·p_in, 5m cache writes × 1.25·p_in, 1h cache writes × 2·p_in,
output × p_out. Multipliers are Anthropic's published ones (pricing.yaml `cache_multipliers`).
When the harness reports its own cost (Claude Code does), it is emitted as `usd_harness_reported`
next to ours so the two can be compared; the platform number is the one aggregated."""

from __future__ import annotations

import fnmatch
from functools import lru_cache
from pathlib import Path

import yaml

from hm.types import Score, Trace

PRICING = Path(__file__).resolve().parent.parent / "pricing.yaml"
DEFAULT_MULTIPLIERS = {"cache_read": 0.1, "cache_write_5m": 1.25, "cache_write_1h": 2.0}


@lru_cache
def _table() -> dict:
    return yaml.safe_load(PRICING.read_text())


def price_for(model: str) -> dict | None:
    return next((r for r in _table()["prices"] if fnmatch.fnmatch(model, r["pattern"])), None)


def usd_for(trace: Trace) -> float | None:
    p = price_for(trace.model)
    if p is None:
        return None
    m = {**DEFAULT_MULTIPLIERS, **_table().get("cache_multipliers", {})}
    t = trace.totals
    write_5m = t.cache_write_tokens - t.cache_write_1h_tokens
    in_equiv = (
        t.input_tokens
        + t.cache_read_tokens * m["cache_read"]
        + write_5m * m["cache_write_5m"]
        + t.cache_write_1h_tokens * m["cache_write_1h"]
    )
    return in_equiv * p["input"] / 1e6 + t.output_tokens * p["output"] / 1e6


def score(trace: Trace) -> list[Score]:
    out: list[Score] = []
    usd = usd_for(trace)
    if usd is None:
        out.append(
            Score(
                scorer="cost",
                name="usd",
                value=None,
                explanation=f"no price for {trace.model!r} in pricing.yaml",
            )
        )
    else:
        out.append(
            Score(scorer="cost", name="usd", value=round(usd, 6), label=f"pricing as of {_table()['as_of']}")
        )
    reported = trace.adapter_extra.get("harness_reported_usd")
    if reported is not None:
        out.append(Score(scorer="cost", name="usd_harness_reported", value=round(float(reported), 6)))
    online = (trace.adapter_extra or {}).get("harness2_online") or {}
    if online.get("usd_improver") is not None:
        out.append(
            Score(
                scorer="cost",
                name="usd_improver",
                value=round(float(online["usd_improver"]), 6),
                label="harness2 ledger + extra rollouts",
                explanation=f"improver sessions ${online.get('usd_sessions', 0):.4f} + "
                f"{online.get('rollouts', 0) - 1} non-delivered rollouts "
                f"${online.get('usd_extra_rollouts', 0):.4f}",
            )
        )
    return out
