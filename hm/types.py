"""Core data model. Everything the platform exchanges is one of these.

The trace schema is aligned with OpenTelemetry GenAI attribute
names where one exists. Raw tool args/results are kept in the trace (decision D2);
only hashes go to Postgres.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- spec


class ModelSpec(BaseModel):
    provider: str = "anthropic"
    name: str
    temperature: float = 0.0


class Limits(BaseModel):
    max_turns: int = 30
    max_tool_calls: int = 60
    max_output_tokens: int = 4096
    wall_clock_s: int = 600


class AgentSpec(BaseModel):
    """Intentionally thin: anything richer is a harness property and is measured."""

    id: str
    model: ModelSpec
    tools_source: Literal["benchmark"] = "benchmark"
    limits: Limits = Field(default_factory=Limits)
    system_prompt: str | None = None  # None = harness default; str = equalized override


# --------------------------------------------------------------------------- tasks / tools


class ToolDef(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(BaseModel):
    content: str
    error: str | None = None
    untrusted: bool = False
    injection_marker: str | None = None


class Task(BaseModel):
    id: str  # e.g. "tau2/airline/0017"
    benchmark: str
    suite: str
    prompt: str
    attacked: bool = False
    attack: str | None = None
    injection_task_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- trace


class LLMCallStep(BaseModel):
    type: Literal["llm_call"] = "llm_call"
    i: int
    input_tokens: int  # uncached input only
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0  # all cache writes (5m + 1h)
    cache_write_1h_tokens: int = 0  # subset of cache_write_tokens written with the 1h TTL
    latency_ms: int
    stop_reason: str | None = None
    text: str | None = None


class ToolCallStep(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    i: int
    name: str  # canonical name, harness prefixes (mcp__x__, x_) stripped
    args: dict[str, Any]
    args_sha256: str
    call_id: str | None = None


class ToolResultStep(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    i: int
    call_id: str | None = None
    name: str
    content: str
    latency_ms: int
    error: str | None = None
    untrusted: bool = False
    injection_marker: str | None = None


Step = LLMCallStep | ToolCallStep | ToolResultStep


class AdapterManifest(BaseModel):
    """Every deviation from the raw spec, declared by the adapter."""

    system_prompt_sha256: str | None
    system_prompt_reason: str | None = None  # why the hash is null, if it is
    injected_retries: int = 0
    extra_tools: list[str] = Field(default_factory=list)
    disabled_native_tools: list[str] = Field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)  # e.g. harness2: k, refined_from


class Totals(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    usd: float | None = None
    wall_clock_ms: int = 0


class Score(BaseModel):
    scorer: str
    name: str
    value: float | None
    label: str | None = None
    explanation: str | None = None


class Trace(BaseModel):
    run_id: str
    task_id: str
    benchmark: str
    suite: str
    model: str
    harness: str
    harness_version: str
    seed: int
    mode: Literal["out_of_the_box", "equalized_prompt"] = "out_of_the_box"
    started_at: datetime = Field(default_factory=utcnow)
    attacked: bool = False
    attack: str | None = None
    adapter_manifest: AdapterManifest
    steps: list[Step] = Field(default_factory=list)
    final_output: str | None = None
    env_state_hash: str | None = None
    env_final: dict[str, Any] | None = None  # benchmark-owned serialization used for grading
    totals: Totals = Field(default_factory=Totals)
    tainted: bool = False
    taint_reason: str | None = None
    error: str | None = None  # harness-level failure (timeout, crash); episode counts as failed
    adapter_extra: dict[str, Any] = Field(default_factory=dict)  # observed facts: models seen, native tools…
    platform_retries: int = 0  # episode re-runs after transient provider errors (not agent retries)
    scores: list[Score] = Field(default_factory=list)

    @property
    def episode_key(self) -> str:
        return f"{self.harness}/{self.task_id}/s{self.seed}"

    def recompute_totals(self) -> None:
        t = Totals()
        for s in self.steps:
            if isinstance(s, LLMCallStep):
                t.turns += 1
                t.input_tokens += s.input_tokens
                t.output_tokens += s.output_tokens
                t.cache_read_tokens += s.cache_read_tokens
                t.cache_write_tokens += s.cache_write_tokens
                t.cache_write_1h_tokens += s.cache_write_1h_tokens
            elif isinstance(s, ToolCallStep):
                t.tool_calls += 1
            elif isinstance(s, ToolResultStep) and s.error:
                t.tool_errors += 1
        t.usd = self.totals.usd
        t.wall_clock_ms = self.totals.wall_clock_ms
        self.totals = t


# --------------------------------------------------------------------------- run manifest


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    model: str
    mode: str
    harnesses: dict[str, AdapterManifest]
    tasks: dict[str, int]  # suite -> count
    seeds: int
    user_sim_model: str | None = None
    spec: AgentSpec
    extra: dict[str, Any] = Field(default_factory=dict)
