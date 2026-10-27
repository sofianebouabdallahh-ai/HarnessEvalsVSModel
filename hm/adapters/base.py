"""Harness adapter interface.

An adapter runs one episode against tools served by the env-server and reports only what
the env-server cannot see: LLM calls, final output, and its own deviations (manifest).
Tool calls/results come from the env-server log, so adapters cannot under-report them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hm.types import AdapterManifest, AgentSpec, LLMCallStep, Task, ToolDef


@dataclass
class EnvHandle:
    url: str
    tools: list[ToolDef]


@dataclass
class AdapterRun:
    llm_steps: list[tuple[float, LLMCallStep]] = field(default_factory=list)  # (unix ts, step)
    final_output: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


class HarnessAdapter(Protocol):
    name: str
    version: str

    def prepare(self, spec: AgentSpec, task: Task) -> AdapterManifest:
        """Declare every deviation from the raw spec: system prompt, retries, extra tools."""
        ...

    def run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        """Execute the task. Must not score."""
        ...

    def teardown(self) -> None: ...


def get_adapter(name: str) -> HarnessAdapter:
    if name == "raw_loop":
        from hm.adapters.raw_loop import RawLoopAdapter

        return RawLoopAdapter()
    if name == "claude_sdk":
        from hm.adapters.claude_sdk import ClaudeSDKAdapter

        return ClaudeSDKAdapter()
    if name == "opencode":
        from hm.adapters.opencode import OpenCodeAdapter

        return OpenCodeAdapter()
    if name == "harness2":
        from hm.adapters.harness2 import Harness2Adapter

        return Harness2Adapter()
    raise KeyError(f"unknown harness {name!r}; have raw_loop, claude_sdk, opencode, harness2")
