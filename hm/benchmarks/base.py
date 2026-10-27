"""Benchmark and environment protocols.

A `Benchmark` owns tasks and ground truth. An `Env` is one task's live environment: it
exposes tools (served to the harness by the env-server) and can serialize itself so the
benchmark can grade from the trace alone (`Trace.env_final`), never from a live object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hm.types import Score, Task, ToolDef, ToolResult, Trace


class Env(ABC):
    @abstractmethod
    def tools(self) -> list[ToolDef]: ...

    @abstractmethod
    def call(self, name: str, args: dict[str, Any]) -> ToolResult: ...

    @abstractmethod
    def snapshot(self, final_output: str | None = None) -> dict[str, Any]:
        """JSON-serializable final state, stored in `Trace.env_final` for grading.

        Called once, after the harness has finished, while the env is still live. Benchmarks
        whose state does not survive a dump/validate roundtrip (AgentDojo) compute their
        verdicts here and store them; the scorer then reads verdicts from the trace."""

    @abstractmethod
    def state_hash(self) -> str: ...

    def prompt_addendum(self) -> str | None:
        """Episode-specific text the runner appends to `Task.prompt` before the harness sees it
        (e.g. τ²'s opening customer utterance). Identical for every harness within an episode."""
        return None


class Benchmark(ABC):
    name: str

    @abstractmethod
    def suites(self) -> list[str]: ...

    @abstractmethod
    def tasks(self, suite: str) -> list[Task]: ...

    @abstractmethod
    def make_env(self, task: Task) -> Env: ...

    @abstractmethod
    def score(self, task: Task, trace: Trace) -> list[Score]:
        """Pure: reads `trace.env_final`, `trace.final_output`, `trace.steps` only."""

    def task(self, task_id: str) -> Task:
        for suite in self.suites():
            for t in self.tasks(suite):
                if t.id == task_id:
                    return t
        raise KeyError(task_id)

    def make_env_tools(self, task_id: str) -> list[ToolDef]:
        """The tool set a task serves. Default builds a throwaway env; override when that is costly."""
        return self.make_env(self.task(task_id)).tools()


_REGISTRY: dict[str, Benchmark] = {}


def register(bench: Benchmark) -> Benchmark:
    _REGISTRY[bench.name] = bench
    return bench


_MODULES = {
    "toy": "hm.benchmarks.toy",
    "dojo": "hm.benchmarks.dojo",
    "tau2": "hm.benchmarks.tau2",
    "custom": "hm.benchmarks.custom",
}


def get_benchmark(name: str) -> Benchmark:
    if name not in _REGISTRY:
        import importlib

        if name not in _MODULES:
            raise KeyError(f"unknown benchmark {name!r}; have {sorted(_MODULES)}")
        importlib.import_module(_MODULES[name])  # registers on import
    return _REGISTRY[name]


def resolve_suite(ref: str) -> tuple[Benchmark, str]:
    """'toy/basic' -> (ToyBenchmark, 'basic')."""
    bench_name, _, suite = ref.partition("/")
    bench = get_benchmark(bench_name)
    if suite not in bench.suites():
        raise KeyError(f"{bench_name} has no suite {suite!r}; have {bench.suites()}")
    return bench, suite
