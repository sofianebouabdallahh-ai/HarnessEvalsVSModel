"""Model backend protocol. Messages use the Anthropic Messages shape (role + content blocks)
so the real backend in Phase 1 is a thin wrapper and the mock is a faithful stand-in."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hm.types import ToolDef


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0  # uncached
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    stop_reason: str = "end_turn"


class ModelBackend(Protocol):
    name: str  # exact model string, logged verbatim in every trace

    def complete(
        self,
        *,
        system: str | None,
        messages: list[dict[str, Any]],
        tools: list[ToolDef],
        max_output_tokens: int,
        temperature: float,
        seed: int,
    ) -> ModelResponse: ...
