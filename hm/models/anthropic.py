"""Anthropic Messages API backend. One Console key for every adapter.

Sampling: as of anthropic SDK 2.3.0 the Messages API exposes no temperature/top_p/top_k.
`ModelSpec.temperature` is therefore recorded in the manifest but cannot be applied (D12).

Prompt caching (on unless HM_ANTHROPIC_CACHE=0): breakpoints on the last tool definition, the system
prompt and the last block of the last message, so each turn re-reads the conversation prefix at the
cache-read rate instead of full price. Claude Code and OpenCode do the same on their side; without
this the bare loop would pay ~10× more per input token than the other columns on long τ² policies.
Cache reads/writes are recorded per call and priced by hm/pricing.yaml either way.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import anthropic

from hm.models.base import ModelResponse, ToolCall

# Transport-level retries on 429/5xx. Not an agent retry: the model never sees them.
HTTP_MAX_RETRIES = 2


def parse_response(msg: Any) -> ModelResponse:
    text_parts, calls = [], []
    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
    u = msg.usage
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    detail = getattr(u, "cache_creation", None)
    write_1h = (getattr(detail, "ephemeral_1h_input_tokens", 0) or 0) if detail is not None else 0
    return ModelResponse(
        text="\n".join(text_parts) if text_parts else None,
        tool_calls=calls,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=write,
        cache_write_1h_tokens=write_1h,
        stop_reason=msg.stop_reason or "end_turn",
    )


CACHE = {"type": "ephemeral"}


def caching_enabled() -> bool:
    return os.environ.get("HM_ANTHROPIC_CACHE", "1") not in ("0", "false", "no")


def build_request(name: str, system, messages, tools, max_output_tokens: int, cache: bool = True) -> dict:
    """Messages API kwargs. With `cache`, cache_control breakpoints go on the last tool, the system prompt
    and the last content block of the last message; the caller's `messages` are not mutated."""
    kwargs: dict[str, Any] = {"model": name, "max_tokens": max_output_tokens, "messages": messages}
    if tools:
        kwargs["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
        ]
    if system:
        kwargs["system"] = system
    if not cache:
        return kwargs
    if tools:
        kwargs["tools"][-1]["cache_control"] = CACHE
    if system:
        kwargs["system"] = [{"type": "text", "text": system, "cache_control": CACHE}]
    if messages:
        msgs = list(messages)
        last = copy.deepcopy(msgs[-1])
        content = last["content"]
        if isinstance(content, str):
            last["content"] = [{"type": "text", "text": content, "cache_control": CACHE}]
        elif content:
            content[-1]["cache_control"] = CACHE
        msgs[-1] = last
        kwargs["messages"] = msgs
    return kwargs


class AnthropicModel:
    def __init__(self, name: str):
        self.name = name
        self._client = anthropic.Anthropic(max_retries=HTTP_MAX_RETRIES)

    def complete(self, *, system, messages, tools, max_output_tokens, temperature, seed) -> ModelResponse:
        kwargs = build_request(self.name, system, messages, tools, max_output_tokens, cache=caching_enabled())
        return parse_response(self._client.messages.create(**kwargs))
