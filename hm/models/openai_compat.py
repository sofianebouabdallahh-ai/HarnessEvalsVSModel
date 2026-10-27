"""A tiny OpenAI-compatible `/v1/chat/completions` server fronting a `ModelBackend`.

Exists so external harnesses (OpenCode, Harness²'s OpenCode executor) can be driven end to
end with the zero-cost `MockModel`: point the harness's provider `baseURL` here and every
request is answered by the same scripted policy `raw_loop` uses. Supports non-streaming and
SSE streaming responses with OpenAI-style tool calls. Runs in a background thread like the
env-server.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from hm.models.base import ModelBackend, ModelResponse
from hm.models.mock import MockModel
from hm.types import ToolDef


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI chat messages -> (system, Anthropic-shaped messages) for `ModelBackend.complete`."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role in ("system", "developer"):
            system_parts.append(content or "")
        elif role == "user":
            flush_results()
            out.append({"role": "user", "content": content or ""})
        elif role == "assistant":
            flush_results()
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": args})
            out.append({"role": "assistant", "content": blocks or ""})
        elif role == "tool":
            pending_results.append(
                {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": content or ""}
            )
    flush_results()
    return ("\n".join(system_parts) or None), out


def openai_tools_to_defs(tools: list[dict[str, Any]] | None) -> list[ToolDef]:
    defs = []
    for t in tools or []:
        fn = t.get("function") or t
        defs.append(
            ToolDef(
                name=fn["name"],
                description=fn.get("description") or "",
                input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return defs


def response_to_openai(resp: ModelResponse, model: str) -> dict[str, Any]:
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
        }
        for tc in resp.tool_calls
    ]
    message: dict[str, Any] = {"role": "assistant", "content": resp.text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": resp.input_tokens,
            "completion_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens,
        },
    }


def _sse_chunks(resp: ModelResponse, model: str):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict | None = None) -> str:
        body: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    if resp.text:
        yield chunk({"content": resp.text})
    for i, tc in enumerate(resp.tool_calls):
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": ""},
                    }
                ]
            }
        )
        yield chunk({"tool_calls": [{"index": i, "function": {"arguments": json.dumps(tc.args)}}]})
    usage = {
        "prompt_tokens": resp.input_tokens,
        "completion_tokens": resp.output_tokens,
        "total_tokens": resp.input_tokens + resp.output_tokens,
    }
    yield chunk({}, finish="tool_calls" if resp.tool_calls else "stop", usage=usage)
    yield "data: [DONE]\n\n"


class OpenAICompatServer:
    """Serves `/v1/chat/completions` (+ `/v1/models`) for one backend factory.

    `backend_for(model_id)` lets the requested model id select/parametrise the backend; the
    default builds a `MockModel` from the id (so `mock?flake=0.2` works as a model id)."""

    def __init__(
        self,
        backend_for=None,
        host: str = "127.0.0.1",
        max_output_tokens: int = 1024,
        max_requests: int | None = None,
    ):
        self.backend_for = backend_for or (lambda model_id: MockModel(model_id))
        self.host = host
        self.port = _free_port()
        self.max_output_tokens = max_output_tokens
        # Harnesses without a step cap (OpenCode) can loop forever on a confused model; after
        # this many requests the server answers with a plain stop message.
        self.max_requests = max_requests
        self.requests: list[dict[str, Any]] = []
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._backends: dict[str, ModelBackend] = {}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _backend(self, model_id: str) -> ModelBackend:
        if model_id not in self._backends:
            self._backends[model_id] = self.backend_for(model_id)
        return self._backends[model_id]

    async def _completions(self, request: Request):
        body = await request.json()
        model_id = body.get("model") or "mock"
        system, messages = openai_messages_to_anthropic(body.get("messages") or [])
        tools = openai_tools_to_defs(body.get("tools"))
        self.requests.append({"model": model_id, "n_messages": len(messages), "n_tools": len(tools)})
        if self.max_requests is not None and len(self.requests) > self.max_requests:
            resp = ModelResponse(text="Stopping: request budget exhausted.", stop_reason="end_turn")
        else:
            resp = self._backend(model_id).complete(
                system=system,
                messages=messages,
                tools=tools,
                max_output_tokens=int(
                    body.get("max_tokens") or body.get("max_completion_tokens") or self.max_output_tokens
                ),
                temperature=float(body.get("temperature") or 0.0),
                seed=int(body.get("seed") or 0),
            )
        if body.get("stream"):
            return StreamingResponse(_sse_chunks(resp, model_id), media_type="text/event-stream")
        return JSONResponse(response_to_openai(resp, model_id))

    async def _models(self, request: Request):
        return JSONResponse({"object": "list", "data": [{"id": "mock", "object": "model", "owned_by": "hm"}]})

    def start(self, timeout_s: float = 10.0) -> None:
        app = Starlette(
            routes=[
                Route("/v1/chat/completions", self._completions, methods=["POST"]),
                Route("/chat/completions", self._completions, methods=["POST"]),
                Route("/v1/models", self._models, methods=["GET"]),
                Route("/models", self._models, methods=["GET"]),
            ]
        )
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name=f"openai-compat-{self.port}", daemon=True
        )
        self._thread.start()
        deadline = time.time() + timeout_s
        while not self._server.started:
            if time.time() > deadline or not self._thread.is_alive():
                raise RuntimeError("openai-compat server failed to start")
            time.sleep(0.01)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> OpenAICompatServer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
