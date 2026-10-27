"""Env-server: serves one Env's tools over MCP (streamable HTTP) for the duration of an episode.

Decision D1. Every harness, including `raw_loop`, reaches tools through this process, so:
- the tool-call log recorded here is the authoritative `tool_call`/`tool_result` record,
- `untrusted` / `injection_marker` are set in one place,
- the tool-call budget is enforced identically for every harness.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server

from hm.benchmarks.base import Env
from hm.types import Limits, Step, ToolCallStep, ToolResultStep, sha256_json

SERVER_NAME = "hm"
BUDGET_ERROR = "tool call budget exhausted"


def canonical_tool_name(name: str) -> str:
    """Strip harness-added MCP prefixes: `mcp__hm__kv_get`, `hm_kv_get` -> `kv_get`."""
    for p in (f"mcp__{SERVER_NAME}__", f"{SERVER_NAME}_", f"{SERVER_NAME}."):
        if name.startswith(p):
            return name[len(p) :]
    return name


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EnvServer:
    def __init__(self, env: Env, limits: Limits, host: str = "127.0.0.1"):
        self.env = env
        self.limits = limits
        self.host = host
        self.port = _free_port()
        self.log: list[tuple[float, Step]] = []
        self.tool_calls = 0
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    # ------------------------------------------------------------------ MCP handlers

    async def _on_list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
                for t in self.env.tools()
            ]
        )

    async def _on_call_tool(self, ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = canonical_tool_name(params.name)
        args = dict(params.arguments or {})
        content, is_error = self.call(name, args)
        return types.CallToolResult(content=[types.TextContent(type="text", text=content)], is_error=is_error)

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool against the env and record it. Shared by MCP and in-process callers."""
        t0 = time.time()
        with self._lock:
            self.tool_calls += 1
            over_budget = self.tool_calls > self.limits.max_tool_calls
            i = len(self.log)
            self.log.append((t0, ToolCallStep(i=i, name=name, args=args, args_sha256=sha256_json(args))))
        if over_budget:
            content, error, untrusted, marker = "", BUDGET_ERROR, False, None
        else:
            r = self.env.call(name, args)
            content, error, untrusted, marker = r.content, r.error, r.untrusted, r.injection_marker
        t1 = time.time()
        text = content if error is None else json.dumps({"error": error})
        with self._lock:
            self.log.append(
                (
                    t1,
                    ToolResultStep(
                        i=len(self.log),
                        name=name,
                        content=text,
                        latency_ms=int((t1 - t0) * 1000),
                        error=error,
                        untrusted=untrusted,
                        injection_marker=marker,
                    ),
                )
            )
        return text, error is not None

    # ------------------------------------------------------------------ lifecycle

    def start(self, timeout_s: float = 10.0) -> None:
        mcp_server = Server(SERVER_NAME, on_list_tools=self._on_list_tools, on_call_tool=self._on_call_tool)
        app = mcp_server.streamable_http_app(json_response=True, stateless_http=True, host=self.host)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name=f"envserver-{self.port}", daemon=True)
        self._thread.start()
        deadline = time.time() + timeout_s
        while not self._server.started:
            if time.time() > deadline or not self._thread.is_alive():
                raise RuntimeError("env-server failed to start")
            time.sleep(0.01)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> EnvServer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
