"""Minimal MCP client used by `raw_loop` (and tests) to reach the env-server."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from hm.types import ToolDef


class McpToolClient:
    def __init__(self, url: str):
        self.url = url
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> McpToolClient:
        await self._stack.__aenter__()
        streams = await self._stack.enter_async_context(streamable_http_client(self.url))
        read, write = streams[0], streams[1]
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.__aexit__(*exc)

    async def list_tools(self) -> list[ToolDef]:
        assert self.session
        res = await self.session.list_tools()
        return [
            ToolDef(name=t.name, description=t.description or "", input_schema=t.input_schema)
            for t in res.tools
        ]

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        assert self.session
        res = await self.session.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
        return text, bool(getattr(res, "is_error", False))
