import json

import pytest

from hm.benchmarks import get_benchmark
from hm.envserver import BUDGET_ERROR, EnvServer, McpToolClient, canonical_tool_name
from hm.types import Limits, ToolCallStep, ToolResultStep


def test_canonical_tool_name():
    assert canonical_tool_name("mcp__hm__kv_get") == "kv_get"
    assert canonical_tool_name("hm_kv_get") == "kv_get"
    assert canonical_tool_name("kv_get") == "kv_get"


@pytest.mark.asyncio
async def test_envserver_serves_tools_and_logs_calls():
    bench = get_benchmark("toy")
    task = bench.tasks("injected")[1]  # reads 'todo', which carries the injection
    env = bench.make_env(task)
    with EnvServer(env, Limits(max_tool_calls=3)) as srv:
        async with McpToolClient(srv.url) as c:
            tools = await c.list_tools()
            assert {t.name for t in tools} == {"kv_get", "kv_set", "kv_delete", "kv_list"}
            text, err = await c.call("kv_get", {"key": "todo"})
            assert not err and "buy milk" in text and "<IMPORTANT>" in text
            text, err = await c.call("kv_get", {"key": "nope"})
            assert err and json.loads(text)["error"].startswith("no such key")
            await c.call("kv_set", {"key": "x", "value": "1"})
            text, err = await c.call("kv_list", {})
            assert err and BUDGET_ERROR in text  # 4th call exceeds budget of 3

    steps = [s for _, s in srv.log]
    assert [type(s) for s in steps[:2]] == [ToolCallStep, ToolResultStep]
    first_result = steps[1]
    assert first_result.untrusted and first_result.injection_marker == "important_instructions#2"
    assert steps[3].error and not steps[3].untrusted
    assert env.snapshot()["kv"]["x"] == "1"
    assert srv.tool_calls == 4
