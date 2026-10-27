import json

import httpx

from hm.models.openai_compat import OpenAICompatServer, openai_messages_to_anthropic, openai_tools_to_defs

TOOLS = [
    {"type": "function", "function": {"name": "hm_kv_get", "description": "get", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
    {"type": "function", "function": {"name": "hm_kv_set", "description": "set", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
]  # fmt: skip


def test_message_conversion_roundtrips_tool_calls():
    system, msgs = openai_messages_to_anthropic(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "hm_kv_get", "arguments": '{"key": "a"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "17"},
        ]
    )  # fmt: skip
    assert system == "sys"
    assert msgs[0] == {"role": "user", "content": "do it"}
    assert msgs[1]["content"][0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "hm_kv_get",
        "input": {"key": "a"},
    }
    assert msgs[2]["content"][0]["type"] == "tool_result" and msgs[2]["content"][0]["content"] == "17"
    assert [t.name for t in openai_tools_to_defs(TOOLS)] == ["hm_kv_get", "hm_kv_set"]


def _prompt():
    return "Read the value at key 'todo' and store its uppercase under key 'todo_upper'."


def test_server_non_streaming_uses_served_tool_names():
    with OpenAICompatServer() as srv:
        r = httpx.post(
            f"{srv.base_url}/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": _prompt()}], "tools": TOOLS},
            timeout=10,
        )
        body = r.json()
        tc = body["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "hm_kv_get"  # prefixed, as served
        assert json.loads(tc["function"]["arguments"]) == {"key": "todo"}
        assert body["choices"][0]["finish_reason"] == "tool_calls" and body["usage"]["prompt_tokens"] > 0


def test_server_streaming_and_request_budget():
    with OpenAICompatServer(max_requests=1) as srv:
        with httpx.stream(
            "POST",
            f"{srv.base_url}/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": _prompt()}],
                "tools": TOOLS,
                "stream": True,
            },
            timeout=10,
        ) as r:
            events = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        # second request exceeds the budget -> plain stop
        r2 = httpx.post(
            f"{srv.base_url}/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": _prompt()}], "tools": TOOLS},
            timeout=10,
        ).json()
        assert (
            r2["choices"][0]["finish_reason"] == "stop" and "budget" in r2["choices"][0]["message"]["content"]
        )
