"""Adapter behaviour that needs no API key."""

import copy
from types import SimpleNamespace

from hm.adapters import get_adapter
from hm.adapters.claude_sdk import DISABLED_NATIVE_TOOLS
from hm.models.anthropic import parse_response
from hm.types import AgentSpec, ModelSpec, Task


def _task():
    return Task(id="toy/basic/0001", benchmark="toy", suite="basic", prompt="p")


def test_claude_sdk_manifest_declares_deviations():
    a = get_adapter("claude_sdk")
    spec = AgentSpec(id="s", model=ModelSpec(name="claude-sonnet-5-5"))
    m = a.prepare(spec, _task())
    assert m.system_prompt_sha256 is None and m.system_prompt_reason
    assert m.extra_tools == ["<observed on first run>"]  # replaced by the init tool list after a run
    assert set(DISABLED_NATIVE_TOOLS) <= set(m.disabled_native_tools) and "Bash" in m.disabled_native_tools
    eq = a.prepare(spec.model_copy(update={"system_prompt": "shared"}), _task())
    assert eq.system_prompt_sha256 is not None and eq.system_prompt_reason is None


def test_raw_loop_manifest_is_empty_prompt():
    m = get_adapter("raw_loop").prepare(AgentSpec(id="s", model=ModelSpec(name="x")), _task())
    assert m.injected_retries == 0 and m.extra_tools == []


def test_anthropic_response_parsing_counts_cache_as_input():
    msg = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hi"),
            SimpleNamespace(type="tool_use", id="t1", name="kv_get", input={"key": "a"}),
        ],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=3, cache_read_input_tokens=7
        ),
        stop_reason="tool_use",
    )
    r = parse_response(msg)
    assert r.text == "hi" and r.tool_calls[0].name == "kv_get" and r.tool_calls[0].args == {"key": "a"}
    assert (r.input_tokens, r.output_tokens, r.stop_reason) == (10, 5, "tool_use")
    assert (r.cache_read_tokens, r.cache_write_tokens, r.cache_write_1h_tokens) == (7, 3, 0)


def test_claude_sdk_reconciles_step_usage_to_result_usage():
    from hm.adapters.claude_sdk import reconcile_usage
    from hm.types import LLMCallStep

    steps = [
        LLMCallStep(
            i=0, input_tokens=2, output_tokens=5, latency_ms=1, cache_read_tokens=5000, cache_write_tokens=400
        ),
        LLMCallStep(
            i=1,
            input_tokens=2,
            output_tokens=3,
            latency_ms=1,
            cache_read_tokens=5000,
            cache_write_tokens=39000,
        ),
    ]
    result_usage = {
        "input_tokens": 4, "output_tokens": 120, "cache_read_input_tokens": 10000,
        "cache_creation_input_tokens": 39400, "cache_creation": {"ephemeral_1h_input_tokens": 39400},
    }  # fmt: skip
    deltas = reconcile_usage(steps, result_usage)
    assert sum(s.output_tokens for s in steps) == 120 and deltas["output_tokens"] == 112
    assert sum(s.cache_write_1h_tokens for s in steps) == 39400
    assert reconcile_usage(steps, None) == {}


def test_cost_scorer_prices_cache_tiers():
    from hm.scoring.cost import usd_for
    from hm.types import AdapterManifest, LLMCallStep, Trace

    tr = Trace(
        run_id="r", task_id="t", benchmark="toy", suite="s", model="claude-sonnet-5-5", harness="h",
        harness_version="0", seed=0, adapter_manifest=AdapterManifest(system_prompt_sha256=None),
        steps=[LLMCallStep(i=0, input_tokens=1_000_000, output_tokens=100_000, latency_ms=1,
                           cache_read_tokens=1_000_000, cache_write_tokens=200_000, cache_write_1h_tokens=100_000)],
    )  # fmt: skip
    tr.recompute_totals()
    # 2.00 (input) + 0.20 (read) + 0.25 (100k 5m writes) + 0.40 (100k 1h writes) + 1.00 (output)
    assert round(usd_for(tr), 4) == 3.85


def test_anthropic_request_has_cache_breakpoints_and_does_not_mutate_messages():
    from hm.models.anthropic import build_request
    from hm.types import ToolDef

    tools = [
        ToolDef(name=f"t{i}", description="d", input_schema={"type": "object", "properties": {}})
        for i in range(2)
    ]
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "t0", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
    ]
    before = copy.deepcopy(messages)
    k = build_request("claude-sonnet-5-5", "be brief", messages, tools, 64)
    assert "cache_control" not in k["tools"][0] and k["tools"][1]["cache_control"] == {"type": "ephemeral"}
    assert k["system"] == [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}]
    assert k["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in k["messages"][0]["content"] and messages == before
    k1 = build_request("m", None, [{"role": "user", "content": "hi"}], [], 64)
    assert k1["messages"][0]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]
    assert "system" not in k1 and "tools" not in k1
    k0 = build_request("m", "s", messages, tools, 64, cache=False)
    assert k0["system"] == "s" and k0["messages"] is messages and "cache_control" not in k0["tools"][-1]


def test_claude_sdk_keeps_the_key_away_from_the_cli_unless_asked(monkeypatch):
    from hm.adapters.claude_sdk import auth_mode, cli_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.delenv("HM_CLAUDE_SDK_AUTH", raising=False)
    env = cli_env()
    assert auth_mode() == "plan" and not any(k.startswith("ANTHROPIC_") for k in env)
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    monkeypatch.setenv("HM_CLAUDE_SDK_AUTH", "key")
    assert auth_mode() == "key" and cli_env()["ANTHROPIC_API_KEY"] == "sk-ant-test"
