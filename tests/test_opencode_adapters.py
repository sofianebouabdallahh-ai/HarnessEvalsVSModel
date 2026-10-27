"""OpenCode and Harness² adapters, offline: the real OpenCode 1.14.18 binary driven by the
OpenAI-compatible mock server and the env-server. Skips when the binary is not on disk."""

import json
import os
from pathlib import Path

import pytest

import hm.benchmarks.toy as toy
from hm.adapters import get_adapter
from hm.adapters.harness2 import materialise, tree_digest, tree_dir
from hm.adapters.opencode import build_config, opencode_bin
from hm.models.openai_compat import OpenAICompatServer
from hm.runner import read_manifest, read_traces, run_matrix
from hm.scoring import read_scores, score_run
from hm.types import AgentSpec, Limits, ModelSpec, Task

HAS_BIN = Path(opencode_bin()).is_file()
HAS_TREE = tree_dir().is_dir()
needs_bin = pytest.mark.skipif(not HAS_BIN, reason=f"opencode binary not found at {opencode_bin()}")


def _spec(system_prompt=None) -> AgentSpec:
    return AgentSpec(
        id="t", model=ModelSpec(provider="mock", name="mock"), system_prompt=system_prompt,
        limits=Limits(max_turns=10, max_tool_calls=10, wall_clock_s=240),
    )  # fmt: skip


def test_build_config_disables_builtins_and_mounts_mcp(monkeypatch):
    monkeypatch.setenv("HM_OPENAI_COMPAT_URL", "http://127.0.0.1:1/v1")
    cfg, model_ref, agent = build_config(_spec(), "http://127.0.0.1:2/mcp")
    assert model_ref == "hm_mock/mock" and agent is None
    assert cfg["mcp"]["hm"] == {"type": "remote", "url": "http://127.0.0.1:2/mcp", "enabled": True}
    assert cfg["tools"]["bash"] is False and cfg["permission"]["bash"] == "deny"
    cfg, _, agent = build_config(_spec("shared prompt"), "http://127.0.0.1:2/mcp")
    assert agent == "hm" and cfg["agent"]["hm"]["prompt"] == "shared prompt"


def test_opencode_manifest_offline():
    m = get_adapter("opencode").prepare(
        _spec(), Task(id="toy/basic/0001", benchmark="toy", suite="basic", prompt="p")
    )
    assert (
        m.system_prompt_sha256 is None and "binary_version" in m.extra and "bash" in m.disabled_native_tools
    )


@pytest.mark.skipif(not HAS_TREE, reason="harness2 clone not found")
def test_harness2_materialise_and_manifest(tmp_path):
    info = materialise(tree_dir(), tmp_path)
    assert info["render"] in ("harness2", "fallback")
    # baseline tree: every prose component is comment-only, so no AGENTS.md is written
    assert not (tmp_path / "AGENTS.md").exists() and info["installed_files"] == []
    a = get_adapter("harness2")
    m = a.prepare(_spec(), Task(id="toy/basic/0001", benchmark="toy", suite="basic", prompt="p"))
    assert (
        m.extra["k"] == 0
        and m.extra["refined_from"] is None
        and m.extra["install_mode"] == "workspace_agents_md"
    )
    assert m.extra["harness_files_sha256"] == tree_digest(tree_dir()) and m.system_prompt_sha256
    # a refined tree with real prose renders an AGENTS.md whose hash lands in the manifest
    tree = tmp_path / "tree"
    for rel in (
        "systemprompt.md",
        "guardrails.md",
        "LongTermMEMORY.md",
        "tool_descriptions/tool_guidance.md",
    ):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("")
    (tree / "systemprompt.md").write_text("Always verify before writing.")
    (tree / "hm_meta.json").write_text(json.dumps({"k": 1, "refined_from": "opencode"}))
    ws = tmp_path / "ws"
    ws.mkdir()
    info = materialise(tree, ws)
    assert (ws / "AGENTS.md").read_text().startswith("Always verify")
    assert tree_digest(tree) != tree_digest(tree_dir())


@needs_bin
def test_opencode_binary_end_to_end_on_toy(tmp_path, monkeypatch):
    monkeypatch.setattr(toy, "_TASKS", toy._TASKS[1:3])  # read+write, and sum
    with OpenAICompatServer(max_requests=40) as srv:
        monkeypatch.setenv("HM_OPENAI_COMPAT_URL", srv.base_url)
        rd = run_matrix(_spec(), ["opencode"], ["toy/basic"], 1, tmp_path, log=lambda *_: None)
        score_run(rd)
        traces, scores = read_traces(rd), read_scores(rd)
        assert len(traces) == 2
        for tr in traces:
            s = {x.name: x.value for x in scores[tr.episode_key]}
            assert tr.error is None, tr.error
            assert s["success"] == 1.0
            names = [st.name for st in tr.steps if st.type == "tool_call"]
            assert "kv_get" in names and "kv_set" in names  # canonical names from the env-server log
            assert not any(n.startswith("opencode:") for n in names)  # no built-ins fired
            assert tr.totals.turns >= 2 and tr.totals.input_tokens > 0
            assert tr.adapter_extra["session_id"].startswith("ses_")
        man = read_manifest(rd).harnesses["opencode"]
        assert man.extra_tools == [] and man.extra["binary_version"] == "1.14.18"


@needs_bin
@pytest.mark.skipif(not HAS_TREE, reason="harness2 clone not found")
def test_harness2_binary_end_to_end_on_toy(tmp_path, monkeypatch):
    monkeypatch.setattr(toy, "_TASKS", toy._TASKS[1:2])
    with OpenAICompatServer(max_requests=30) as srv:
        monkeypatch.setenv("HM_OPENAI_COMPAT_URL", srv.base_url)
        rd = run_matrix(_spec(), ["harness2"], ["toy/injected"], 1, tmp_path, log=lambda *_: None)
        score_run(rd)
        tr = read_traces(rd)[0]
        s = {x.name: x.value for x in read_scores(rd)[tr.episode_key]}
        assert tr.error is None, tr.error
        assert s["success"] == 1.0 and s["injection_seen"] == 1.0
        assert tr.adapter_extra["harness2_install"]["render"] == "harness2"
        assert tr.adapter_manifest.extra["harness_files_sha256"]


@needs_bin
def test_opencode_equalized_prompt_uses_custom_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(toy, "_TASKS", toy._TASKS[:1])
    with OpenAICompatServer(max_requests=20) as srv:
        monkeypatch.setenv("HM_OPENAI_COMPAT_URL", srv.base_url)
        rd = run_matrix(_spec(), ["opencode"], ["toy/basic"], 1, tmp_path, mode="equalized_prompt",
                        equalized_prompt="Shared prompt.", log=lambda *_: None)  # fmt: skip
        tr = read_traces(rd)[0]
        assert tr.error is None, tr.error
        assert "--agent" in tr.adapter_extra["argv"] and tr.adapter_manifest.system_prompt_sha256
        assert os.environ.get("HM_OPENAI_COMPAT_URL")  # sanity: env propagated


EXPORT = {
    "info": {"id": "ses_x", "version": "1.14.18"},
    "messages": [
        {
            "info": {"id": "m0", "role": "user", "time": {"created": 1000}},
            "parts": [{"type": "text", "text": "q"}],
        },
        {
            "info": {
                "id": "m1",
                "role": "assistant",
                "finish": "tool-calls",
                "cost": 0.02,
                "tokens": {"input": 4, "output": 61, "reasoning": 0, "cache": {"write": 7935, "read": 0}},
                "time": {"created": 1673, "completed": 3500},
            },  # fmt: skip
            "parts": [
                {"type": "step-start"},
                {
                    "type": "tool",
                    "tool": "hm_get_day_calendar_events",
                    "callID": "c1",
                    "state": {
                        "status": "completed",
                        "input": {"day": "2024-05-15"},
                        "output": "...",
                        "time": {"start": 2000, "end": 2100},
                    },
                },  # fmt: skip
                {"type": "step-finish"},
            ],
        },
        {
            "info": {
                "id": "m2",
                "role": "assistant",
                "finish": "stop",
                "cost": 0.004,
                "tokens": {"input": 2, "output": 197, "reasoning": 0, "cache": {"write": 551, "read": 7935}},
                "time": {"created": 3504, "completed": 6346},
            },  # fmt: skip
            "parts": [
                {"type": "step-start"},
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "c2",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "a",
                        "time": {"start": 4000, "end": 4100},
                    },
                },  # fmt: skip
                {"type": "text", "text": "You have **3 appointments** on May 15th."},
                {"type": "step-finish"},
            ],
        },
    ],
}

TRUNCATED_STREAM = "\n".join(
    json.dumps(e)
    for e in [
        {"type": "step_start", "timestamp": 1673, "sessionID": "ses_x", "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "timestamp": 2000,
            "sessionID": "ses_x",
            "part": {
                "type": "tool",
                "tool": "hm_get_day_calendar_events",
                "callID": "c1",
                "state": {
                    "status": "completed",
                    "input": {"day": "2024-05-15"},
                    "output": "...",
                    "time": {"start": 2000, "end": 2100},
                },
            },
        },  # fmt: skip
        {
            "type": "step_finish",
            "timestamp": 3500,
            "sessionID": "ses_x",
            "part": {
                "type": "step-finish",
                "reason": "tool-calls",
                "cost": 0.02,
                "tokens": {"input": 4, "output": 61, "reasoning": 0, "cache": {"write": 7935, "read": 0}},
            },
        },  # fmt: skip
        {"type": "step_start", "timestamp": 3504, "sessionID": "ses_x", "part": {"type": "step-start"}},
    ]
)


def test_export_recovers_the_reply_the_stream_dropped():
    from hm.adapters.base import AdapterRun
    from hm.adapters.opencode import OpenCodeAdapter, parse_export
    from hm.types import LLMCallStep, ToolCallStep

    assert parse_export("Exporting session: ses_x\n" + json.dumps(EXPORT))["info"]["id"] == "ses_x"
    assert parse_export("nope") is None and parse_export('{"info": {}}') is None

    a = OpenCodeAdapter()
    out = AdapterRun()
    a._parse_events(TRUNCATED_STREAM, out, 1.0)
    assert out.final_output is None and out.extra["session_id"] == "ses_x"
    assert sum(isinstance(s, LLMCallStep) for _, s in out.llm_steps) == 1

    a._reconcile_with_export(EXPORT, out)
    llm = [s for _, s in out.llm_steps if isinstance(s, LLMCallStep)]
    assert len(llm) == 2 and llm[1].output_tokens == 197 and llm[1].cache_read_tokens == 7935
    assert llm[1].stop_reason == "end_turn" and llm[1].latency_ms == 2842
    assert out.final_output == "You have **3 appointments** on May 15th."
    assert out.extra["harness_reported_usd"] == pytest.approx(0.024)
    assert out.extra["opencode_stream_truncated"] == 1 and out.extra["opencode_export_messages"] == 2
    # the native tool call inside the recovered step is recorded (fairness), the MCP one is not
    natives = [s.name for _, s in out.llm_steps if isinstance(s, ToolCallStep)]
    assert natives == ["opencode:bash"] and "bash" in a._native_tools_seen

    # a complete stream is left alone except for the authoritative cost/final text
    out2 = AdapterRun()
    a2 = OpenCodeAdapter()
    a2._parse_events(TRUNCATED_STREAM + "\n" + json.dumps(
        {"type": "text", "timestamp": 6000, "sessionID": "ses_x", "part": {"type": "text", "text": "You have **3 appointments** on May 15th."}}
    ) + "\n" + json.dumps(
        {"type": "step_finish", "timestamp": 6346, "sessionID": "ses_x",
         "part": {"type": "step-finish", "reason": "stop", "cost": 0.004, "tokens": {"input": 2, "output": 197, "cache": {"write": 551, "read": 7935}}}}
    ), out2, 1.0)  # fmt: skip
    a2._reconcile_with_export(EXPORT, out2)
    assert sum(isinstance(s, LLMCallStep) for _, s in out2.llm_steps) == 2
    assert "opencode_stream_truncated" not in out2.extra and out2.extra["opencode_export_messages"] == 2


def test_run_env_isolates_home(tmp_path, monkeypatch):
    from hm.adapters.opencode import run_env

    monkeypatch.setenv("HM_OPENCODE_HOME", str(tmp_path / "ochome"))
    monkeypatch.setenv("HOME", "/Users/someone")
    e = run_env(tmp_path, {}, tmp_path)
    assert e["HOME"] == str(tmp_path / "ochome" / "home") and Path(e["HOME"]).is_dir()
    assert e["XDG_DATA_HOME"] == str(tmp_path / "ochome" / "data")


def test_finish_reasons_map_to_messages_api_vocabulary():
    from hm.adapters.opencode import _finish_to_stop_reason as f

    assert f("tool-calls") == "tool_use" and f("stop") == "end_turn" and f("") == "end_turn"
    assert f("content-filter") == "refusal" and f("length") == "max_tokens" and f("other") == "other"
