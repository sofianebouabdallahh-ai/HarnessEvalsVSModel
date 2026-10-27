import pytest

from hm import preflight
from hm.benchmarks.custom import IMAGE, WORK, create_argv


def test_check_raises_below_threshold(tmp_path):
    free = preflight.host_free_gb(tmp_path)
    with pytest.raises(preflight.DiskGuardError):
        preflight.check(tmp_path, min_free_gb=free + 1)
    rep = preflight.check(tmp_path, min_free_gb=0)
    assert rep["host_free_gb"] >= 0 and "docker" not in rep


def test_parse_docker_df():
    text = (
        '{"Type":"Images","TotalCount":"50","Active":"18","Size":"25.03GB","Reclaimable":"9.543GB (38%)"}\n'
        '{"Type":"Build Cache","TotalCount":"710","Active":"0","Size":"27.18GB","Reclaimable":"17.31GB"}\n'
    )
    rows = preflight.parse_docker_df(text)
    assert [r["Type"] for r in rows] == ["Images", "Build Cache"]
    assert rows[1]["Reclaimable"] == "17.31GB"


def test_format_report_without_docker():
    out = preflight.format_report({"host_free_gb": 4.2, "min_free_gb": 5, "hard_floor_gb": 2})
    assert "4.2 GB" in out and "threshold 5" in out


def test_custom_container_cannot_grow_or_leak():
    argv = create_argv("hm-custom-test")
    joined = " ".join(argv)
    assert "--read-only" in argv and "--network none" in joined
    assert f"--tmpfs {WORK}:rw,exec,size=" in joined and "--tmpfs /tmp:" in joined
    assert "--memory" in argv and "--pids-limit" in argv and "--log-driver none" in joined
    assert "--label hm.suite=custom" in joined
    assert argv[-3:] == [IMAGE, "sleep", "infinity"]


def test_run_matrix_refuses_without_space(tmp_path, monkeypatch):
    from hm.runner import run_matrix
    from hm.types import AgentSpec, ModelSpec

    monkeypatch.setattr(preflight, "MIN_FREE_GB", preflight.host_free_gb(tmp_path) + 1)
    with pytest.raises(preflight.DiskGuardError):
        run_matrix(
            AgentSpec(id="t", model=ModelSpec(provider="mock", name="mock")),
            ["raw_loop"],
            ["toy/basic"],
            1,
            tmp_path,
            log=lambda *_: None,
        )


def test_run_matrix_stops_gracefully_under_hard_floor(tmp_path, monkeypatch):
    from hm.runner import read_manifest, read_traces, run_matrix
    from hm.types import AgentSpec, ModelSpec

    calls = {"n": 0}

    def fake_floor(_):
        calls["n"] += 1
        return calls["n"] > 2  # first two episodes run, then the host is "full"

    monkeypatch.setattr(preflight, "below_hard_floor", fake_floor)
    rd = run_matrix(
        AgentSpec(id="t", model=ModelSpec(provider="mock", name="mock")),
        ["raw_loop"],
        ["toy/basic"],
        1,
        tmp_path,
        log=lambda *_: None,
    )
    assert len(read_traces(rd)) == 2
    m = read_manifest(rd)
    assert "stopped_early" in m.extra and m.extra["episodes_written"] == 2


def test_claude_session_dirs_follow_claude_code_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    cwd = tmp_path / "T" / "hm-claude-sdk-ab_c.1"
    cwd.mkdir(parents=True)
    dirs = preflight.claude_session_dirs(cwd)
    assert dirs[0].parent == tmp_path / "cfg" / "projects"
    assert dirs[0].name == "".join(c if c.isalnum() else "-" for c in str(cwd))
    assert "hm-claude-sdk-ab-c-1" in dirs[0].name


def test_remove_claude_session_only_touches_our_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    ours = tmp_path / "T" / "hm-claude-sdk-xyz12345"
    theirs = tmp_path / "T" / "someone-elses-repo"
    for cwd in (ours, theirs):
        cwd.mkdir(parents=True)
        d = preflight.claude_session_dirs(cwd)[0]
        d.mkdir(parents=True)
        (d / "s.jsonl").write_text("{}")
    assert preflight.remove_claude_session(theirs) == []  # not ours: untouched
    assert preflight.claude_session_dirs(theirs)[0].is_dir()
    removed = preflight.remove_claude_session(ours)
    assert removed and not removed[0].exists()
    # stale sweep sees only our prefix as well
    stale_cwd = tmp_path / "T" / "hm-claude-sdk-old00000"
    sd = preflight.claude_session_dirs(stale_cwd)[0]
    sd.mkdir()
    (sd / "s.jsonl").write_text("x" * 200_000)
    rep = preflight.report(tmp_path, want_docker=False)
    assert rep["claude_sessions"]["count"] == 1 and rep["claude_sessions"]["mb"] > 0
    assert "claude_sdk transcripts left" in preflight.format_report(rep)
    assert preflight.clean_claude_sessions() == [sd] and not sd.exists()
    assert preflight.claude_session_dirs(theirs)[0].is_dir()
    assert preflight.report(tmp_path, want_docker=False)["claude_sessions"] == {"count": 0, "mb": 0.0}


def test_claude_sdk_episode_cwd_removes_its_transcript(tmp_path, monkeypatch):
    from hm.adapters.claude_sdk import _episode_cwd

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    with _episode_cwd() as cwd:
        d = preflight.claude_session_dirs(cwd)[-1]  # the CLI writes under the resolved path
        d.mkdir(parents=True)
        (d / "session.jsonl").write_text("{}")
    assert not d.exists()


def test_run_matrix_stops_at_spend_cap(tmp_path, monkeypatch):
    import hm.runner as runner
    from hm.runner import read_manifest, read_traces, run_matrix
    from hm.types import AgentSpec, ModelSpec

    monkeypatch.setattr(runner, "_episode_usd", lambda tr: 0.6)
    monkeypatch.delenv("HM_MAX_USD", raising=False)
    rd = run_matrix(
        AgentSpec(id="t", model=ModelSpec(provider="mock", name="mock")),
        ["raw_loop"],
        ["toy/basic"],
        1,
        tmp_path,
        log=lambda *_: None,
        max_usd=1.0,
    )
    assert len(read_traces(rd)) == 2  # 0.6, 1.2 >= cap -> stop after the second episode
    m = read_manifest(rd)
    assert "spend cap" in m.extra["stopped_early"] and m.extra["usd_table"] == 1.2
    # env var is the default; unset means no cap and the total is still recorded
    monkeypatch.setenv("HM_MAX_USD", "0.5")
    rd2 = run_matrix(
        AgentSpec(id="t", model=ModelSpec(provider="mock", name="mock")),
        ["raw_loop"], ["toy/basic"], 1, tmp_path, log=lambda *_: None,
    )  # fmt: skip
    assert len(read_traces(rd2)) == 1
