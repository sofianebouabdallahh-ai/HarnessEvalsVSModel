"""`harness2`: the `opencode` column plus a frozen Harness² harness tree materialised into the
workspace before the run (decision D5: refine on a train split, freeze, evaluate on held-out).

A Harness² "harness" is a directory tree (c1..c8): systemprompt.md, guardrails.md,
LongTermMEMORY.md, tool_descriptions/tool_guidance.md, skills/*/SKILL.md,
sub_agents/*/AGENT.md, scripts/*.py, plugin/*.ts, harness.yaml. Install mode
`workspace_agents_md` (what Harness² uses for its OpenCode JobBench runs) renders the prose
components into `AGENTS.md` (appended on top of OpenCode's stock prompt) and the rest into
`.opencode/{skills,agent,plugin}` + `scripts/` inside the workspace.

Rendering uses Harness²'s own `render_opencode_tree` when the clone at `HM_HARNESS2_DIR` is
importable (it has no third-party dependencies), so the installed harness is byte-identical to
what Harness² would run; a small re-implementation is the fallback and is flagged in the
manifest (`render: "fallback"`).

Fairness notes (on top of the opencode adapter's):
- `system_prompt_sha256` = sha256 of the composed AGENTS.md body: that IS the prompt layer the
  refinement edits. `extra` carries `k`, `refined_from`, `harness_files_sha256` (digest of the
  whole tree), `install_mode` and the tree path, so a cell can be traced to the exact harness.
- Project config must stay enabled for this column (AGENTS.md, .opencode/ are project files);
  the workspace is platform-created and empty apart from the harness, so nothing else leaks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hm.adapters.base import AdapterRun, EnvHandle
from hm.adapters.opencode import OpenCodeAdapter, run_env
from hm.types import AdapterManifest, AgentSpec, Task, sha256_text

DEFAULT_H2_DIR = "/Users/sarahch/Documents/Repos/HarnessEvalsVSModel/.tools/harness2"
BASELINE_REL = "third_party/jb_runner/harness_baseline_oc"
INSTALL_MODE = "workspace_agents_md"
ALWAYS_ON = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md")
META_FILE = "hm_meta.json"  # sidecar written by `hm refine`: {"k": 1, "refined_from": "opencode", ...}


def harness2_dir() -> Path:
    return Path(os.environ.get("HM_HARNESS2_DIR", DEFAULT_H2_DIR))


def tree_dir() -> Path:
    return Path(os.environ.get("HM_HARNESS2_TREE", str(harness2_dir() / BASELINE_REL)))


def tree_digest(tree: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in tree.rglob("*") if x.is_file() and x.name != META_FILE):
        h.update(str(p.relative_to(tree)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def tree_meta(tree: Path) -> dict[str, Any]:
    p = tree / META_FILE
    if p.is_file():
        return json.loads(p.read_text())
    return {"k": 0, "refined_from": None, "note": "unrefined baseline tree"}


def render_ctx() -> dict[str, str]:
    """Template slots the JobBench harness uses; ours describe the MCP tool workspace."""
    return {
        "env_description": "A workspace whose tools are provided by an MCP server named `hm`; "
        "no shell or filesystem tools are available.",
        "domain_focus": "benchmark task",
        "date": datetime.now(UTC).date().isoformat(),
        "working_directory": "the temp workspace",
    }


def _import_harness2():
    d = harness2_dir()
    if not (d / "harness2" / "__init__.py").is_file():
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        from harness2.benchmarks import jobbench  # noqa: PLC0415

        return jobbench
    except Exception:  # pragma: no cover - import environment dependent
        return None


# ------------------------------------------------------------------ fallback renderer (mirrors jobbench.py)

_TOKEN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _component(staged: Path, rel: str, ctx: dict[str, str]) -> str:
    p = staged / rel
    if not p.is_file():
        return ""
    txt = _TOKEN.sub(lambda m: ctx.get(m.group(1), ""), p.read_text(encoding="utf-8", errors="replace"))
    return _COMMENT.sub("", txt).strip()


def compose_agent_body_fallback(staged: Path, ctx: dict[str, str]) -> str:
    parts = []
    if sp := _component(staged, "systemprompt.md", ctx):
        parts.append(sp)
    if gr := _component(staged, "guardrails.md", ctx):
        parts.append("# Operating Guardrails\n\n" + gr)
    if tg := _component(staged, "tool_descriptions/tool_guidance.md", ctx):
        parts.append(tg)
    if mem := _component(staged, "LongTermMEMORY.md", ctx):
        parts.append("# Long-Term Memory\n\n" + mem)
    scripts = (
        sorted(p for p in (staged / "scripts").glob("*.py") if not p.name.startswith("_"))
        if (staged / "scripts").is_dir()
        else []
    )
    if scripts:
        parts.append(
            "# Helper scripts (run via Bash)\n\n" + "\n".join(f"- `python scripts/{s.name}`" for s in scripts)
        )
    return "\n\n".join(p for p in parts if p.strip()) + "\n"


def render_tree_fallback(staged: Path, render_dir: Path, ctx: dict[str, str]) -> None:
    (render_dir / ".opencode" / "agent").mkdir(parents=True, exist_ok=True)
    (render_dir / ".opencode" / "skills").mkdir(parents=True, exist_ok=True)
    body = compose_agent_body_fallback(staged, ctx)
    if body.strip():
        (render_dir / "AGENTS.md").write_text(body, encoding="utf-8")
    for skill_md in sorted((staged / "skills").glob("*/SKILL.md")) if (staged / "skills").is_dir() else []:
        shutil.copytree(
            skill_md.parent, render_dir / ".opencode" / "skills" / skill_md.parent.name, dirs_exist_ok=True
        )
    for agent_md in (
        sorted((staged / "sub_agents").glob("*/AGENT.md")) if (staged / "sub_agents").is_dir() else []
    ):
        text = agent_md.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?m)^mode:\s*subagent", text):
            text = "---\nmode: subagent\ndescription: delegated subagent\n---\n\n" + text
        (render_dir / ".opencode" / "agent" / f"{agent_md.parent.name}.md").write_text(text, encoding="utf-8")
    for s in sorted((staged / "scripts").glob("*.py")) if (staged / "scripts").is_dir() else []:
        if not s.name.startswith("_"):
            (render_dir / "scripts").mkdir(exist_ok=True)
            shutil.copy(s, render_dir / "scripts" / s.name)
    for p in sorted((staged / "plugin").glob("*.ts")) if (staged / "plugin").is_dir() else []:
        if not p.name.startswith("_"):
            (render_dir / ".opencode" / "plugin").mkdir(parents=True, exist_ok=True)
            shutil.copy(p, render_dir / ".opencode" / "plugin" / p.name)


def materialise(tree: Path, workspace: Path) -> dict[str, Any]:
    """Install the frozen tree into `workspace` (which may already hold task files)."""
    jb = _import_harness2()
    ctx = render_ctx()
    if jb is not None:
        tmp = workspace.parent / (workspace.name + ".render")
        jb.render_opencode_tree(tree, tmp, ctx)  # wipes tmp first; we merge into the workspace
        for p in tmp.rglob("*"):
            if p.is_file():
                dst = workspace / p.relative_to(tmp)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        body = jb.compose_agent_body(tree, ctx)
        render = "harness2"
    else:
        render_tree_fallback(tree, workspace, ctx)
        body = compose_agent_body_fallback(tree, ctx)
        render = "fallback"
    installed = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file())
    return {"render": render, "agents_md_sha256": sha256_text(body), "installed_files": installed}


class Harness2Adapter(OpenCodeAdapter):
    name = "harness2"

    def __init__(self) -> None:
        super().__init__()
        self.tree = tree_dir()
        self.meta = tree_meta(self.tree) if self.tree.is_dir() else {"k": None, "refined_from": None}
        self._digest = tree_digest(self.tree) if self.tree.is_dir() else None
        self._agents_md_sha: str | None = None
        self._render: str | None = None
        self.version = f"{self.version}+h2-{(self._digest or 'missing')[:12]}"

    def prepare(self, spec: AgentSpec, task: Task) -> AdapterManifest:
        base = super().prepare(spec, task)
        if self._agents_md_sha is None and self.tree.is_dir():
            jb = _import_harness2()
            body = (jb.compose_agent_body if jb else compose_agent_body_fallback)(self.tree, render_ctx())
            self._agents_md_sha = sha256_text(body)
            self._render = "harness2" if jb else "fallback"
        equalized = spec.system_prompt is not None
        return base.model_copy(
            update={
                "system_prompt_sha256": self._agents_md_sha,
                "system_prompt_reason": "sha256 of the composed AGENTS.md (the harness's prompt layer), "
                "appended to " + ("the equalized agent prompt" if equalized else "OpenCode's default prompt"),
                "notes": base.notes
                + "; Harness² tree installed in workspace (workspace_agents_md); project config "
                "enabled so AGENTS.md/.opencode load",
                "extra": {
                    **base.extra,
                    "k": self.meta.get("k"),
                    "refined_from": self.meta.get("refined_from"),
                    "harness_files_sha256": self._digest,
                    "install_mode": INSTALL_MODE,
                    "tree_dir": str(self.tree),
                    "render": self._render,
                },
            }
        )

    def run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        if not self.tree.is_dir():
            return AdapterRun(error=f"harness2 tree not found at {self.tree}")
        return self._run(
            spec, task, seed, env, prompt=task.prompt, session=None, workspace_setup=self._install
        )

    def _install(self, workspace: Path) -> dict[str, Any]:
        info = materialise(self.tree, workspace)
        return {"harness2_install": info}

    # Project config carries the installed harness; the opencode adapter disables it, we do not.
    @staticmethod
    def _env(workspace: Path, config: dict[str, Any], cfg_dir: Path) -> dict[str, str]:
        e = run_env(workspace, config, cfg_dir)
        e.pop("OPENCODE_DISABLE_PROJECT_CONFIG", None)
        return e
