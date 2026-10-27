"""`claude_sdk`: Claude Code via the Claude Agent SDK (bundled CLI, no Node needed).

What is the harness here: Claude Code's default system prompt, its loop, retries, planning
(TodoWrite) and whatever else the CLI does between turns. What is held constant: the model
string, the benchmark tools (served by the env-server over MCP), the limits. Native file,
shell, web and subagent tools are disabled on sim benchmarks (decision D8) and declared.

Fairness notes recorded in the manifest:
- Claude Code's default system prompt is not retrievable through the SDK, so
  `system_prompt_sha256` is null with a reason, and the CLI version is recorded instead.
- User settings and project CLAUDE.md files are not loaded (`setting_sources=[]`, empty cwd).
- Each episode gets a fresh temp cwd; the session transcript Claude Code writes for it under
  `~/.claude/projects/` is removed after the episode (the normalized trace is the record).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import os
import tempfile
import time

from hm import preflight
from hm.adapters.base import AdapterRun, EnvHandle
from hm.envserver.server import SERVER_NAME
from hm.types import AdapterManifest, AgentSpec, LLMCallStep, Task, ToolCallStep, ToolResultStep, sha256_text

# Belt and braces: `tools=[]` removes the built-in set; this list additionally denies the
# tools that would matter most if a CLI version ignored the empty list.
DISABLED_NATIVE_TOOLS = [
    "Bash", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "Skill", "KillShell", "BashOutput", "LS",
]  # fmt: skip


def _sdk_version() -> str:
    try:
        return importlib.metadata.version("claude-agent-sdk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


USAGE_FIELDS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_write_tokens": "cache_creation_input_tokens",
}


def _usage_to_fields(u: dict) -> dict[str, int]:
    f = {k: int(u.get(src) or 0) for k, src in USAGE_FIELDS.items()}
    f["cache_write_1h_tokens"] = int((u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0)
    return f


def reconcile_usage(steps: list[LLMCallStep], result_usage: dict | None) -> dict[str, int]:
    """Make per-step token sums equal the session usage the CLI reports in its ResultMessage.

    The SDK streams one AssistantMessage per content block, each carrying a usage snapshot, so the
    per-message numbers over-count. The ResultMessage usage is the session total (it reproduces the
    CLI's own cost figure exactly). Any difference is applied to the last step; the deltas are returned
    so they can be recorded. Steps are left untouched when there is no result usage."""
    if not result_usage or not steps:
        return {}
    target = _usage_to_fields(result_usage)
    deltas = {}
    for field, total in target.items():
        have = sum(getattr(s, field) for s in steps)
        if have != total:
            deltas[field] = total - have
            setattr(steps[-1], field, max(0, getattr(steps[-1], field) + total - have))
    return deltas


@contextlib.contextmanager
def _episode_cwd():
    """Empty working directory for one episode. Claude Code keys its on-disk session transcript by this
    path; the transcript is deleted with the directory so a run leaves nothing behind on the host."""
    with tempfile.TemporaryDirectory(prefix=preflight.SDK_SESSION_PREFIX) as cwd:
        try:
            yield cwd
        finally:
            preflight.remove_claude_session(cwd)


def auth_mode() -> str:
    """`plan` (default): the CLI uses the Claude plan login and no ANTHROPIC_* variable reaches it, even
    if ANTHROPIC_API_KEY is exported for the other columns. `key`: HM_CLAUDE_SDK_AUTH=key forwards the
    ANTHROPIC_* variables so this column bills the Console key like the others (write-up check)."""
    return "key" if os.environ.get("HM_CLAUDE_SDK_AUTH", "plan").lower() == "key" else "plan"


def cli_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.startswith("ANTHROPIC_")} if auth_mode() == "key" else {}
    env.update({"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"})
    return env


class ClaudeSDKAdapter:
    name = "claude_sdk"

    def __init__(self) -> None:
        self.version = f"sdk-{_sdk_version()}"
        # Learned from the first episode's init message; the runner re-derives the run manifest
        # after the run so these land there too.
        self._cli_version: str | None = None
        self._native_tools: list[str] | None = None

    def prepare(self, spec: AgentSpec, task: Task) -> AdapterManifest:
        sp = spec.system_prompt
        return AdapterManifest(
            system_prompt_sha256=sha256_text(sp) if sp is not None else None,
            system_prompt_reason=None
            if sp is not None
            else "Claude Code default system prompt; not retrievable via SDK, CLI version recorded",
            injected_retries=0,
            extra_tools=self._native_tools if self._native_tools is not None else ["<observed on first run>"],
            disabled_native_tools=DISABLED_NATIVE_TOOLS,
            notes=f"auth: {auth_mode()} (HM_CLAUDE_SDK_AUTH); "
            "Claude Code loop; user settings and CLAUDE.md not loaded; empty cwd; built-in tool set "
            "emptied (tools=[]), survivors listed in extra_tools; benchmark tools over MCP (http); "
            "MCP tools may be deferred behind ToolSearch by the CLI; sampling: provider default",
            extra={"sdk_version": _sdk_version(), "cli_version": self._cli_version},
        )

    def run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        return asyncio.run(self._run(spec, task, seed, env))

    async def _run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        out = AdapterRun()
        mcp_prefix = f"mcp__{SERVER_NAME}__"
        with _episode_cwd() as cwd:
            options = ClaudeAgentOptions(
                model=spec.model.name,
                system_prompt=spec.system_prompt,  # None -> Claude Code default (out_of_the_box)
                mcp_servers={SERVER_NAME: {"type": "http", "url": env.url}},
                tools=[],  # no built-in tools (D8); whatever the CLI keeps anyway is recorded
                allowed_tools=[mcp_prefix + t.name for t in env.tools],
                disallowed_tools=DISABLED_NATIVE_TOOLS,
                permission_mode="bypassPermissions",
                max_turns=spec.limits.max_turns,
                cwd=cwd,
                setting_sources=[],
                env=cli_env(),
            )
            observed_models: set[str] = set()
            pending_native: dict[str, tuple[float, str]] = {}
            steps_by_msg: dict[str, LLMCallStep] = {}  # one LLM call per API message id
            last_text: str | None = None
            t_last = time.time()
            result_usage: dict | None = None

            async def consume() -> None:
                nonlocal last_text, t_last, result_usage
                async for msg in query(prompt=task.prompt, options=options):
                    now = time.time()
                    if isinstance(msg, SystemMessage) and msg.subtype == "init":
                        self._cli_version = msg.data.get("claude_code_version") or msg.data.get("version")
                        native = [t for t in (msg.data.get("tools") or []) if not t.startswith("mcp__")]
                        self._native_tools = native
                        out.extra["init"] = {
                            k: msg.data.get(k)
                            for k in ("model", "permissionMode", "mcp_servers", "apiKeySource")
                        }
                        out.extra["native_tools"] = native
                        out.extra["cli_version"] = self._cli_version
                    elif isinstance(msg, AssistantMessage):
                        observed_models.add(msg.model)
                        usage = _usage_to_fields(msg.usage or {})
                        texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                        tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
                        stop = msg.stop_reason or ("tool_use" if tool_uses else "end_turn")
                        key = msg.message_id or f"anon-{len(steps_by_msg)}"
                        step = steps_by_msg.get(key)
                        if step is None:
                            step = LLMCallStep(
                                i=0, latency_ms=int((now - t_last) * 1000), stop_reason=stop, **usage
                            )
                            steps_by_msg[key] = step
                            out.llm_steps.append((t_last, step))
                        else:  # same API call, next content block: latest usage snapshot, merged text
                            for f, v in usage.items():
                                setattr(step, f, v)
                            step.stop_reason = stop
                            step.latency_ms = int((now - t_last) * 1000)
                        if texts:
                            step.text = "\n".join(filter(None, [step.text, *texts]))
                            last_text = step.text
                        # Native tools never reach the env-server; record them so they are not invisible.
                        for b in tool_uses:
                            if not b.name.startswith(mcp_prefix):
                                pending_native[b.id] = (now, b.name)
                                out.llm_steps.append(
                                    (
                                        now,
                                        ToolCallStep(
                                            i=0,
                                            name=f"claude_code:{b.name}",
                                            args=dict(b.input),
                                            args_sha256=sha256_text(str(sorted(b.input.items()))),
                                            call_id=b.id,
                                        ),  # fmt: skip
                                    )
                                )
                    elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                        for b in msg.content:
                            if isinstance(b, ToolResultBlock) and b.tool_use_id in pending_native:
                                t0, name = pending_native.pop(b.tool_use_id)
                                content = b.content if isinstance(b.content, str) else str(b.content)
                                out.llm_steps.append(
                                    (
                                        now,
                                        ToolResultStep(
                                            i=0,
                                            call_id=b.tool_use_id,
                                            name=f"claude_code:{name}",
                                            content=content,
                                            latency_ms=int((now - t0) * 1000),
                                            error="tool error" if b.is_error else None,
                                        ),  # fmt: skip
                                    )
                                )
                    elif isinstance(msg, ResultMessage):
                        out.final_output = msg.result if msg.result is not None else last_text
                        out.extra["harness_reported_usd"] = msg.total_cost_usd
                        out.extra["num_turns"] = msg.num_turns
                        out.extra["session_id"] = msg.session_id
                        result_usage = msg.usage or None
                        out.extra["result_usage"] = result_usage
                        if msg.is_error:
                            detail = (msg.result or "")[:200] or (str(msg.errors)[:200] if msg.errors else "")
                            out.error = f"claude_sdk result error: {msg.subtype}: {detail}"
                        elif msg.subtype == "error_max_turns":
                            out.error = "max turns exceeded"
                    t_last = now

            try:
                await asyncio.wait_for(consume(), timeout=spec.limits.wall_clock_s)
            except TimeoutError:
                out.error = "wall clock exceeded"
            out.extra["observed_models"] = sorted(observed_models)
            if observed_models and observed_models != {spec.model.name}:
                out.extra["model_mismatch"] = True
            deltas = reconcile_usage(
                [s for _, s in out.llm_steps if isinstance(s, LLMCallStep)], result_usage
            )
            if deltas:
                out.extra["usage_reconciled"] = deltas
        return out

    def teardown(self) -> None:
        pass
