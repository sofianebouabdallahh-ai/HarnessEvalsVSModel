"""`raw_loop`: the floor. messages -> model -> tool -> repeat. No system prompt, no retries,
no memory, no planning. Any harness that does not beat this is noise."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hm import __version__
from hm.adapters.base import AdapterRun, EnvHandle
from hm.envserver.client import McpToolClient
from hm.models import make_backend
from hm.types import AdapterManifest, AgentSpec, LLMCallStep, Task, sha256_text


class RawLoopAdapter:
    name = "raw_loop"
    version = __version__

    def prepare(self, spec: AgentSpec, task: Task) -> AdapterManifest:
        sp = spec.system_prompt
        return AdapterManifest(
            system_prompt_sha256=sha256_text(sp) if sp is not None else sha256_text(""),
            injected_retries=0,
            extra_tools=[],
            notes="bare tool loop; system prompt is empty unless equalized; "
            "sampling: provider default (Messages API exposes no temperature); "
            "prompt caching: breakpoints on tools/system/last message (HM_ANTHROPIC_CACHE)",
        )

    def run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        return asyncio.run(self._run(spec, task, seed, env))

    async def _run(self, spec: AgentSpec, task: Task, seed: int, env: EnvHandle) -> AdapterRun:
        model = make_backend(spec.model)
        out = AdapterRun()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        deadline = time.time() + spec.limits.wall_clock_s
        async with McpToolClient(env.url) as client:
            tools = await client.list_tools()
            for _turn in range(spec.limits.max_turns):
                if time.time() > deadline:
                    out.error = "wall clock exceeded"
                    break
                t0 = time.time()
                try:
                    resp = model.complete(
                        system=spec.system_prompt,
                        messages=messages,
                        tools=tools,
                        max_output_tokens=spec.limits.max_output_tokens,
                        temperature=spec.model.temperature,
                        seed=seed,
                    )
                except Exception as e:  # provider error: fail the episode with the real message
                    out.error = f"model error: {type(e).__name__}: {str(e)[:300]}"
                    return out
                out.llm_steps.append(
                    (
                        t0,
                        LLMCallStep(
                            i=0,
                            input_tokens=resp.input_tokens,
                            output_tokens=resp.output_tokens,
                            cache_read_tokens=resp.cache_read_tokens,
                            cache_write_tokens=resp.cache_write_tokens,
                            cache_write_1h_tokens=resp.cache_write_1h_tokens,
                            latency_ms=int((time.time() - t0) * 1000),
                            stop_reason=resp.stop_reason,
                            text=resp.text,
                        ),
                    )
                )
                if not resp.tool_calls:
                    out.final_output = resp.text
                    return out
                assistant: list[dict[str, Any]] = []
                if resp.text:
                    assistant.append({"type": "text", "text": resp.text})
                assistant += [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                    for tc in resp.tool_calls
                ]
                messages.append({"role": "assistant", "content": assistant})
                results = []
                for tc in resp.tool_calls:
                    text, is_error = await client.call(tc.name, tc.args)
                    results.append(
                        {"type": "tool_result", "tool_use_id": tc.id, "content": text, "is_error": is_error}
                    )
                messages.append({"role": "user", "content": results})
            else:
                out.error = "max turns exceeded"
        return out

    def teardown(self) -> None:
        pass
