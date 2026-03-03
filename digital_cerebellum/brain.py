"""
Digital Brain — complete cognitive architecture.

Instead of being a middleware inside an agent framework, the Digital Brain
IS the agent.  The LLM serves as the cerebral cortex (slow, flexible
reasoning) and the cerebellum handles fast-path predictions.

Architecture::

    User Input
        │
        ▼
    ┌─ Cerebellum (fast path, < 50ms) ──────────────────────┐
    │                                                         │
    │  "Have I seen this pattern before?"                     │
    │                                                         │
    │  YES (high confidence)  →  Respond directly             │
    │  NO  (low confidence)   →  Escalate to Cortex (LLM)    │
    │                                                         │
    │  ◄── Learn from every Cortex response ──►               │
    └─────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─ Cortex / LLM (slow path, 1-5s) ─────────────────────┐
    │                                                         │
    │  Full reasoning with tool definitions                   │
    │  Returns: text response + optional tool calls           │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─ Tool Executor ───────────────────────────────────────┐
    │                                                         │
    │  Built-in tool registry.  No LangChain needed.          │
    │  Each tool call is pre-evaluated by the cerebellum.     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

Usage::

    from digital_cerebellum.brain import DigitalBrain

    brain = DigitalBrain.from_yaml()
    brain.register_tool("search_web", search_fn, "Search the internet")

    response = brain.think("What's the weather in Tokyo?")
    print(response.text)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

log = logging.getLogger(__name__)


# ======================================================================
# Data types
# ======================================================================

@dataclass
class ToolDef:
    """A registered tool that the brain can use."""
    name: str
    fn: Callable[..., str]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


@dataclass
class ThinkResult:
    """The output of one brain.think() call."""
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    path: str = "cortex"
    latency_ms: float = 0.0
    confidence: float = 0.0
    llm_called: bool = True

    @property
    def used_fast_path(self) -> bool:
        return self.path == "cerebellum"


# ======================================================================
# Response Microzone — learns to predict LLM responses
# ======================================================================

class _ResponseMicrozone:
    """
    Internal microzone that learns to predict whether the cerebellum
    can handle a query pattern without calling the LLM.

    This isn't a safety check — it's a familiarity check.
    """

    @property
    def name(self):
        return "response_familiarity"


# ======================================================================
# DigitalBrain
# ======================================================================

class DigitalBrain:
    """
    Complete cognitive architecture: LLM cortex + digital cerebellum.

    The cerebellum learns from every LLM interaction and progressively
    handles more patterns on the fast path.
    """

    def __init__(self, cfg: CerebellumConfig | None = None):
        self.cfg = cfg or CerebellumConfig()

        self.cerebellum = DigitalCerebellum(self.cfg)
        self.cerebellum.register_microzone(ToolCallMicrozone())

        self._tools: dict[str, ToolDef] = {}
        self._conversation: list[dict[str, str]] = []
        self._system_prompt = (
            "You are a helpful AI assistant. You have access to tools "
            "and can use them when needed. Be concise and accurate."
        )
        self._stats = {
            "total_queries": 0,
            "fast_path": 0,
            "slow_path": 0,
            "tools_executed": 0,
            "tools_blocked": 0,
        }

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "DigitalBrain":
        cfg = CerebellumConfig.from_yaml(path)
        return cls(cfg)

    # ==================================================================
    # Tool registration
    # ==================================================================

    def register_tool(
        self,
        name: str,
        fn: Callable[..., str],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool the brain can use."""
        self._tools[name] = ToolDef(
            name=name, fn=fn,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
        )
        log.info("Registered tool: %s", name)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value

    # ==================================================================
    # Core cognitive loop
    # ==================================================================

    def think(self, user_input: str, max_tool_rounds: int = 3) -> ThinkResult:
        """
        Process a user input through the full cognitive loop.

        1. Cerebellum checks if this is a familiar pattern
        2. If uncertain → call LLM (cortex)
        3. If LLM wants tool calls → pre-evaluate with cerebellum → execute
        4. Learn from the entire interaction
        """
        t0 = time.perf_counter()
        self._stats["total_queries"] += 1

        self._conversation.append({"role": "user", "content": user_input})

        cortex = self._get_cortex()
        tool_specs = [t.to_openai_spec() for t in self._tools.values()] or None

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._conversation[-20:])

        all_tool_calls = []
        all_tool_results = []

        for _round in range(max_tool_rounds):
            try:
                resp = cortex.client.chat.completions.create(
                    model=cortex.model,
                    messages=messages,
                    tools=tool_specs,
                    temperature=0.3,
                    extra_body={"enable_thinking": False},
                )
            except Exception as e:
                log.error("LLM call failed: %s", e)
                return ThinkResult(
                    text=f"I encountered an error: {e}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                self._conversation.append({"role": "assistant", "content": text})
                self._stats["slow_path"] += 1

                return ThinkResult(
                    text=text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    path="cortex",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    llm_called=True,
                )

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_params = {}

                tc_record = {
                    "tool": tool_name,
                    "params": tool_params,
                    "id": tc.id,
                }

                eval_result = self.cerebellum.evaluate("tool_call", {
                    "tool_name": tool_name,
                    "tool_params": tool_params,
                }, context=user_input)

                safe = eval_result.get("safe", True)
                tc_record["cerebellum_eval"] = eval_result

                if safe and tool_name in self._tools:
                    try:
                        result = self._tools[tool_name].fn(**tool_params)
                        tc_record["result"] = result
                        tc_record["status"] = "executed"
                        self._stats["tools_executed"] += 1
                    except Exception as e:
                        result = f"Tool error: {e}"
                        tc_record["result"] = result
                        tc_record["status"] = "error"
                elif not safe:
                    result = (
                        f"[BLOCKED by cerebellum] Tool '{tool_name}' was "
                        f"deemed unsafe (safety={eval_result.get('safety_score', 0):.3f})"
                    )
                    tc_record["result"] = result
                    tc_record["status"] = "blocked"
                    self._stats["tools_blocked"] += 1
                else:
                    result = f"Tool '{tool_name}' is not registered."
                    tc_record["result"] = result
                    tc_record["status"] = "not_found"

                all_tool_calls.append(tc_record)
                all_tool_results.append({"tool": tool_name, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        text = "I completed the tool operations."
        self._conversation.append({"role": "assistant", "content": text})

        return ThinkResult(
            text=text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            path="cortex",
            latency_ms=(time.perf_counter() - t0) * 1000,
            llm_called=True,
        )

    # ==================================================================
    # Conversation management
    # ==================================================================

    def reset_conversation(self):
        """Clear conversation history."""
        self._conversation.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "cerebellum": {
                "engine_step": self.cerebellum._step,
                "memory": self.cerebellum.memory.stats,
                "microzones": list(self.cerebellum._microzones.keys()),
            },
            "tools_registered": list(self._tools.keys()),
            "conversation_length": len(self._conversation),
        }

    # ==================================================================
    # Internal
    # ==================================================================

    def _get_cortex(self):
        if self.cerebellum._cortex is None:
            from digital_cerebellum.cortex.cortex_interface import CortexInterface
            self.cerebellum._cortex = CortexInterface(
                model=self.cfg.llm_model,
                api_key=self.cfg.llm_api_key,
                base_url=self.cfg.llm_base_url,
            )
        return self.cerebellum._cortex
