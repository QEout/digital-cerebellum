"""
Digital Brain — complete cognitive architecture.

The Digital Brain unifies two modes of operation:

1. **Text mode** (think) — LLM cortex + cerebellum skill store
   Request-response: user text → skill match or LLM reasoning → response

2. **Control mode** (control) — micro-operation engine
   Continuous loop: observe → predict → act → learn at 285Hz+

Both share the same cerebellar principles (RFF pattern separation,
forward models, error-driven learning) but operate at different
timescales and on different input modalities.

Architecture::

    User Input (text)          Environment (numeric, 60Hz+)
        │                              │
        ▼                              ▼
    ┌─ Text Path ──────┐      ┌─ Control Path ──────────┐
    │                    │      │                          │
    │  Skill Store       │      │  StateEncoder            │
    │  → match?          │      │  PatternSeparator (RFF)  │
    │  YES → replay      │      │  ForwardModel            │
    │  NO  → LLM → learn │      │  ActionNet → act         │
    │                    │      │  SPE → online learn       │
    └────────────────────┘      └──────────────────────────┘

Usage::

    from digital_cerebellum.brain import DigitalBrain

    # Text mode
    brain = DigitalBrain.from_yaml()
    r = brain.think("What's the weather in Tokyo?")

    # Control mode
    from digital_cerebellum.micro_ops.environments import TargetTracker
    env = TargetTracker()
    summary = brain.control(env, n_steps=500)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.micro_ops.engine import MicroOpEngine, MicroOpConfig, Environment

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
    skill_id: str | None = None

    @property
    def used_fast_path(self) -> bool:
        return self.path == "cerebellum"


# ======================================================================
# DigitalBrain
# ======================================================================

class DigitalBrain:
    """
    Complete cognitive architecture: LLM cortex + digital cerebellum.

    The cerebellum learns from every LLM interaction and progressively
    handles more patterns on the fast path — not just evaluating safety,
    but actually EXECUTING learned skills.
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
            "skill_hits": 0,
            "skill_misses": 0,
            "tools_executed": 0,
            "tools_blocked": 0,
            "tools_replayed": 0,
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

        Stage 1: Check skill store — can the cerebellum handle this alone?
        Stage 2: If not → call LLM (cortex)
        Stage 3: If LLM wants tool calls → pre-evaluate → execute
        Stage 4: Learn the interaction as a new skill
        """
        t0 = time.perf_counter()
        self._stats["total_queries"] += 1
        self._conversation.append({"role": "user", "content": user_input})

        # ── Stage 1: Skill matching (cerebellum fast path) ──
        skill_match = self.cerebellum.match_skill(user_input)

        if skill_match is not None and skill_match.should_execute:
            result = self._execute_skill(user_input, skill_match, t0)
            if result is not None:
                return result

        # ── Stage 2-3: Cortex slow path ──
        self._stats["skill_misses"] += 1
        return self._cortex_path(user_input, max_tool_rounds, t0)

    def _execute_skill(
        self,
        user_input: str,
        skill_match: Any,
        t0: float,
    ) -> ThinkResult | None:
        """
        Execute a matched skill from procedural memory.

        If the skill has tool calls, replay them.
        If it's a direct response, return it immediately.
        """
        skill = skill_match.skill
        all_tool_calls = []
        all_tool_results = []

        if skill.tool_calls:
            for tc_record in skill.tool_calls:
                tool_name = tc_record.get("tool", tc_record.get("name", ""))
                tool_params = tc_record.get("params", tc_record.get("arguments", {}))

                if tool_name not in self._tools:
                    log.warning("Skill references unregistered tool '%s', falling back to cortex", tool_name)
                    return None

                eval_result = self.cerebellum.evaluate("tool_call", {
                    "tool_name": tool_name,
                    "tool_params": tool_params,
                }, context=user_input)

                if not eval_result.get("safe", True):
                    log.warning("Skill tool call '%s' blocked by safety, falling back to cortex", tool_name)
                    return None

                try:
                    result = self._tools[tool_name].fn(**tool_params)
                    all_tool_calls.append({
                        "tool": tool_name, "params": tool_params,
                        "result": result, "status": "replayed",
                    })
                    all_tool_results.append({"tool": tool_name, "result": result})
                    self._stats["tools_replayed"] += 1
                except Exception as e:
                    log.warning("Skill tool replay failed: %s, falling back to cortex", e)
                    return None

        text = skill.response_text
        self._conversation.append({"role": "assistant", "content": text})

        self._stats["fast_path"] += 1
        self._stats["skill_hits"] += 1
        skill.access_count += 1

        log.info(
            "SKILL HIT: similarity=%.3f confidence=%.3f skill_id=%s tools=%d",
            skill_match.similarity, skill_match.match_confidence,
            skill.id[:8], len(skill.tool_calls),
        )

        return ThinkResult(
            text=text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            path="cerebellum",
            latency_ms=(time.perf_counter() - t0) * 1000,
            confidence=skill_match.match_confidence,
            llm_called=False,
            skill_id=skill.id,
        )

    def _cortex_path(
        self,
        user_input: str,
        max_tool_rounds: int,
        t0: float,
    ) -> ThinkResult:
        """Full LLM reasoning path. Learns a skill from the interaction."""
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

                skill_id = self.cerebellum.learn_skill(
                    input_text=user_input,
                    response_text=text,
                    tool_calls=all_tool_calls if all_tool_calls else None,
                    domain="response",
                )

                return ThinkResult(
                    text=text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    path="cortex",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    llm_called=True,
                    skill_id=skill_id,
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

        skill_id = self.cerebellum.learn_skill(
            input_text=user_input,
            response_text=text,
            tool_calls=[
                {"tool": tc["tool"], "params": tc["params"]}
                for tc in all_tool_calls if tc.get("status") == "executed"
            ],
            domain="response",
        )

        return ThinkResult(
            text=text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            path="cortex",
            latency_ms=(time.perf_counter() - t0) * 1000,
            llm_called=True,
            skill_id=skill_id,
        )

    # ==================================================================
    # Skill feedback
    # ==================================================================

    def skill_feedback(self, result: ThinkResult, success: bool) -> None:
        """
        Provide feedback on a brain.think() result.

        Reinforces successful skills, weakens failed ones.
        This drives the cerebellum's learning: skills that work
        become more confident and more likely to be used on the fast path.
        """
        if result.skill_id is None:
            return
        if success:
            self.cerebellum.skill_store.reinforce(result.skill_id)
        else:
            self.cerebellum.skill_store.weaken(result.skill_id)

    # ==================================================================
    # Active exploration
    # ==================================================================

    def get_exploration_suggestions(self) -> list[dict[str, Any]]:
        """
        Ask the cerebellum what domains are worth exploring.

        Returns domains ranked by learning potential — areas where
        practice would yield the most improvement.
        """
        ranking = self.cerebellum.curiosity_drive.get_exploration_ranking()
        suggestions = []
        for domain, score in ranking:
            if score > 0:
                suggestions.append({
                    "domain": domain,
                    "curiosity_score": round(score, 4),
                    "recommendation": "explore" if score > 0.1 else "practice",
                })
        return suggestions

    # ==================================================================
    # Conversation management
    # ==================================================================

    def reset_conversation(self):
        """Clear conversation history."""
        self._conversation.clear()

    def introspect(self, domain: str | None = None):
        """Metacognitive self-report — what the brain knows about itself."""
        return self.cerebellum.introspect(domain)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._stats["total_queries"] or 1
        engines = getattr(self, "_engines", {})
        return {
            **self._stats,
            "skill_hit_rate": round(self._stats["skill_hits"] / total, 3),
            "automation_ratio": round(self._stats["fast_path"] / total, 3),
            "cerebellum": self.cerebellum.stats,
            "tools_registered": list(self._tools.keys()),
            "conversation_length": len(self._conversation),
            "micro_op_engines": {
                f"{k[0]}d_state_{k[1]}d_action": v.stats
                for k, v in engines.items()
            },
        }

    # ==================================================================
    # Control mode (Phase 6 — continuous micro-operations)
    # ==================================================================

    def control(
        self,
        env: Environment,
        n_steps: int | None = None,
        target_hz: float = 60.0,
        cfg: MicroOpConfig | None = None,
    ) -> dict[str, Any]:
        """
        Run continuous control on an environment.

        This is the "body" mode — the cerebellum directly controls an
        environment at 60Hz+, learning from prediction errors, without
        any LLM involvement.

        Parameters
        ----------
        env : Environment with observe() and execute() methods
        n_steps : number of control steps (default: 10000)
        target_hz : target loop frequency (default: 60)
        cfg : optional MicroOpConfig for engine tuning

        Returns
        -------
        Summary dict with performance metrics and learning progress.
        """
        engine = self._get_or_create_engine(env, cfg)
        return engine.run(env, n_steps=n_steps, target_hz=target_hz)

    def control_step(self, env: Environment, cfg: MicroOpConfig | None = None):
        """Execute a single control step. Returns StepResult."""
        engine = self._get_or_create_engine(env, cfg)
        return engine.step(env)

    def _get_or_create_engine(
        self,
        env: Environment,
        cfg: MicroOpConfig | None = None,
    ) -> MicroOpEngine:
        """Get or create a MicroOpEngine matched to the environment dimensions."""
        key = (env.state_dim, env.action_dim)
        if not hasattr(self, "_engines"):
            self._engines: dict[tuple[int, int], MicroOpEngine] = {}

        if key not in self._engines:
            self._engines[key] = MicroOpEngine(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                cfg=cfg or MicroOpConfig(),
            )
        return self._engines[key]

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
