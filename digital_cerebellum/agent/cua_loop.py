"""
CerebellumAgent — GPT-5.4 Computer Use + Digital Cerebellum.

The CUA (Computer Using Agent) loop:
    screenshot → GPT-5.4 reasons → returns actions → execute → screenshot → repeat

Digital Cerebellum wraps this loop with:
    1. SkillStore intercept: known task → skip GPT-5.4, replay stored actions (0 tokens)
    2. StepMonitor: predict/detect errors before they cascade
    3. HabitObserver: learn temporal patterns across sessions
    4. FluidMemory: persistent cross-session context

Biological mapping:
    GPT-5.4 = cortex (perception, reasoning, planning)
    Digital Cerebellum = cerebellum (learning, acceleration, error prediction, timing)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ── Computer interface ──────────────────────────────────────────────

@runtime_checkable
class Computer(Protocol):
    """Anything that can execute UI actions and take screenshots."""

    async def screenshot(self) -> bytes:
        """Return PNG screenshot bytes."""
        ...

    async def click(self, x: int, y: int, button: str = "left") -> None: ...
    async def double_click(self, x: int, y: int) -> None: ...
    async def type(self, text: str) -> None: ...
    async def keypress(self, keys: list[str]) -> None: ...
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None: ...
    async def move(self, x: int, y: int) -> None: ...
    async def drag(self, path: list[dict[str, int]]) -> None: ...
    async def wait(self) -> None: ...


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class ActionRecord:
    """One CUA action returned by GPT-5.4."""
    action_type: str
    args: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result of a complete CerebellumAgent.run() call."""
    task: str
    success: bool
    actions_executed: int
    tokens_used: int
    skill_hit: bool
    time_seconds: float
    final_text: str
    errors: list[str] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)


@dataclass
class AgentConfig:
    model: str = "gpt-5.4"
    reasoning_effort: str = "low"
    skill_confidence_threshold: float = 0.8
    max_turns: int = 50
    screenshot_detail: str = "auto"
    environment: str = "browser"
    display_width: int = 1440
    display_height: int = 900
    confirm_risky_actions: bool = True


# ── Core agent ──────────────────────────────────────────────────────

class CerebellumAgent:
    """
    GPT-5.4 Computer Use Agent enhanced with Digital Cerebellum.

    Usage::

        from openai import OpenAI
        from digital_cerebellum.agent import CerebellumAgent

        client = OpenAI()
        agent = CerebellumAgent(client)

        result = await agent.run("Book a flight to Tokyo", computer)
    """

    def __init__(
        self,
        openai_client: Any,
        config: AgentConfig | None = None,
    ):
        self._client = openai_client
        self.config = config or AgentConfig()

        from digital_cerebellum.monitor.step_monitor import StepMonitor
        from digital_cerebellum.memory.habit_observer import HabitObserver
        from digital_cerebellum.rhythm.engine import RhythmEngine

        self.monitor = StepMonitor()
        self.habits = HabitObserver()
        self.rhythm = RhythmEngine(self.habits)

        self._session_actions: list[ActionRecord] = []
        self._total_tokens = 0

    # ── Public API ──────────────────────────────────────────────

    async def run(self, task: str, computer: Computer) -> TaskResult:
        """
        Execute a desktop task. Returns TaskResult with full trace.

        Fast path: if SkillStore has a high-confidence match, replay
        stored actions without calling GPT-5.4 (0 tokens, <10ms).

        Slow path: GPT-5.4 CUA loop with cerebellum monitoring.
        """
        t0 = time.time()
        self._session_actions = []
        self._total_tokens = 0
        errors: list[str] = []

        # ── Fast path: SkillStore intercept ─────────────────────
        skill_match = self._try_skill_match(task)
        if skill_match is not None:
            log.info("Skill hit for '%s' (confidence=%.2f) — replaying",
                     task[:60], skill_match["confidence"])
            replay_ok = await self._replay_skill(skill_match, computer)
            self.habits.record(task, domain="desktop", success=replay_ok)
            return TaskResult(
                task=task,
                success=replay_ok,
                actions_executed=len(skill_match["tool_calls"]),
                tokens_used=0,
                skill_hit=True,
                time_seconds=time.time() - t0,
                final_text=f"[Skill replay] {skill_match['response_text'][:200]}",
                actions=self._session_actions,
            )

        # ── Slow path: GPT-5.4 CUA loop ────────────────────────
        log.info("No skill match — entering GPT-5.4 CUA loop for: %s", task[:80])

        screenshot_b64 = await self._take_screenshot(computer)

        response = self._create_response(
            task=task,
            screenshot_b64=screenshot_b64,
        )
        self._total_tokens += _count_tokens(response)

        turn = 0
        final_text = ""

        while turn < self.config.max_turns:
            turn += 1

            computer_call = _extract_computer_call(response)
            if computer_call is None:
                final_text = _extract_text(response)
                break

            actions = computer_call.get("actions", [])
            action_desc = _describe_actions(actions)

            # ── Cerebellum: before_step ─────────────────────────
            pred = self.monitor.before_step(
                action=action_desc,
                state=f"turn={turn}, task={task[:60]}",
            )

            if not pred.should_proceed:
                log.warning("Cerebellum BLOCKED action: %s", pred.failure_warning)
                errors.append(f"Blocked at turn {turn}: {pred.failure_warning}")
                break

            # ── Execute actions ──────────────────────────────────
            for action in actions:
                await self._execute_action(computer, action)

            # ── Take screenshot ──────────────────────────────────
            screenshot_b64 = await self._take_screenshot(computer)

            # ── Cerebellum: after_step ───────────────────────────
            verdict = self.monitor.after_step(
                outcome=f"Executed {len(actions)} actions at turn {turn}",
                success=True,
            )

            if verdict.should_pause:
                log.warning("Cerebellum detected CASCADE at turn %d", turn)
                errors.append(f"Cascade at turn {turn}")
                plan = self.monitor.get_rollback_plan()
                if plan:
                    errors.append(f"Rollback: {plan.recommendation}")
                break

            # ── Send screenshot back ─────────────────────────────
            response = self._send_screenshot(
                response_id=response.id,
                call_id=computer_call["call_id"],
                screenshot_b64=screenshot_b64,
            )
            self._total_tokens += _count_tokens(response)

        # ── Learn from this interaction ──────────────────────────
        success = len(errors) == 0
        self._learn_task(task, success)
        self.habits.record(task, domain="desktop", success=success)

        return TaskResult(
            task=task,
            success=success,
            actions_executed=len(self._session_actions),
            tokens_used=self._total_tokens,
            skill_hit=False,
            time_seconds=time.time() - t0,
            final_text=final_text,
            errors=errors,
            actions=self._session_actions,
        )

    def save(self, path: str = ".digital-cerebellum") -> None:
        """Persist all learned state (skills, memory, habits)."""
        self.monitor.save(path)
        self.habits.save(path + "/habits")
        log.info("Agent state saved to %s", path)

    def load(self, path: str = ".digital-cerebellum") -> None:
        """Load previously learned state."""
        self.monitor.load(path)
        self.habits.load(path + "/habits")
        log.info("Agent state loaded from %s", path)

    def suggest(self) -> list[dict[str, Any]]:
        """Proactive suggestions based on learned habits."""
        return self.rhythm.get_proactive_suggestions()

    # ── Skill matching / replay ─────────────────────────────────

    def _try_skill_match(self, task: str) -> dict[str, Any] | None:
        """Check SkillStore for a known pattern."""
        store = self.monitor._forward_model  # Access via monitor's encoder
        try:
            encoder = self.monitor._get_encoder()
            embedding = encoder.encode_text(task)
            embedding = self.monitor._fit_dim(embedding)

            from digital_cerebellum.memory.skill_store import SkillStore
            skill_store: SkillStore = getattr(self.monitor, "_skill_store", None)

            if skill_store is None:
                if hasattr(self.monitor, "skill_store"):
                    skill_store = self.monitor.skill_store
                else:
                    return None

            match = skill_store.match(embedding)
            if match is None:
                return None

            if (match.match_confidence >= self.config.skill_confidence_threshold
                    and match.skill.is_sequence):
                return {
                    "skill_id": match.skill.id,
                    "confidence": match.match_confidence,
                    "response_text": match.skill.response_text,
                    "tool_calls": match.skill.tool_calls,
                }
        except Exception as e:
            log.debug("Skill match failed: %s", e)
        return None

    async def _replay_skill(
        self, skill_data: dict[str, Any], computer: Computer
    ) -> bool:
        """Replay a stored action sequence directly on the computer."""
        try:
            for tc in skill_data["tool_calls"]:
                action_type = tc.get("type") or tc.get("action_type", "")
                args = tc.get("args", tc.get("arguments", {}))
                if isinstance(args, str):
                    args = json.loads(args)

                await self._execute_action(computer, {
                    "type": action_type,
                    **args,
                })
            return True
        except Exception as e:
            log.error("Skill replay failed: %s", e)
            return False

    def _learn_task(self, task: str, success: bool) -> None:
        """Store the completed task as a skill for future fast-path."""
        if not success or len(self._session_actions) < 2:
            return

        try:
            encoder = self.monitor._get_encoder()
            embedding = encoder.encode_text(task)
            embedding = self.monitor._fit_dim(embedding)

            tool_calls = [
                {"type": a.action_type, "args": a.args}
                for a in self._session_actions
            ]

            from digital_cerebellum.memory.skill_store import SkillStore
            skill_store: SkillStore | None = getattr(
                self.monitor, "skill_store", None
            )
            if skill_store is None:
                return

            skill_store.learn_from_interaction(
                input_embedding=embedding,
                input_text=task,
                response_text=f"Executed {len(tool_calls)} desktop actions",
                tool_calls=tool_calls,
                domain="desktop",
            )
            log.info("Learned skill: '%s' (%d actions)", task[:60], len(tool_calls))
        except Exception as e:
            log.debug("Failed to learn task: %s", e)

    # ── GPT-5.4 API calls ──────────────────────────────────────

    def _create_response(self, task: str, screenshot_b64: str) -> Any:
        """First turn: send task + initial screenshot to GPT-5.4."""
        return self._client.responses.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            tools=[{
                "type": "computer",
                "environment": self.config.environment,
                "display_width": self.config.display_width,
                "display_height": self.config.display_height,
            }],
            input=[
                {"role": "user", "content": task},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": self.config.screenshot_detail,
                    },
                },
            ],
            truncation="auto",
        )

    def _send_screenshot(
        self, response_id: str, call_id: str, screenshot_b64: str
    ) -> Any:
        """Continue the CUA loop: send screenshot back after action execution."""
        return self._client.responses.create(
            model=self.config.model,
            previous_response_id=response_id,
            tools=[{
                "type": "computer",
                "environment": self.config.environment,
                "display_width": self.config.display_width,
                "display_height": self.config.display_height,
            }],
            input=[{
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }],
            truncation="auto",
        )

    # ── Action execution ────────────────────────────────────────

    async def _execute_action(
        self, computer: Computer, action: dict[str, Any]
    ) -> None:
        """Execute a single CUA action on the computer."""
        action_type = action.get("type", "")
        record = ActionRecord(action_type=action_type, args={})

        if action_type == "screenshot":
            pass  # handled separately
        elif action_type == "click":
            x, y = action["x"], action["y"]
            button = action.get("button", "left")
            record.args = {"x": x, "y": y, "button": button}
            await computer.click(x, y, button)
        elif action_type == "double_click":
            x, y = action["x"], action["y"]
            record.args = {"x": x, "y": y}
            await computer.double_click(x, y)
        elif action_type == "type":
            text = action["text"]
            record.args = {"text": text}
            await computer.type(text)
        elif action_type == "keypress":
            keys = action["keys"]
            record.args = {"keys": keys}
            await computer.keypress(keys)
        elif action_type == "scroll":
            record.args = {
                "x": action["x"], "y": action["y"],
                "scroll_x": action["scroll_x"], "scroll_y": action["scroll_y"],
            }
            await computer.scroll(
                action["x"], action["y"],
                action["scroll_x"], action["scroll_y"],
            )
        elif action_type == "move":
            record.args = {"x": action["x"], "y": action["y"]}
            await computer.move(action["x"], action["y"])
        elif action_type == "drag":
            path = action["path"]
            record.args = {"path": path}
            await computer.drag(path)
        elif action_type == "wait":
            await computer.wait()
        else:
            log.warning("Unknown action type: %s", action_type)
            return

        self._session_actions.append(record)

    async def _take_screenshot(self, computer: Computer) -> str:
        """Capture screenshot and return as base64 string."""
        png_bytes = await computer.screenshot()
        return base64.b64encode(png_bytes).decode("ascii")


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_computer_call(response: Any) -> dict[str, Any] | None:
    """Extract computer_call from a Responses API response."""
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "computer_call":
            return {
                "call_id": item.id,
                "actions": [
                    _action_to_dict(a) for a in getattr(item, "actions", [])
                ],
            }
    return None


def _action_to_dict(action: Any) -> dict[str, Any]:
    """Convert an API action object to a plain dict."""
    if isinstance(action, dict):
        return action
    d = {"type": getattr(action, "type", "unknown")}
    for attr in ("x", "y", "button", "text", "keys", "scroll_x",
                 "scroll_y", "path"):
        val = getattr(action, attr, None)
        if val is not None:
            d[attr] = val
    return d


def _extract_text(response: Any) -> str:
    """Extract final text output from a Responses API response."""
    parts = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text"):
                    parts.append(content.text)
    return "\n".join(parts)


def _describe_actions(actions: list[dict[str, Any]]) -> str:
    """Human-readable description of a batch of CUA actions."""
    parts = []
    for a in actions:
        t = a.get("type", "?")
        if t == "click":
            parts.append(f"click({a.get('x')},{a.get('y')})")
        elif t == "type":
            text = a.get("text", "")
            parts.append(f'type("{text[:30]}")')
        elif t == "keypress":
            parts.append(f"keypress({a.get('keys')})")
        elif t == "scroll":
            parts.append(f"scroll({a.get('scroll_x')},{a.get('scroll_y')})")
        else:
            parts.append(t)
    return " → ".join(parts) if parts else "screenshot"


def _count_tokens(response: Any) -> int:
    """Extract token usage from API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, "total_tokens", 0)
