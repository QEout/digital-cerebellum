"""
LangChain integration — CerebellumCallback.

Drop-in callback handler that wraps any LangChain agent with the
Digital Cerebellum's speed (SkillStore) and reliability (StepMonitor)
capabilities.

Usage::

    from digital_cerebellum.integrations.langchain import CerebellumCallback

    cb = CerebellumCallback()
    agent = initialize_agent(..., callbacks=[cb])
    agent.run("Deploy the production build")

    # After the run:
    print(cb.stats)

The callback monitors every tool call and LLM invocation.  When an
error cascade is detected, it raises CerebellumPause so the outer
loop can handle recovery.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import UUID

from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
from digital_cerebellum.monitor import StepMonitor, RollbackPlan

log = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain is required for this integration. "
            "Install it with: pip install langchain-core"
        )


class CerebellumPause(Exception):
    """Raised when StepMonitor detects an error cascade during agent execution."""

    def __init__(self, rollback_plan: RollbackPlan, verdict: Any):
        self.rollback_plan = rollback_plan
        self.verdict = verdict
        super().__init__(
            f"Cerebellum cascade detected at step {verdict.step_number}. "
            f"Rollback to step {rollback_plan.rollback_to_step}. "
            f"{rollback_plan.recommendation}"
        )


@dataclass
class _ToolState:
    """Tracks a single in-flight tool call."""
    tool_name: str
    tool_input: str
    start_time: float


class CerebellumCallback(BaseCallbackHandler):
    """
    LangChain callback handler that adds cerebellum monitoring.

    Monitors every tool call with StepMonitor, evaluates safety
    with DigitalCerebellum, and raises CerebellumPause on cascade.
    """

    def __init__(
        self,
        cerebellum: DigitalCerebellum | None = None,
        monitor: StepMonitor | None = None,
        pause_on_cascade: bool = True,
        auto_rollback: bool = True,
    ):
        super().__init__()

        if cerebellum is None:
            try:
                cfg = CerebellumConfig.from_yaml()
            except Exception:
                cfg = CerebellumConfig()
            cerebellum = DigitalCerebellum(cfg)

        self._cerebellum = cerebellum

        if monitor is None:
            monitor = StepMonitor(
                cerebellum=cerebellum,
                auto_rollback=auto_rollback,
            )

        self._monitor = monitor
        self._pause_on_cascade = pause_on_cascade

        self._in_flight: dict[UUID, _ToolState] = {}
        self._stats = {
            "tools_monitored": 0,
            "tools_blocked": 0,
            "cascades_detected": 0,
            "total_tool_time_ms": 0.0,
        }

    # ── Tool call lifecycle ──

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")

        pred = self._monitor.before_step(
            action=f"tool_call: {tool_name}({input_str[:200]})",
            state=f"agent step, tags={tags or []}",
        )

        self._in_flight[run_id] = _ToolState(
            tool_name=tool_name,
            tool_input=input_str[:500],
            start_time=time.perf_counter(),
        )
        self._stats["tools_monitored"] += 1

        if not pred.should_proceed:
            self._stats["tools_blocked"] += 1
            log.warning(
                "Cerebellum blocked tool %s (risk=%.2f)",
                tool_name, pred.risk_score,
            )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._in_flight.pop(run_id, None)
        if state is None:
            return

        elapsed = (time.perf_counter() - state.start_time) * 1000
        self._stats["total_tool_time_ms"] += elapsed

        verdict = self._monitor.after_step(
            outcome=f"tool_result: {output[:300]}",
            success=True,
        )

        if verdict.should_pause:
            self._stats["cascades_detected"] += 1
            plan = self._monitor.get_rollback_plan()
            if plan and self._pause_on_cascade:
                raise CerebellumPause(rollback_plan=plan, verdict=verdict)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._in_flight.pop(run_id, None)
        if state is None:
            return

        elapsed = (time.perf_counter() - state.start_time) * 1000
        self._stats["total_tool_time_ms"] += elapsed

        verdict = self._monitor.after_step(
            outcome=f"tool_error: {type(error).__name__}: {str(error)[:200]}",
            success=False,
        )

        if verdict.should_pause:
            self._stats["cascades_detected"] += 1
            plan = self._monitor.get_rollback_plan()
            if plan and self._pause_on_cascade:
                raise CerebellumPause(rollback_plan=plan, verdict=verdict)

    # ── Accessors ──

    @property
    def monitor(self) -> StepMonitor:
        return self._monitor

    @property
    def cerebellum(self) -> DigitalCerebellum:
        return self._cerebellum

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "monitor": self._monitor.stats,
        }

    def reset(self) -> dict[str, Any]:
        """Reset for a new agent run. Returns episode summary."""
        summary = self._monitor.reset()
        self._in_flight.clear()
        return summary
