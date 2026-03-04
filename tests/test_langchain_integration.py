"""
Tests for LangChain CerebellumCallback integration.

Tests the callback handler independently (no LangChain agent needed)
by calling the lifecycle methods directly.
"""

import pytest
from uuid import uuid4

from digital_cerebellum.integrations.langchain import (
    CerebellumCallback,
    CerebellumPause,
)


class TestCerebellumCallback:
    """Test the LangChain callback handler."""

    def test_init_creates_monitor(self):
        cb = CerebellumCallback()
        assert cb.monitor is not None
        assert cb.cerebellum is not None

    def test_tool_start_end_lifecycle(self):
        cb = CerebellumCallback()
        run_id = uuid4()

        cb.on_tool_start(
            serialized={"name": "search"},
            input_str="query about weather",
            run_id=run_id,
        )

        cb.on_tool_end(
            output="Sunny, 25°C",
            run_id=run_id,
        )

        assert cb.stats["tools_monitored"] == 1
        assert cb.stats["cascades_detected"] == 0

    def test_tool_error_records_failure(self):
        cb = CerebellumCallback()
        run_id = uuid4()

        cb.on_tool_start(
            serialized={"name": "database_query"},
            input_str="SELECT * FROM users",
            run_id=run_id,
        )

        cb.on_tool_error(
            error=ConnectionError("Database unreachable"),
            run_id=run_id,
        )

        assert cb.stats["tools_monitored"] == 1

    def test_cascade_raises_pause(self):
        cb = CerebellumCallback(pause_on_cascade=True)
        # Use low cascade limit for faster detection
        cb._monitor = type(cb._monitor)(
            cascade_consecutive_limit=2,
            cascade_risk_threshold=0.5,
        )

        run_id1 = uuid4()
        cb.on_tool_start(
            serialized={"name": "api_call"},
            input_str="POST /deploy",
            run_id=run_id1,
        )
        cb.on_tool_error(
            error=RuntimeError("deploy failed"),
            run_id=run_id1,
        )

        caught = False
        try:
            run_id2 = uuid4()
            cb.on_tool_start(
                serialized={"name": "api_call"},
                input_str="POST /rollback",
                run_id=run_id2,
            )
            cb.on_tool_error(
                error=RuntimeError("rollback also failed"),
                run_id=run_id2,
            )
        except CerebellumPause as e:
            caught = True
            assert e.rollback_plan is not None
            assert e.rollback_plan.steps_wasted >= 1

        assert caught, "CerebellumPause should be raised on cascade"

    def test_no_pause_when_disabled(self):
        cb = CerebellumCallback(pause_on_cascade=False)
        cb._monitor = type(cb._monitor)(
            cascade_consecutive_limit=2,
            cascade_risk_threshold=0.5,
        )

        for _ in range(3):
            run_id = uuid4()
            cb.on_tool_start(
                serialized={"name": "bad_tool"},
                input_str="fail",
                run_id=run_id,
            )
            cb.on_tool_error(
                error=RuntimeError("error"),
                run_id=run_id,
            )

        assert cb.stats["cascades_detected"] >= 1

    def test_reset_clears_state(self):
        cb = CerebellumCallback()
        run_id = uuid4()

        cb.on_tool_start(
            serialized={"name": "test"},
            input_str="hello",
            run_id=run_id,
        )
        cb.on_tool_end(output="world", run_id=run_id)

        summary = cb.reset()
        assert summary["steps"] == 1
        assert cb.monitor.get_rollback_plan() is None

    def test_cerebellum_pause_has_rollback_info(self):
        err = CerebellumPause.__new__(CerebellumPause)
        from digital_cerebellum.monitor.types import RollbackPlan
        plan = RollbackPlan(
            rollback_to_step=2,
            last_safe_state="everything ok",
            last_safe_outcome="step 2 passed",
            failed_steps=[{"step": 3, "action": "deploy"}],
            total_steps=3,
            steps_wasted=1,
            cascade_risk=0.8,
            recommendation="Roll back to step 2",
        )
        err.rollback_plan = plan
        assert err.rollback_plan.rollback_to_step == 2

    def test_stats_include_monitor(self):
        cb = CerebellumCallback()
        stats = cb.stats
        assert "monitor" in stats
        assert "tools_monitored" in stats
        assert "cascades_detected" in stats
