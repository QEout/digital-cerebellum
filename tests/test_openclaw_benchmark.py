"""
Tests for OpenClaw Desktop Automation Benchmark.

Ensures the benchmark itself is runnable and produces structurally
correct results without requiring real LLM or OpenClaw.
"""

import pytest

from benchmarks.openclaw_benchmark import (
    SCENARIOS,
    WARM_QUERIES,
    BenchmarkResults,
    CurvePoint,
    ReliabilityResult,
    Scenario,
    SpeedResult,
    Step,
    run_round3,
    run_round4,
    to_json,
)


class TestScenarioDefinitions:
    """Verify scenario structure is valid."""

    def test_five_scenarios(self):
        assert len(SCENARIOS) == 5

    def test_five_warm_queries(self):
        assert len(WARM_QUERIES) == 5

    def test_scenarios_have_required_fields(self):
        for s in SCENARIOS:
            assert s.name
            assert s.task_query
            assert s.resolution
            assert len(s.steps) >= 6
            assert 0 <= s.failure_step < len(s.steps)
            assert s.cascade_consequence

    def test_steps_have_both_outcomes(self):
        for s in SCENARIOS:
            for step in s.steps:
                assert step.action
                assert step.state
                assert step.outcome_ok
                assert step.outcome_fail

    def test_failure_step_differs_from_ok(self):
        for s in SCENARIOS:
            fail_step = s.steps[s.failure_step]
            assert fail_step.outcome_ok != fail_step.outcome_fail, (
                f"Scenario '{s.name}' failure step should have different ok/fail outcomes"
            )


class TestReliabilityRound:
    """Test Round 3 (failure injection)."""

    def test_all_cascades_detected(self):
        results = run_round3()
        for rr in results:
            assert rr.detected_at is not None, (
                f"Cascade not detected in '{rr.name}'"
            )

    def test_rollback_plans_correct(self):
        results = run_round3()
        for rr in results:
            assert rr.rollback_correct, (
                f"Rollback plan incorrect in '{rr.name}': "
                f"rollback_to={rr.rollback_to}, failure_step={rr.failure_step}"
            )

    def test_steps_saved_positive(self):
        results = run_round3()
        total_saved = sum(rr.steps_saved for rr in results)
        assert total_saved > 0, "Should save at least some wasted steps"

    def test_result_structure(self):
        results = run_round3()
        assert len(results) == 5
        for rr in results:
            assert isinstance(rr, ReliabilityResult)
            assert rr.total_steps > 0
            assert rr.steps_wasted_no_cb >= 0
            assert rr.steps_wasted_cb >= 0
            assert rr.steps_saved >= 0


class TestLearningCurve:
    """Test Round 4 (convergence)."""

    def test_curve_has_10_points(self):
        curve = run_round4()
        assert len(curve) == 10

    def test_starts_at_zero(self):
        curve = run_round4()
        assert curve[0].hit_rate == 0.0, "First round should have 0% hit rate (cold start)"

    def test_converges(self):
        curve = run_round4()
        assert curve[-1].hit_rate > 0.5, (
            f"Should converge to >50% hit rate, got {curve[-1].hit_rate:.0%}"
        )

    def test_latency_decreases(self):
        curve = run_round4()
        assert curve[-1].avg_latency_ms < curve[0].avg_latency_ms, (
            "Latency should decrease as skills are learned"
        )


class TestJsonOutput:
    """Test JSON serialization."""

    def test_json_structure(self):
        results = BenchmarkResults(
            speed=[SpeedResult("test", 100.0, 10.0, 10.0, True)],
            reliability=[ReliabilityResult("test", 8, 3, 4, 5, 1, 4, 2, True)],
            curve=[CurvePoint(1, 0.0, 100.0), CurvePoint(2, 1.0, 10.0)],
        )
        j = to_json(results)
        assert "speed" in j
        assert "reliability" in j
        assert "learning_curve" in j
        assert "aggregate" in j
        assert j["speed"][0]["skill_hit"] is True
        assert j["reliability"][0]["rollback_correct"] is True
