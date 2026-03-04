"""
Tests for Phase 7: StepMonitor universal agent monitoring protocol.

Tests cover:
  - StepForwardModel: predict, learn, SPE computation
  - ErrorCascadeDetector: cascade detection, trend analysis
  - FailureMemory: record, check, decay
  - StepMonitor: full before_step/after_step protocol
  - Integration: cascade detection over multi-step sequences
  - Vector input mode (for game/robot scenarios)
"""

import numpy as np
import pytest

from digital_cerebellum.monitor.step_forward_model import StepForwardModel
from digital_cerebellum.monitor.cascade_detector import ErrorCascadeDetector
from digital_cerebellum.monitor.failure_memory import FailureMemory
from digital_cerebellum.monitor.step_monitor import StepMonitor
from digital_cerebellum.monitor.types import (
    CascadeStatus,
    FailureWarning,
    RollbackPlan,
    StepPrediction,
    StepVerdict,
)


# ======================================================================
# StepForwardModel tests
# ======================================================================

class TestStepForwardModel:

    def test_predict_returns_correct_shape(self):
        fm = StepForwardModel(embedding_dim=64)
        state = np.random.randn(64).astype(np.float32)
        action = np.random.randn(64).astype(np.float32)
        pred, confidence = fm.predict(state, action)
        assert pred.shape == (64,)
        assert 0.0 < confidence <= 1.0

    def test_learn_reduces_error(self):
        fm = StepForwardModel(embedding_dim=32, lr=0.01)
        state = np.random.randn(32).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)
        outcome = state + 0.1 * np.random.randn(32).astype(np.float32)

        errors = []
        for _ in range(50):
            error = fm.learn(state, action, outcome)
            errors.append(error)

        assert errors[-1] < errors[0], "Forward model should improve with training"

    def test_compute_spe(self):
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.0, 3.2])
        spe = StepForwardModel.compute_spe(predicted, actual)
        assert spe > 0
        assert isinstance(spe, float)

    def test_compute_spe_identical(self):
        vec = np.array([1.0, 2.0, 3.0])
        spe = StepForwardModel.compute_spe(vec, vec)
        assert spe == pytest.approx(0.0, abs=1e-6)

    def test_stats(self):
        fm = StepForwardModel(embedding_dim=16)
        assert fm.stats["step"] == 0

        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        outcome = np.random.randn(16).astype(np.float32)
        fm.learn(state, action, outcome)

        assert fm.stats["step"] == 1
        assert "mean_recent_error" in fm.stats

    def test_is_improving(self):
        fm = StepForwardModel(embedding_dim=16, lr=0.01)
        assert fm.is_improving is True

        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        outcome = state + 0.05 * np.random.randn(16).astype(np.float32)

        for _ in range(30):
            fm.learn(state, action, outcome)

        assert fm.is_improving is True


# ======================================================================
# ErrorCascadeDetector tests
# ======================================================================

class TestErrorCascadeDetector:

    def test_low_spe_no_cascade(self):
        det = ErrorCascadeDetector(spe_threshold=0.5)
        for _ in range(5):
            status = det.observe(0.1)
        assert not status.is_cascading
        assert status.consecutive_high == 0

    def test_high_spe_triggers_cascade(self):
        det = ErrorCascadeDetector(
            spe_threshold=0.5,
            consecutive_limit=3,
            cascade_risk_threshold=0.5,
        )
        for _ in range(4):
            status = det.observe(0.8)
        assert status.is_cascading
        assert status.consecutive_high == 4

    def test_recovery_resets_consecutive(self):
        det = ErrorCascadeDetector(spe_threshold=0.5)
        det.observe(0.8)
        det.observe(0.8)
        status = det.observe(0.1)
        assert status.consecutive_high == 0

    def test_trend_detection(self):
        det = ErrorCascadeDetector(window_size=10)
        for spe in [0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]:
            status = det.observe(spe)
        assert status.trend == "increasing"

    def test_stable_trend(self):
        det = ErrorCascadeDetector(window_size=10)
        for _ in range(10):
            status = det.observe(0.3)
        assert status.trend == "stable"

    def test_reset_clears_state(self):
        det = ErrorCascadeDetector(spe_threshold=0.5)
        det.observe(0.8)
        det.observe(0.8)
        det.reset()
        status = det.observe(0.1)
        assert status.consecutive_high == 0
        assert status.risk < 0.3

    def test_stats(self):
        det = ErrorCascadeDetector()
        det.observe(0.1)
        det.observe(0.9)
        stats = det.stats
        assert stats["total_steps"] == 2
        assert "mean_recent_spe" in stats


# ======================================================================
# FailureMemory tests
# ======================================================================

class TestFailureMemory:

    def test_no_records_returns_none(self):
        fm = FailureMemory()
        state = np.random.randn(64).astype(np.float32)
        action = np.random.randn(64).astype(np.float32)
        assert fm.check(state, action) is None

    def test_record_and_check_similar(self):
        fm = FailureMemory(similarity_threshold=0.7)

        state = np.random.randn(64).astype(np.float32)
        action = np.random.randn(64).astype(np.float32)

        fm.record(
            state, action,
            action_text="click save",
            error_description="button not found",
        )

        noise = 0.05 * np.random.randn(64).astype(np.float32)
        warning = fm.check(state + noise, action + noise)
        assert warning is not None
        assert warning.similarity > 0.7
        assert "button not found" in warning.pattern_description

    def test_dissimilar_no_warning(self):
        fm = FailureMemory(similarity_threshold=0.8)

        state1 = np.random.randn(64).astype(np.float32)
        action1 = np.random.randn(64).astype(np.float32)
        fm.record(state1, action1, error_description="error A")

        state2 = np.random.randn(64).astype(np.float32)
        action2 = np.random.randn(64).astype(np.float32)
        warning = fm.check(state2, action2)
        assert warning is None

    def test_decay_reduces_strength(self):
        fm = FailureMemory(decay_rate=0.9)
        state = np.random.randn(32).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)
        fm.record(state, action)

        initial_strength = fm._records[0].strength
        fm.decay()
        assert fm._records[0].strength < initial_strength

    def test_stats(self):
        fm = FailureMemory()
        assert fm.stats["stored_failures"] == 0

        state = np.random.randn(32).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)
        fm.record(state, action)
        assert fm.stats["stored_failures"] == 1
        assert fm.stats["total_recorded"] == 1


# ======================================================================
# StepMonitor tests (vector mode — no sentence-transformer needed)
# ======================================================================

class TestStepMonitorVector:
    """Test StepMonitor with pre-computed vectors (no encoder needed)."""

    def test_before_step_returns_prediction(self):
        monitor = StepMonitor(embedding_dim=32)
        action = np.random.randn(32).astype(np.float32)
        state = np.random.randn(32).astype(np.float32)

        pred = monitor.before_step(action=action, state=state)
        assert isinstance(pred, StepPrediction)
        assert pred.predicted_outcome.shape == (32,)
        assert pred.step_number == 1
        assert pred.should_proceed is True

    def test_after_step_returns_verdict(self):
        monitor = StepMonitor(embedding_dim=32)
        action = np.random.randn(32).astype(np.float32)
        state = np.random.randn(32).astype(np.float32)
        outcome = np.random.randn(32).astype(np.float32)

        monitor.before_step(action=action, state=state)
        verdict = monitor.after_step(outcome=outcome)

        assert isinstance(verdict, StepVerdict)
        assert verdict.spe >= 0
        assert verdict.learned is True
        assert verdict.step_number == 1

    def test_after_step_without_before_step(self):
        monitor = StepMonitor(embedding_dim=32)
        outcome = np.random.randn(32).astype(np.float32)
        verdict = monitor.after_step(outcome=outcome)
        assert verdict.suggestion == "no_prediction_to_compare"

    def test_cascade_detection_over_sequence(self):
        monitor = StepMonitor(
            embedding_dim=16,
            spe_threshold=0.3,
            cascade_consecutive_limit=3,
            cascade_risk_threshold=0.5,
        )

        state = np.zeros(16, dtype=np.float32)

        for i in range(5):
            action = np.random.randn(16).astype(np.float32)
            monitor.before_step(action=action, state=state)
            wildly_wrong = np.random.randn(16).astype(np.float32) * 5
            verdict = monitor.after_step(outcome=wildly_wrong)

        assert verdict.should_pause is True
        assert verdict.consecutive_errors >= 3
        assert "cascade" in verdict.suggestion.lower() or "pause" in verdict.suggestion.lower()

    def test_no_cascade_when_outcomes_match(self):
        monitor = StepMonitor(embedding_dim=16, spe_threshold=5.0)

        for i in range(5):
            state = np.random.randn(16).astype(np.float32)
            action = np.random.randn(16).astype(np.float32)
            pred = monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=pred.predicted_outcome)

        assert monitor.stats["total_pauses"] == 0

    def test_failure_memory_integration(self):
        monitor = StepMonitor(embedding_dim=32, spe_threshold=0.3)

        state = np.random.randn(32).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)

        monitor.before_step(action=action, state=state)
        bad_outcome = np.random.randn(32).astype(np.float32) * 10
        monitor.after_step(outcome=bad_outcome, success=False)

        noise = 0.05 * np.random.randn(32).astype(np.float32)
        pred = monitor.before_step(action=action + noise, state=state + noise)
        assert pred.failure_warning is not None

    def test_reset_keeps_learned_knowledge(self):
        monitor = StepMonitor(embedding_dim=16)

        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        outcome = state + 0.1 * np.random.randn(16).astype(np.float32)

        for _ in range(10):
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=outcome)

        fm_steps_before = monitor.forward_model.stats["step"]
        summary = monitor.reset()

        assert summary["steps"] == 10
        assert monitor.forward_model.stats["step"] == fm_steps_before
        assert monitor._step_count == 0

    def test_state_optional(self):
        monitor = StepMonitor(embedding_dim=16)
        action = np.random.randn(16).astype(np.float32)
        pred = monitor.before_step(action=action)
        assert pred.predicted_outcome.shape == (16,)

    def test_forward_model_learns_pattern(self):
        monitor = StepMonitor(embedding_dim=16)

        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        expected_outcome = state + 0.1 * action

        spes = []
        for _ in range(30):
            monitor.before_step(action=action, state=state)
            verdict = monitor.after_step(outcome=expected_outcome)
            spes.append(verdict.spe)

        assert spes[-1] < spes[0], "SPE should decrease as forward model learns"

    def test_episode_summary(self):
        monitor = StepMonitor(embedding_dim=16)

        for _ in range(3):
            action = np.random.randn(16).astype(np.float32)
            monitor.before_step(action=action)
            outcome = np.random.randn(16).astype(np.float32)
            monitor.after_step(outcome=outcome, success=True)

        summary = monitor.episode_summary
        assert summary["steps"] == 3
        assert summary["success_rate"] == 1.0

    def test_stats_structure(self):
        monitor = StepMonitor(embedding_dim=16)
        stats = monitor.stats
        assert "step_count" in stats
        assert "forward_model" in stats
        assert "cascade_detector" in stats
        assert "failure_memory" in stats
        assert "episode" in stats

    def test_multiple_episodes(self):
        monitor = StepMonitor(embedding_dim=16)

        for ep in range(3):
            for step in range(5):
                action = np.random.randn(16).astype(np.float32)
                monitor.before_step(action=action)
                monitor.after_step(outcome=np.random.randn(16).astype(np.float32))
            summary = monitor.reset()
            assert summary["steps"] == 5

        assert monitor.forward_model.stats["step"] == 15

    def test_dim_padding(self):
        monitor = StepMonitor(embedding_dim=32)
        short_vec = np.random.randn(16).astype(np.float32)
        pred = monitor.before_step(action=short_vec)
        assert pred.predicted_outcome.shape == (32,)

    def test_dim_truncation(self):
        monitor = StepMonitor(embedding_dim=16)
        long_vec = np.random.randn(64).astype(np.float32)
        pred = monitor.before_step(action=long_vec)
        assert pred.predicted_outcome.shape == (16,)

    def test_save_load_roundtrip(self, tmp_path):
        monitor = StepMonitor(embedding_dim=16)

        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        outcome = state + 0.1 * np.random.randn(16).astype(np.float32)

        for _ in range(5):
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=outcome, success=True)

        monitor.before_step(action=action, state=state)
        monitor.after_step(outcome=np.random.randn(16).astype(np.float32) * 10, success=False)

        failures_before = monitor.failure_memory.stats["stored_failures"]
        save_path = tmp_path / "monitor_ckpt"
        monitor.save(save_path)

        monitor2 = StepMonitor(embedding_dim=16)
        monitor2.load(save_path)

        assert monitor2.forward_model.stats["step"] == monitor.forward_model.stats["step"]
        assert monitor2.failure_memory.stats["stored_failures"] == failures_before
        assert monitor2._total_pauses == monitor._total_pauses

    def test_save_load_preserves_predictions(self, tmp_path):
        monitor = StepMonitor(embedding_dim=16)
        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(16).astype(np.float32)
        outcome = state + 0.1 * action

        for _ in range(20):
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=outcome, success=True)

        pred_before = monitor.before_step(action=action, state=state)
        monitor.after_step(outcome=outcome, success=True)

        monitor.save(tmp_path / "ckpt")
        monitor2 = StepMonitor(embedding_dim=16)
        monitor2.load(tmp_path / "ckpt")

        pred_after = monitor2.before_step(action=action, state=state)
        np.testing.assert_allclose(
            pred_before.predicted_outcome,
            pred_after.predicted_outcome,
            atol=0.02,
        )

    def test_dict_input_with_matching_dim(self):
        monitor = StepMonitor(embedding_dim=32)
        action_dict = {"type": "click", "x": 100, "y": 200}
        action_vec = monitor._encode(action_dict)
        assert action_vec.shape == (32,)

    def test_dict_input_full_dim(self):
        monitor = StepMonitor()
        action_dict = {"type": "click", "x": 100, "y": 200}
        action_vec = monitor._encode(action_dict)
        assert action_vec.shape == (384,)


# ======================================================================
# Text mode tests (require sentence-transformer)
# ======================================================================

class TestStepMonitorText:
    """Test StepMonitor with real text inputs via sentence-transformer."""

    @pytest.fixture(autouse=True)
    def _skip_if_slow(self):
        """These tests load sentence-transformer, mark them accordingly."""
        pass

    def test_text_before_after_step(self):
        monitor = StepMonitor()
        pred = monitor.before_step(
            action="click the save button",
            state="file editor is open with unsaved changes",
        )
        assert isinstance(pred, StepPrediction)
        assert pred.predicted_outcome.shape[0] == 384
        assert pred.should_proceed is True

        verdict = monitor.after_step(outcome="save dialog appeared")
        assert isinstance(verdict, StepVerdict)
        assert verdict.spe > 0

    def test_learned_pattern_low_spe(self):
        """After learning a pattern, repeating it should have low SPE."""
        monitor = StepMonitor()

        for _ in range(15):
            monitor.before_step(
                action="click the save button",
                state="file editor is open",
            )
            monitor.after_step(outcome="save dialog box appeared", success=True)

        monitor.before_step(
            action="click the save button",
            state="file editor is open",
        )
        verdict = monitor.after_step(outcome="save dialog box appeared")
        assert verdict.spe < 0.5, f"Learned pattern should have low SPE, got {verdict.spe}"

    def test_unexpected_outcome_high_spe(self):
        """After learning a pattern, a different outcome should have high SPE."""
        monitor = StepMonitor()

        for _ in range(15):
            monitor.before_step(
                action="click the save button",
                state="file editor is open",
            )
            monitor.after_step(outcome="save dialog box appeared", success=True)

        monitor.before_step(
            action="click the save button",
            state="file editor is open",
        )
        verdict = monitor.after_step(outcome="error: application crashed with segfault")
        assert verdict.spe > 0.5, f"Unexpected outcome should have high SPE, got {verdict.spe}"

    def test_text_cascade_detection(self):
        monitor = StepMonitor(cascade_consecutive_limit=3, cascade_risk_threshold=0.5)

        steps = [
            ("open file manager", "file manager window appeared"),
            ("navigate to documents", "error: directory not found"),
            ("try alternate path", "error: permission denied"),
            ("retry with sudo", "error: authentication failed"),
            ("try another approach", "error: filesystem read-only"),
        ]

        for action, outcome in steps:
            monitor.before_step(action=action, state="desktop")
            monitor.after_step(outcome=outcome)

        assert monitor.cascade_detector.stats["consecutive_high"] >= 2

    def test_text_failure_memory_warning(self):
        monitor = StepMonitor()

        monitor.before_step(
            action="delete important file",
            state="root directory",
        )
        monitor.after_step(
            outcome="critical system files deleted, system unstable",
            success=False,
        )

        pred = monitor.before_step(
            action="delete important file",
            state="root directory",
        )
        assert pred.failure_warning is not None

    def test_text_forward_model_learns(self):
        monitor = StepMonitor()

        spes = []
        for _ in range(10):
            monitor.before_step(
                action="check email inbox",
                state="email client open",
            )
            verdict = monitor.after_step(outcome="inbox loaded with 5 new messages")
            spes.append(verdict.spe)

        assert spes[-1] < spes[0], "SPE should decrease as forward model learns the pattern"

    def test_text_with_context(self):
        monitor = StepMonitor()
        pred = monitor.before_step(
            action="submit form",
            state="registration page loaded",
            context="user is signing up for a new account",
        )
        assert isinstance(pred, StepPrediction)
        assert pred.step_number == 1


# ======================================================================
# AutoRollback tests
# ======================================================================

class TestAutoRollback:
    """Tests for AutoRollback — rollback plan computation on cascade."""

    def test_rollback_plan_on_cascade(self):
        monitor = StepMonitor(cascade_consecutive_limit=2, embedding_dim=16)
        dim = 16

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=True)

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        v1 = monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        v2 = monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        assert v1.should_pause or v2.should_pause, "Cascade should be detected with 2 consecutive errors"

        plan = monitor.get_rollback_plan()
        assert plan is not None
        assert isinstance(plan, RollbackPlan)
        assert plan.rollback_to_step == 1
        assert plan.steps_wasted >= 1

    def test_no_rollback_plan_when_all_success(self):
        monitor = StepMonitor(
            embedding_dim=16,
            cascade_consecutive_limit=10,
        )
        dim = 16

        for _ in range(5):
            monitor.before_step(action=np.random.randn(dim).astype(np.float32))
            monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=True)

        assert monitor.get_rollback_plan() is None

    def test_rollback_plan_identifies_safe_checkpoint(self):
        monitor = StepMonitor(cascade_consecutive_limit=2, embedding_dim=16)
        dim = 16

        for step_num in range(1, 4):
            monitor.before_step(
                action=np.random.randn(dim).astype(np.float32),
                state=f"state at step {step_num}",
            )
            monitor.after_step(
                outcome=np.random.randn(dim).astype(np.float32),
                success=True,
            )

        monitor.before_step(
            action=np.random.randn(dim).astype(np.float32),
            state="state at step 4",
        )
        monitor.after_step(
            outcome=np.random.randn(dim).astype(np.float32),
            success=False,
        )

        monitor.before_step(
            action=np.random.randn(dim).astype(np.float32),
            state="state at step 5",
        )
        monitor.after_step(
            outcome=np.random.randn(dim).astype(np.float32),
            success=False,
        )

        plan = monitor.get_rollback_plan()
        assert plan is not None
        assert plan.rollback_to_step == 3
        assert len(plan.failed_steps) == 2

    def test_rollback_plan_reset_clears(self):
        monitor = StepMonitor(cascade_consecutive_limit=2, embedding_dim=16)
        dim = 16

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)
        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        assert monitor.get_rollback_plan() is not None

        monitor.reset()
        assert monitor.get_rollback_plan() is None

    def test_rollback_plan_in_verdict_details(self):
        monitor = StepMonitor(cascade_consecutive_limit=2, embedding_dim=16)
        dim = 16

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        monitor.before_step(action=np.random.randn(dim).astype(np.float32))
        verdict = monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        if verdict.should_pause:
            assert "rollback_plan" in verdict.details
            rp = verdict.details["rollback_plan"]
            assert "rollback_to_step" in rp
            assert "recommendation" in rp
            assert "failed_steps" in rp

    def test_rollback_plan_all_fail_goes_to_zero(self):
        monitor = StepMonitor(cascade_consecutive_limit=2, embedding_dim=16)
        dim = 16

        for _ in range(3):
            monitor.before_step(action=np.random.randn(dim).astype(np.float32))
            monitor.after_step(outcome=np.random.randn(dim).astype(np.float32), success=False)

        plan = monitor.get_rollback_plan()
        assert plan is not None
        assert plan.rollback_to_step == 0
        assert "initial state" in plan.last_safe_state
