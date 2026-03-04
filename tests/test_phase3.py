"""Tests for Phase 3: Emergent cognitive properties."""

import numpy as np
import pytest

from digital_cerebellum.core.types import HeadPrediction
from digital_cerebellum.emergence.somatic_marker import SomaticMarker, GutFeeling
from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive, CuriositySignal
from digital_cerebellum.emergence.self_model import SelfModel, SelfReport


# ======================================================================
# Helpers
# ======================================================================

def _make_head_preds(k: int = 4, dim: int = 32, seed: int = 0) -> list[HeadPrediction]:
    rng = np.random.RandomState(seed)
    return [
        HeadPrediction(
            action_embedding=rng.randn(dim).astype(np.float32),
            outcome_embedding=rng.randn(dim).astype(np.float32),
        )
        for _ in range(k)
    ]


def _make_similar_head_preds(k: int = 4, dim: int = 32) -> list[HeadPrediction]:
    """All heads produce nearly identical output."""
    base_a = np.ones(dim, dtype=np.float32)
    base_o = np.ones(dim, dtype=np.float32) * 0.5
    return [
        HeadPrediction(
            action_embedding=base_a + np.random.randn(dim).astype(np.float32) * 0.01,
            outcome_embedding=base_o + np.random.randn(dim).astype(np.float32) * 0.01,
        )
        for _ in range(k)
    ]


# ======================================================================
# SomaticMarker Tests
# ======================================================================

class TestSomaticMarker:

    def test_empty_markers_neutral_feeling(self):
        sm = SomaticMarker()
        heads = _make_head_preds()
        gf = sm.feel(heads)
        assert gf.intensity == 0.0
        assert gf.label == "neutral"

    def test_fingerprint_shape(self):
        heads = _make_head_preds(k=4, dim=32)
        fp = SomaticMarker.extract_fingerprint(heads)
        # 4 heads → 6 action pairs + 6 outcome pairs = 12
        assert fp.shape == (12,)

    def test_fingerprint_deterministic(self):
        heads = _make_head_preds(k=4, dim=32, seed=42)
        fp1 = SomaticMarker.extract_fingerprint(heads)
        heads2 = _make_head_preds(k=4, dim=32, seed=42)
        fp2 = SomaticMarker.extract_fingerprint(heads2)
        np.testing.assert_array_almost_equal(fp1, fp2)

    def test_record_and_feel_negative(self):
        sm = SomaticMarker(similarity_threshold=0.5)
        # Record several bad outcomes with the same pattern
        for i in range(10):
            heads = _make_head_preds(k=4, dim=32, seed=42)
            sm.record(heads, valence=-1.0, domain="test")

        # Same pattern should now trigger alarm
        test_heads = _make_head_preds(k=4, dim=32, seed=42)
        gf = sm.feel(test_heads, domain="test")
        assert gf.valence < 0, f"Expected negative valence, got {gf.valence}"
        assert gf.intensity > 0

    def test_record_and_feel_positive(self):
        sm = SomaticMarker(similarity_threshold=0.5)
        for i in range(10):
            heads = _make_head_preds(k=4, dim=32, seed=99)
            sm.record(heads, valence=1.0, domain="test")

        test_heads = _make_head_preds(k=4, dim=32, seed=99)
        gf = sm.feel(test_heads, domain="test")
        assert gf.valence > 0

    def test_should_override_triggers_on_strong_negative(self):
        gf = GutFeeling(valence=-0.8, intensity=0.9, trigger_pattern="test")
        assert gf.should_override is True

    def test_should_override_false_for_mild(self):
        gf = GutFeeling(valence=-0.1, intensity=0.3, trigger_pattern="test")
        assert gf.should_override is False

    def test_decay_reduces_strength(self):
        sm = SomaticMarker(decay_rate=0.5)
        heads = _make_head_preds(seed=1)
        sm.record(heads, valence=1.0)
        sm.decay()
        assert sm._markers[0].strength == pytest.approx(0.5)
        sm.decay()
        assert sm._markers[0].strength == pytest.approx(0.25)

    def test_different_domains_isolated(self):
        sm = SomaticMarker(similarity_threshold=0.5)
        heads_a = _make_head_preds(k=4, dim=32, seed=10)
        heads_b = _make_head_preds(k=4, dim=32, seed=20)

        for _ in range(10):
            sm.record(heads_a, valence=-1.0, domain="payment")
            sm.record(heads_b, valence=1.0, domain="code")

        # When asking about "code" domain, the payment markers shouldn't affect
        gf = sm.feel(heads_b, domain="code")
        assert gf.valence >= 0

    def test_stats(self):
        sm = SomaticMarker()
        heads = _make_head_preds(seed=1)
        sm.record(heads, valence=1.0)
        sm.record(heads, valence=-1.0)
        s = sm.stats
        assert s["count"] == 2
        assert "mean_valence" in s


# ======================================================================
# CuriosityDrive Tests
# ======================================================================

class TestCuriosityDrive:

    def test_first_observation_maximal_novelty(self):
        cd = CuriosityDrive()
        vec = np.random.randn(64).astype(np.float32)
        sig = cd.assess("test", error=0.5, feature_vec=vec)
        assert sig.novelty == 1.0  # first observation is maximally novel

    def test_repeated_input_low_novelty(self):
        cd = CuriosityDrive()
        vec = np.ones(64, dtype=np.float32)
        # Observe the same vector many times
        for _ in range(20):
            sig = cd.assess("test", error=0.3, feature_vec=vec)
        assert sig.novelty < 0.3, f"Expected low novelty, got {sig.novelty}"

    def test_learning_progress_positive_when_improving(self):
        cd = CuriosityDrive(progress_window=10)
        # First 10 observations: high error
        for _ in range(10):
            cd.record_error("test", 0.8)
        # Next 10: low error
        for _ in range(10):
            cd.record_error("test", 0.2)
        tracker = cd._trackers["test"]
        assert tracker.learning_progress > 0

    def test_learning_progress_negative_when_regressing(self):
        cd = CuriosityDrive(progress_window=10)
        for _ in range(10):
            cd.record_error("test", 0.2)
        for _ in range(10):
            cd.record_error("test", 0.8)
        tracker = cd._trackers["test"]
        assert tracker.learning_progress < 0

    def test_explore_recommendation(self):
        cd = CuriosityDrive(progress_window=10, explore_threshold=0.05)
        for _ in range(10):
            cd.record_error("test", 0.9)
        for _ in range(10):
            cd.record_error("test", 0.3)
        vec = np.random.randn(64).astype(np.float32)
        sig = cd.assess("test", error=0.3, feature_vec=vec)
        assert sig.recommendation == "explore"

    def test_abandon_recommendation(self):
        cd = CuriosityDrive(progress_window=10, abandon_threshold=-0.05)
        for _ in range(10):
            cd.record_error("noisy", 0.2)
        for _ in range(10):
            cd.record_error("noisy", 0.9)
        vec = np.random.randn(64).astype(np.float32)
        sig = cd.assess("noisy", error=0.9, feature_vec=vec)
        assert sig.recommendation == "abandon"

    def test_exploration_ranking(self):
        cd = CuriosityDrive(progress_window=5)
        # Domain A: improving
        for _ in range(10):
            cd.record_error("A", 0.8)
        for _ in range(10):
            cd.record_error("A", 0.2)
        # Domain B: stable
        for _ in range(20):
            cd.record_error("B", 0.5)

        ranking = cd.get_exploration_ranking()
        assert len(ranking) == 2
        # A should rank higher (positive learning progress)
        assert ranking[0][0] == "A"

    def test_stats(self):
        cd = CuriosityDrive()
        cd.record_error("test", 0.5)
        cd.record_error("test", 0.3)
        s = cd.stats
        assert "test" in s
        assert s["test"]["observations"] == 2


# ======================================================================
# SelfModel Tests
# ======================================================================

class TestSelfModel:

    def test_empty_report(self):
        sm = SelfModel()
        report = sm.introspect()
        assert isinstance(report, SelfReport)
        assert len(report.competencies) == 0

    def test_novice_with_few_observations(self):
        sm = SelfModel()
        for _ in range(5):
            sm.record("test", correct=True, confidence=0.8)
        report = sm.introspect("test")
        assert report.competencies["test"].skill_level == "novice"

    def test_expert_with_high_accuracy(self):
        sm = SelfModel()
        for _ in range(50):
            sm.record("test", correct=True, confidence=0.9)
        report = sm.introspect("test")
        assert report.competencies["test"].skill_level == "expert"
        assert report.competencies["test"].accuracy > 0.9

    def test_learning_with_mixed_results(self):
        sm = SelfModel()
        for i in range(50):
            sm.record("test", correct=(i % 3 != 0), confidence=0.5 + 0.01 * i)
        report = sm.introspect("test")
        assert report.competencies["test"].skill_level in ("learning", "competent")

    def test_calibration_error_perfect(self):
        """Perfect calibration: confidence matches accuracy."""
        sm = SelfModel()
        for _ in range(50):
            sm.record("test", correct=True, confidence=1.0)
        for _ in range(50):
            sm.record("test", correct=False, confidence=0.0)
        report = sm.introspect("test")
        assert report.competencies["test"].calibration_error < 0.15

    def test_calibration_error_poor(self):
        """Poor calibration: always confident but often wrong."""
        sm = SelfModel()
        for _ in range(50):
            sm.record("test", correct=False, confidence=0.95)
        for _ in range(50):
            sm.record("test", correct=True, confidence=0.95)
        report = sm.introspect("test")
        # ECE should be significant since confidence doesn't match accuracy
        assert report.competencies["test"].calibration_error > 0.0

    def test_multiple_domains(self):
        sm = SelfModel()
        for _ in range(50):
            sm.record("payment", correct=True, confidence=0.9)
        for _ in range(50):
            sm.record("code", correct=False, confidence=0.3)
        report = sm.introspect()
        assert "payment" in report.strengths
        assert "code" in report.weaknesses

    def test_suggest_thresholds_expert(self):
        sm = SelfModel()
        for _ in range(50):
            sm.record("payment", correct=True, confidence=0.9)
        thresholds = sm.suggest_thresholds("payment")
        assert thresholds["threshold_high"] < 0.85

    def test_suggest_thresholds_novice(self):
        sm = SelfModel()
        thresholds = sm.suggest_thresholds("unknown_domain")
        assert thresholds["threshold_high"] >= 0.95

    def test_to_prompt_format(self):
        sm = SelfModel()
        for _ in range(20):
            sm.record("test", correct=True, confidence=0.8)
        report = sm.introspect()
        prompt = report.to_prompt()
        assert "[Self-Assessment]" in prompt
        assert "test" in prompt

    def test_fast_path_tracking(self):
        sm = SelfModel()
        for _ in range(20):
            sm.record("test", correct=True, confidence=0.9, route="fast")
        for _ in range(5):
            sm.record("test", correct=True, confidence=0.3, route="slow")
        report = sm.introspect("test")
        assert report.competencies["test"].fast_path_ratio > 0.7

    def test_learning_trend_improving(self):
        sm = SelfModel()
        # First half: mostly wrong
        for _ in range(25):
            sm.record("test", correct=False, confidence=0.5)
        # Second half: mostly right
        for _ in range(25):
            sm.record("test", correct=True, confidence=0.8)
        report = sm.introspect("test")
        assert report.competencies["test"].learning_trend == "improving"


# ======================================================================
# Integration: all three systems together
# ======================================================================

class TestPhase3Integration:

    def test_somatic_marker_guides_routing(self):
        """Gut feeling from somatic marker should influence decisions."""
        sm = SomaticMarker(similarity_threshold=0.5)

        # Build up negative markers
        bad_pattern = _make_head_preds(k=4, dim=32, seed=42)
        for _ in range(15):
            sm.record(bad_pattern, valence=-1.0, domain="payment")

        gf = sm.feel(bad_pattern, domain="payment")
        assert gf.should_override, "Strong negative pattern should trigger override"

    def test_curiosity_and_self_model_alignment(self):
        """Curiosity recommends exploring domains where self-model says we're novice."""
        cd = CuriosityDrive(progress_window=10, explore_threshold=0.05)
        sm = SelfModel()

        # Domain with improving errors but few observations
        for _ in range(10):
            cd.record_error("new_domain", 0.8)
        for _ in range(10):
            cd.record_error("new_domain", 0.3)
            sm.record("new_domain", correct=False, confidence=0.5)

        sig = cd.assess("new_domain", error=0.3,
                        feature_vec=np.random.randn(64).astype(np.float32))
        report = sm.introspect("new_domain")

        # Curiosity says explore, self-model says learning
        assert sig.recommendation == "explore"
        assert report.competencies["new_domain"].skill_level in ("novice", "learning")

    def test_full_lifecycle(self):
        """Simulate a complete lifecycle: observe, feel, learn, introspect."""
        sm = SomaticMarker(similarity_threshold=0.3)
        cd = CuriosityDrive(progress_window=5)
        self_m = SelfModel()

        rng = np.random.RandomState(0)

        for step in range(40):
            heads = [
                HeadPrediction(
                    action_embedding=rng.randn(32).astype(np.float32),
                    outcome_embedding=rng.randn(32).astype(np.float32),
                )
                for _ in range(4)
            ]
            success = step > 20  # starts failing, then learns
            valence = 1.0 if success else -1.0
            error = 0.8 - step * 0.015 if step > 10 else 0.8

            gf = sm.feel(heads, domain="test")
            sm.record(heads, valence=valence, domain="test")
            cd.assess("test", error=error,
                      feature_vec=rng.randn(64).astype(np.float32))
            self_m.record("test", correct=success, confidence=0.5)

        report = self_m.introspect()
        assert "test" in report.competencies
        assert sm.stats["count"] == 40
        assert cd.stats["test"]["observations"] == 40
