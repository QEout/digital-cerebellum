"""Tests for Phase 2 components: Frequency Filter, Golgi Gate, State Estimator."""

import time
import torch
import numpy as np
import pytest

from digital_cerebellum.core.frequency_filter import FrequencyFilter
from digital_cerebellum.core.golgi_gate import GolgiGate
from digital_cerebellum.core.state_estimator import StateEstimator, StateSnapshot


# ======================================================================
# Frequency Filter
# ======================================================================

class TestFrequencyFilter:

    def test_gate_mode_preserves_dim(self):
        ff = FrequencyFilter(dim=128, alpha=0.1, mode="gate")
        z = torch.randn(128)
        out = ff(z)
        assert out.shape == (128,)

    def test_concat_mode_doubles_dim(self):
        ff = FrequencyFilter(dim=128, alpha=0.1, mode="concat")
        z = torch.randn(128)
        out = ff(z)
        assert out.shape == (256,)
        assert ff.output_dim == 256

    def test_batch_input(self):
        ff = FrequencyFilter(dim=64, alpha=0.2, mode="gate")
        z = torch.randn(8, 64)
        out = ff(z)
        assert out.shape == (8, 64)

    def test_ema_tracks_signal(self):
        ff = FrequencyFilter(dim=32, alpha=0.5, mode="gate")
        constant = torch.ones(32) * 3.0
        for _ in range(20):
            ff(constant)
        np.testing.assert_allclose(
            ff._ema.numpy(), np.full(32, 3.0), atol=0.1
        )

    def test_high_pass_detects_transients(self):
        ff = FrequencyFilter(dim=16, alpha=0.1, mode="concat")
        baseline = torch.zeros(16)
        for _ in range(10):
            ff(baseline)
        spike = torch.ones(16) * 10.0
        out = ff(spike)
        low_part = out[:16]
        high_part = out[16:]
        assert high_part.abs().mean() > low_part.abs().mean()

    def test_reset(self):
        ff = FrequencyFilter(dim=32, alpha=0.5, mode="gate")
        ff(torch.randn(32))
        assert ff._initialized
        ff.reset()
        assert not ff._initialized
        assert ff._ema.abs().sum().item() == 0.0


# ======================================================================
# Golgi Gate
# ======================================================================

class TestGolgiGate:

    def test_output_shape(self):
        gg = GolgiGate(dim=128, target_sparsity=0.1)
        z = torch.randn(128)
        out = gg(z)
        assert out.shape == (128,)

    def test_batch_shape(self):
        gg = GolgiGate(dim=64)
        z = torch.randn(4, 64)
        out = gg(z)
        assert out.shape == (4, 64)

    def test_gating_reduces_magnitude(self):
        gg = GolgiGate(dim=128, target_sparsity=0.1)
        z = torch.randn(128) * 10.0
        out = gg(z)
        assert out.abs().mean() <= z.abs().mean()

    def test_inhibition_increases_with_high_activity(self):
        gg = GolgiGate(dim=64, target_sparsity=0.05, feedback_lr=0.1)
        z_active = torch.ones(64) * 5.0
        init_inhib = gg._inhibition_bias.mean().item()
        for _ in range(20):
            gg(z_active)
        final_inhib = gg._inhibition_bias.mean().item()
        assert final_inhib > init_inhib

    def test_stats(self):
        gg = GolgiGate(dim=32, target_sparsity=0.2)
        gg(torch.randn(32))
        s = gg.stats
        assert "mean_activity" in s
        assert "mean_inhibition" in s
        assert s["target_sparsity"] == 0.2


# ======================================================================
# State Estimator
# ======================================================================

class TestStateEstimator:

    def test_empty_snapshot(self):
        se = StateEstimator(state_dim=32)
        snap = se.get_snapshot()
        assert snap.action_count == 0
        assert snap.success_rate == 0.5

    def test_record_updates_count(self):
        se = StateEstimator(state_dim=32)
        se.record_event("send_email", "fast", 0.95, success=True)
        se.record_event("search_web", "slow", 0.3, success=True)
        snap = se.get_snapshot()
        assert snap.action_count == 2
        assert snap.fast_path_count == 1
        assert snap.slow_path_count == 1

    def test_success_rate(self):
        se = StateEstimator(state_dim=32)
        for _ in range(8):
            se.record_event("tool_a", "fast", 0.9, success=True)
        for _ in range(2):
            se.record_event("tool_b", "slow", 0.3, success=False)
        snap = se.get_snapshot()
        assert abs(snap.success_rate - 0.8) < 0.01

    def test_risk_ema_tracks(self):
        se = StateEstimator(state_dim=32)
        for _ in range(50):
            se.record_event("tool", "fast", 0.9, risk_score=1.0)
        snap = se.get_snapshot()
        assert snap.risk_level > 0.9

    def test_forward_produces_embedding(self):
        se = StateEstimator(state_dim=64)
        se.record_event("tool", "fast", 0.8, success=True)
        vec = se()
        assert vec.shape == (64,)

    def test_forward_without_events(self):
        se = StateEstimator(state_dim=32)
        vec = se()
        assert vec.shape == (32,)

    def test_stats(self):
        se = StateEstimator(state_dim=32)
        se.record_event("tool", "fast", 0.95, success=True)
        s = se.stats
        assert s["action_count"] == 1
        assert "fast_ratio" in s
        assert "risk_level" in s

    def test_recent_tools(self):
        se = StateEstimator(state_dim=32)
        for name in ["a", "b", "c", "d", "e", "f"]:
            se.record_event(name, "fast", 0.9)
        snap = se.get_snapshot()
        assert snap.recent_tools == ["b", "c", "d", "e", "f"]


# ======================================================================
# Integration: Phase 2 components in the pipeline
# ======================================================================

class TestPhase2Integration:

    def test_cerebellum_with_all_phase2(self):
        """DigitalCerebellum can be created with all Phase 2 components enabled."""
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

        cfg = CerebellumConfig(
            enable_frequency_filter=True,
            frequency_alpha=0.2,
            enable_golgi_gate=True,
            golgi_target_sparsity=0.1,
            enable_state_estimator=True,
            state_dim=32,
        )
        cb = DigitalCerebellum(cfg)
        cb.register_microzone(ToolCallMicrozone())

        assert cb._freq_filter is not None
        assert cb._golgi_gate is not None
        assert cb._state_estimator is not None

    def test_cerebellum_default_no_phase2(self):
        """By default, Phase 2 components are disabled."""
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum

        cb = DigitalCerebellum(CerebellumConfig())
        assert cb._freq_filter is None
        assert cb._golgi_gate is None
        assert cb._state_estimator is None
