"""Tests for TemporalPatternDetector and adaptive Phase 2 integration."""
import numpy as np
import pytest

from digital_cerebellum.core.temporal_detector import TemporalPatternDetector


class TestTemporalPatternDetector:
    def test_initial_strength_zero(self):
        td = TemporalPatternDetector()
        assert td.temporal_strength == 0.0

    def test_below_min_samples_returns_zero(self):
        td = TemporalPatternDetector(min_samples=5)
        for _ in range(4):
            s = td.observe(np.random.randn(128))
            assert s == 0.0

    def test_iid_inputs_low_strength(self):
        """Random i.i.d. inputs should yield low temporal_strength."""
        td = TemporalPatternDetector(window=20, min_samples=3)
        for _ in range(50):
            td.observe(np.random.randn(128))
        assert td.temporal_strength < 0.5

    def test_repeated_inputs_high_strength(self):
        """Identical repeated inputs should yield high temporal_strength."""
        td = TemporalPatternDetector(window=20, min_samples=3)
        x = np.random.randn(128)
        for _ in range(50):
            td.observe(x + np.random.randn(128) * 0.01)
        assert td.temporal_strength > 0.4

    def test_sequential_pattern_detected(self):
        """Alternating between two patterns should show some temporal structure."""
        td = TemporalPatternDetector(window=20, min_samples=3)
        a = np.ones(128) * 2.0
        b = np.ones(128) * -2.0
        for i in range(50):
            td.observe(a if i % 2 == 0 else b)
        assert td.temporal_strength > 0.3

    def test_stats_structure(self):
        td = TemporalPatternDetector()
        td.observe(np.random.randn(128))
        s = td.stats
        assert "temporal_strength" in s
        assert "autocorrelation" in s
        assert "variance" in s
        assert "observations" in s
        assert s["observations"] == 1

    def test_reset(self):
        td = TemporalPatternDetector(min_samples=2)
        x = np.ones(128)
        for _ in range(20):
            td.observe(x)
        assert td.temporal_strength > 0
        td.reset()
        assert td.temporal_strength == 0.0
        assert td.stats["observations"] == 0


class TestAdaptivePhase2Integration:
    """Integration tests: Phase 2 components should not degrade static performance."""

    def test_golgi_blend_near_zero_on_iid(self):
        """When temporal_strength is near 0, Golgi barely modifies z."""
        from digital_cerebellum.core.golgi_gate import GolgiGate
        import torch

        gate = GolgiGate(dim=64)
        z_orig = torch.randn(64)

        z_gated = gate(z_orig.clone())
        t_strength = 0.05
        z_blended = (1 - t_strength) * z_orig + t_strength * z_gated

        diff = (z_blended - z_orig).abs().mean().item()
        assert diff < 0.5, f"Golgi changed z too much at low t_strength: {diff}"

    def test_freq_filter_blend_near_zero_on_iid(self):
        """When temporal_strength is near 0, FreqFilter barely modifies z."""
        from digital_cerebellum.core.frequency_filter import FrequencyFilter
        import torch

        ff = FrequencyFilter(dim=64, mode="gate")
        z_orig = torch.randn(64)

        z_filtered = ff(z_orig.clone())
        t_strength = 0.05
        z_blended = (1 - t_strength) * z_orig + t_strength * z_filtered

        diff = (z_blended - z_orig).abs().mean().item()
        assert diff < 0.5, f"FreqFilter changed z too much at low t_strength: {diff}"

    def test_cerebellum_with_adaptive_phase2(self):
        """Full cerebellum with Phase 2 + Phase 3 should work without error."""
        from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig
        from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

        cfg = CerebellumConfig()
        cfg.enable_frequency_filter = True
        cfg.enable_golgi_gate = True
        cfg.enable_state_estimator = True
        cfg.enable_somatic_marker = True
        cfg.enable_curiosity_drive = True
        cfg.enable_self_model = True
        cfg.threshold_high = 0.01
        cfg.threshold_low = 0.005

        cb = DigitalCerebellum(cfg)
        cb.register_microzone(ToolCallMicrozone())

        assert cb._temporal_detector is not None

        r = cb.evaluate("tool_call", {
            "tool_name": "ls", "tool_params": {"path": "."},
        })
        assert "_event_id" in r

        stats = cb.stats
        assert "temporal_detector" in stats
        assert stats["temporal_detector"]["observations"] == 1

    def test_temporal_strength_stays_low_for_diverse_inputs(self):
        """Diverse tool calls should keep temporal_strength low."""
        from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig
        from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

        cfg = CerebellumConfig()
        cfg.enable_frequency_filter = True
        cfg.enable_golgi_gate = True
        cfg.threshold_high = 0.01
        cfg.threshold_low = 0.005

        cb = DigitalCerebellum(cfg)
        cb.register_microzone(ToolCallMicrozone())

        tools = ["ls", "cat", "grep", "mkdir", "rm", "cp", "mv", "touch",
                 "find", "curl", "wget", "ssh", "scp", "git", "docker"]
        for t in tools:
            cb.evaluate("tool_call", {
                "tool_name": t,
                "tool_params": {"path": f"/{t}_target"},
            })

        ts = cb._temporal_detector.temporal_strength
        assert ts < 0.6, f"temporal_strength too high for diverse inputs: {ts}"
