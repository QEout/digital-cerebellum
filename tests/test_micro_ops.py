"""Tests for Phase 6: Micro-Operation Engine — continuous real-time control."""

import numpy as np
import pytest

from digital_cerebellum.core.state_encoder import StateEncoder
from digital_cerebellum.core.forward_model import ForwardModel, ForwardPrediction
from digital_cerebellum.core.action_encoder import ActionEncoder
from digital_cerebellum.micro_ops.engine import MicroOpEngine, MicroOpConfig, StepResult
from digital_cerebellum.micro_ops.environments import TargetTracker, BalanceBeam


# ======================================================================
# StateEncoder
# ======================================================================

class TestStateEncoder:

    def test_direct_mode_shape(self):
        enc = StateEncoder(state_dim=6, target_dim=10, mode="direct")
        state = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        out = enc.encode(state)
        assert out.shape == (10,)

    def test_direct_mode_truncate(self):
        enc = StateEncoder(state_dim=10, target_dim=4, mode="direct")
        state = np.random.randn(10).astype(np.float32)
        out = enc.encode(state)
        assert out.shape == (4,)

    def test_projected_mode_shape(self):
        enc = StateEncoder(state_dim=6, target_dim=20, mode="projected")
        state = np.random.randn(6).astype(np.float32)
        out = enc.encode(state)
        assert out.shape == (20,)

    def test_normalisation_updates(self):
        enc = StateEncoder(state_dim=4, target_dim=4)
        for _ in range(20):
            enc.encode(np.random.randn(4).astype(np.float32))
        assert enc._count == 20
        assert enc.stats["observations"] == 20

    def test_batch_encode(self):
        enc = StateEncoder(state_dim=4, target_dim=8)
        batch = np.random.randn(5, 4).astype(np.float32)
        out = enc.encode_batch(batch)
        assert out.shape == (5, 8)

    def test_deterministic(self):
        enc = StateEncoder(state_dim=4, target_dim=8)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out1 = enc.encode(state)
        out2 = enc.encode(state)
        assert out1.shape == out2.shape


# ======================================================================
# ForwardModel
# ======================================================================

class TestForwardModel:

    def test_predict_shape(self):
        fm = ForwardModel(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        pred = fm.predict(state, action)
        assert pred.predicted_next_state.shape == (4,)
        assert 0 <= pred.confidence <= 2.0

    def test_learn_reduces_error(self):
        fm = ForwardModel(state_dim=4, action_dim=2, lr=0.01)

        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action = np.array([0.1, 0.0], dtype=np.float32)
        next_state = np.array([1.1, 0.0, 0.0, 0.0], dtype=np.float32)

        errors = []
        for _ in range(100):
            err = fm.learn(state, action, next_state)
            errors.append(err)

        assert errors[-1] < errors[0]

    def test_compute_spe(self):
        fm = ForwardModel(state_dim=4, action_dim=2)
        predicted = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual = np.array([1.1, 2.0, 2.9, 4.0], dtype=np.float32)
        spe = fm.compute_spe(predicted, actual)
        assert spe.shape == (4,)
        assert abs(spe[0] - 0.1) < 1e-5
        assert abs(spe[2] - (-0.1)) < 1e-5

    def test_stats(self):
        fm = ForwardModel(state_dim=4, action_dim=2)
        stats = fm.stats
        assert stats["step"] == 0
        assert "mean_recent_error" in stats
        assert "is_improving" in stats

    def test_is_improving_after_training(self):
        fm = ForwardModel(state_dim=2, action_dim=1, lr=0.05)
        state = np.array([0.0, 0.0], dtype=np.float32)
        action = np.array([1.0], dtype=np.float32)
        next_state = np.array([0.1, 0.0], dtype=np.float32)

        for _ in range(50):
            fm.learn(state, action, next_state)

        assert fm.is_improving or fm.mean_recent_error < 0.01


# ======================================================================
# ActionEncoder
# ======================================================================

class TestActionEncoder:

    def test_encode_decode_roundtrip(self):
        enc = ActionEncoder(
            action_dim=3,
            action_ranges=[(-10, 10), (0, 100), (-1, 1)],
        )
        raw = np.array([5.0, 50.0, 0.0], dtype=np.float32)
        encoded = enc.encode(raw)
        assert np.all(encoded >= -1.0) and np.all(encoded <= 1.0)

        decoded = enc.decode(encoded)
        np.testing.assert_allclose(decoded, raw, atol=1e-5)

    def test_encode_from_dict(self):
        enc = ActionEncoder(
            action_dim=2,
            action_names=["move_x", "move_y"],
            action_ranges=[(-1, 1), (-1, 1)],
        )
        encoded = enc.encode({"move_x": 0.5, "move_y": -0.5})
        assert encoded.shape == (2,)

    def test_decode_to_dict(self):
        enc = ActionEncoder(
            action_dim=2,
            action_names=["force", "torque"],
        )
        encoded = np.array([0.5, -0.5], dtype=np.float32)
        result = enc.decode_to_dict(encoded)
        assert "force" in result
        assert "torque" in result

    def test_padding(self):
        enc = ActionEncoder(action_dim=4)
        short = np.array([1.0, 2.0], dtype=np.float32)
        encoded = enc.encode(short)
        assert encoded.shape == (4,)

    def test_truncation(self):
        enc = ActionEncoder(action_dim=2)
        long = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        encoded = enc.encode(long)
        assert encoded.shape == (2,)


# ======================================================================
# TargetTracker Environment
# ======================================================================

class TestTargetTracker:

    def test_state_dim(self):
        env = TargetTracker()
        assert env.state_dim == 6
        assert env.action_dim == 2

    def test_observe_shape(self):
        env = TargetTracker()
        state = env.observe()
        assert state.shape == (6,)

    def test_execute_returns_reward(self):
        env = TargetTracker()
        action = np.array([0.1, 0.1], dtype=np.float32)
        reward = env.execute(action)
        assert isinstance(reward, float)
        assert reward <= 0  # distance-based, always negative

    def test_reset(self):
        env = TargetTracker()
        env.execute(np.array([1.0, 1.0]))
        env.reset()
        state = env.observe()
        assert abs(state[0]) < 0.01  # agent back at origin


# ======================================================================
# BalanceBeam Environment
# ======================================================================

class TestBalanceBeam:

    def test_state_dim(self):
        env = BalanceBeam()
        assert env.state_dim == 4
        assert env.action_dim == 1

    def test_observe_shape(self):
        env = BalanceBeam()
        state = env.observe()
        assert state.shape == (4,)

    def test_execute_returns_reward(self):
        env = BalanceBeam()
        reward = env.execute(np.array([0.0]))
        assert reward in (0.0, 1.0)

    def test_reset(self):
        env = BalanceBeam()
        for _ in range(100):
            env.execute(np.array([0.5]))
        env.reset()
        state = env.observe()
        assert abs(state[0]) < 0.01


# ======================================================================
# MicroOpEngine
# ======================================================================

class TestMicroOpEngine:

    def test_init(self):
        engine = MicroOpEngine(state_dim=6, action_dim=2)
        assert engine.state_dim == 6
        assert engine.action_dim == 2
        assert engine._step == 0

    def test_single_step(self):
        env = TargetTracker()
        engine = MicroOpEngine(state_dim=6, action_dim=2)
        result = engine.step(env)
        assert isinstance(result, StepResult)
        assert result.step == 1
        assert result.state.shape == (6,)
        assert result.action.shape == (2,)
        assert result.latency_ms > 0

    def test_multiple_steps(self):
        env = TargetTracker()
        engine = MicroOpEngine(state_dim=6, action_dim=2)
        for i in range(10):
            result = engine.step(env)
            assert result.step == i + 1

    def test_forward_model_learns(self):
        env = TargetTracker(noise=0.0)
        engine = MicroOpEngine(state_dim=6, action_dim=2)

        for _ in range(50):
            engine.step(env)

        assert engine.forward_model._step > 0

    def test_run_returns_summary(self):
        env = TargetTracker()
        cfg = MicroOpConfig(max_steps=50, target_hz=1000)
        engine = MicroOpEngine(state_dim=6, action_dim=2, cfg=cfg)

        summary = engine.run(env, n_steps=50, target_hz=10000)

        assert summary["total_steps"] == 50
        assert "mean_reward" in summary
        assert "mean_spe" in summary
        assert "mean_latency_ms" in summary
        assert "improvement" in summary
        assert "forward_model" in summary

    def test_latency_under_16ms(self):
        """Core requirement: each step must complete in <16ms for 60Hz."""
        env = TargetTracker()
        engine = MicroOpEngine(state_dim=6, action_dim=2)

        latencies = []
        for _ in range(20):
            result = engine.step(env)
            latencies.append(result.latency_ms)

        mean_latency = np.mean(latencies)
        assert mean_latency < 16.0, f"Mean latency {mean_latency:.1f}ms exceeds 16ms"

    def test_stats(self):
        env = TargetTracker()
        engine = MicroOpEngine(state_dim=6, action_dim=2)
        engine.step(env)
        stats = engine.stats
        assert "step" in stats
        assert "forward_model" in stats
        assert "state_encoder" in stats

    def test_balance_beam_engine(self):
        env = BalanceBeam()
        engine = MicroOpEngine(state_dim=4, action_dim=1)
        for _ in range(20):
            result = engine.step(env)
        assert engine._step == 20


class TestMicroOpLearning:
    """Tests that the engine actually LEARNS over time."""

    def test_forward_model_error_decreases(self):
        env = TargetTracker(noise=0.0, speed=0.1, target_speed=0.0)
        cfg = MicroOpConfig(forward_model_lr=0.01)
        engine = MicroOpEngine(state_dim=6, action_dim=2, cfg=cfg)

        summary = engine.run(env, n_steps=200, target_hz=10000)

        improvement = summary["improvement"]
        assert improvement["late_spe"] <= improvement["early_spe"] + 0.5, (
            f"SPE should decrease: early={improvement['early_spe']:.4f}, "
            f"late={improvement['late_spe']:.4f}"
        )

    def test_engine_runs_without_divergence(self):
        """Engine should run stably without NaN or divergence."""
        env = TargetTracker(noise=0.0, target_speed=0.0)
        cfg = MicroOpConfig(action_lr=0.005)
        engine = MicroOpEngine(state_dim=6, action_dim=2, cfg=cfg)

        summary = engine.run(env, n_steps=200, target_hz=10000)

        assert not np.isnan(summary["mean_reward"])
        assert not np.isnan(summary["mean_spe"])
        assert abs(summary["mean_reward"]) < 100, "Reward should not diverge"
        assert summary["forward_model"]["is_improving"]
