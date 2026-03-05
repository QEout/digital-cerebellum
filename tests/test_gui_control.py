"""Tests for Phase 8b: GUI Control components."""

from __future__ import annotations

import numpy as np
import pytest

from digital_cerebellum.micro_ops.screen_state_encoder import (
    ScreenStateEncoder,
    ScreenStateConfig,
    ROISpec,
)
from digital_cerebellum.micro_ops.gui_action_space import (
    GUIActionSpace,
    GUIActionSpaceConfig,
    GUIActionType,
)
from digital_cerebellum.micro_ops.aim_trainer import AimTrainerEnv, AimTrainerConfig
from digital_cerebellum.micro_ops.gui_controller import GUIController, GUIControlConfig


# ── ScreenStateEncoder ────────────────────────────────────────────

class TestScreenStateEncoder:

    def test_roi_encoding_produces_correct_dim(self):
        cfg = ScreenStateConfig(
            strategy="roi",
            roi_specs=[ROISpec("target", 100, 200, 50, 50)],
            screen_w=800,
            screen_h=600,
        )
        enc = ScreenStateEncoder(cfg)
        assert enc.state_dim == 6  # 1 roi * 4 + 2 (cursor)

        rois = [{"x": 100, "y": 200, "w": 50, "h": 50, "cursor_x": 0.5, "cursor_y": 0.5}]
        vec = enc.encode_rois(rois)
        assert vec.shape == (6,)
        assert vec.dtype == np.float32

    def test_downsample_encoding_produces_correct_dim(self):
        cfg = ScreenStateConfig(strategy="downsample", downsample_size=(8, 8), grayscale=True)
        enc = ScreenStateEncoder(cfg)
        assert enc.state_dim == 64

        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        vec = enc.encode_image(image)
        assert vec.shape == (64,)

    def test_downsample_rgb(self):
        cfg = ScreenStateConfig(strategy="downsample", downsample_size=(4, 4), grayscale=False)
        enc = ScreenStateEncoder(cfg)
        assert enc.state_dim == 48  # 4*4*3

    def test_hybrid_combines_roi_and_image(self):
        cfg = ScreenStateConfig(
            strategy="hybrid",
            roi_specs=[ROISpec("t", 0, 0, 10, 10)],
            downsample_size=(4, 4),
        )
        enc = ScreenStateEncoder(cfg)
        expected_dim = (1 * 4 + 2) + (4 * 4)  # roi_part + img_part
        assert enc.state_dim == expected_dim

    def test_normalized_values_change_over_time(self):
        cfg = ScreenStateConfig(strategy="roi", roi_specs=[ROISpec("t", 0, 0, 1, 1)])
        enc = ScreenStateEncoder(cfg)
        rois = [{"x": 400, "y": 300, "w": 50, "h": 50}]
        vecs = [enc.encode_rois(rois) for _ in range(20)]
        assert not np.allclose(vecs[0], vecs[-1])


# ── GUIActionSpace ────────────────────────────────────────────────

class TestGUIActionSpace:

    def test_decode_move(self):
        space = GUIActionSpace(GUIActionSpaceConfig(move_scale=100))
        action = space.decode(np.array([0.5, -0.3, -1.0, -1.0, 0.0]))
        assert action.dx == pytest.approx(50.0)
        assert action.dy == pytest.approx(-30.0)
        assert action.click is False
        assert action.action_type == GUIActionType.MOVE

    def test_decode_click(self):
        space = GUIActionSpace(GUIActionSpaceConfig(click_threshold=0.3))
        action = space.decode(np.array([0.0, 0.0, 0.8, -1.0, 0.0]))
        assert action.click is True
        assert action.action_type == GUIActionType.CLICK

    def test_encode_decode_roundtrip(self):
        space = GUIActionSpace(GUIActionSpaceConfig(move_scale=50))
        encoded = space.encode(dx=25.0, dy=-10.0, click=True)
        assert encoded[0] == pytest.approx(0.5)
        assert encoded[1] == pytest.approx(-0.2)
        assert encoded[2] == pytest.approx(1.0)

    def test_encode_absolute_move(self):
        space = GUIActionSpace(GUIActionSpaceConfig(move_scale=100))
        vec = space.encode_absolute_move(100, 100, 150, 200, click=False)
        assert vec[0] == pytest.approx(50 / 100)
        assert vec[1] == pytest.approx(100 / 100)


# ── AimTrainerEnv ─────────────────────────────────────────────────

class TestAimTrainerEnv:

    def test_environment_protocol(self):
        env = AimTrainerEnv()
        assert env.state_dim == 9
        assert env.action_dim == 3

    def test_observe_returns_correct_shape(self):
        env = AimTrainerEnv()
        state = env.observe()
        assert state.shape == (9,)
        assert state.dtype == np.float32

    def test_execute_returns_reward(self):
        env = AimTrainerEnv()
        reward = env.execute(np.array([0.0, 0.0, 0.0]))
        assert isinstance(reward, float)

    def test_hit_gives_positive_reward(self):
        env = AimTrainerEnv(AimTrainerConfig(
            target_radius=60, auto_click=True, move_speed=200, noise=0.0,
        ))
        got_hit = False
        for _ in range(50):
            state = env.observe()
            reward = env.execute(np.array([state[4] * 3, state[5] * 3, 0.0]))
            if env.stats["hits"] >= 1:
                got_hit = True
                break
        assert got_hit, f"No hits after 50 steps, final stats: {env.stats}"

    def test_stats_track_hits(self):
        env = AimTrainerEnv(AimTrainerConfig(
            target_radius=60, auto_click=True, move_speed=200, noise=0.0,
        ))
        for _ in range(50):
            state = env.observe()
            env.execute(np.array([state[4] * 3, state[5] * 3, 0.0]))
        assert env.stats["hits"] >= 1

    def test_reset_clears_state(self):
        env = AimTrainerEnv()
        for _ in range(50):
            env.execute(np.array([0.5, 0.5, 0.0]))
        env.reset()
        assert env.stats["total_steps"] == 0
        assert env.stats["hits"] == 0

    def test_adaptive_difficulty_shrinks_target(self):
        env = AimTrainerEnv(AimTrainerConfig(
            target_radius=50.0,
            target_radius_min=10.0,
            adaptive_difficulty=True,
            auto_click=True,
            move_speed=200,
            noise=0.0,
        ))
        for _ in range(2000):
            state = env.observe()
            env.execute(np.array([state[4] * 3, state[5] * 3, 0.0]))
        assert env.stats["hits"] >= 5, f"Only {env.stats['hits']} hits"
        assert env.stats["target_radius"] < 50.0, f"Radius still {env.stats['target_radius']}"


# ── GUIController ─────────────────────────────────────────────────

class TestGUIController:

    def test_step_produces_trial_result(self):
        env = AimTrainerEnv()
        ctrl = GUIController(env.state_dim, env.action_dim)
        result = ctrl.step(env)
        assert result.step == 1
        assert result.latency_ms > 0
        assert result.final_action.shape == (3,)

    def test_cortex_signal_points_toward_target(self):
        env = AimTrainerEnv()
        ctrl = GUIController(env.state_dim, env.action_dim, GUIControlConfig(
            cortex_gain=5.0, cortex_noise=0.0,
        ))
        state = env.observe()
        signal = ctrl.cortex_signal(state)
        dx, dy = state[4], state[5]
        assert np.sign(signal[0]) == np.sign(dx) or abs(dx) < 0.01
        assert np.sign(signal[1]) == np.sign(dy) or abs(dy) < 0.01

    def test_run_episode_returns_metrics(self):
        env = AimTrainerEnv()
        ctrl = GUIController(env.state_dim, env.action_dim)
        result = ctrl.run_episode(env, n_steps=50)
        assert "total_reward" in result
        assert "mean_spe" in result
        assert "noise_scale" in result

    def test_noise_decays_per_episode(self):
        env = AimTrainerEnv()
        ctrl = GUIController(env.state_dim, env.action_dim, GUIControlConfig(
            cortex_noise=1.0, noise_decay=0.5,
        ))
        ctrl.run_episode(env, n_steps=10)
        assert ctrl._noise_scale == pytest.approx(0.5)
        ctrl.run_episode(env, n_steps=10)
        assert ctrl._noise_scale == pytest.approx(0.25)

    def test_learning_improves_over_episodes(self):
        """Core test: hits should increase over multiple episodes."""
        env = AimTrainerEnv(AimTrainerConfig(
            target_radius=50, auto_click=True, move_speed=60,
        ))
        ctrl = GUIController(env.state_dim, env.action_dim, GUIControlConfig(
            cortex_gain=0.8, cortex_noise=1.0, noise_decay=0.85,
        ))

        early_hits = []
        late_hits = []
        for ep in range(15):
            env.reset()
            ctrl.run_episode(env, n_steps=300)
            if ep < 3:
                early_hits.append(env.stats["hits"])
            if ep >= 12:
                late_hits.append(env.stats["hits"])

        assert np.mean(late_hits) >= np.mean(early_hits)
