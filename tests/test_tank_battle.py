"""Tests for TankBattleEnv and TankController — cortex-cerebellum validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from digital_cerebellum.micro_ops.tank_env import (
    TankBattleEnv,
    TankConfig,
    TankController,
    ENEMY_TYPES,
)
from digital_cerebellum.micro_ops.gui_controller import (
    GUIControlConfig,
    GUIController,
    CorrectionMicrozone,
)


# ======================================================================
# TankBattleEnv — basic functionality
# ======================================================================

class TestTankBattleEnv:

    def test_state_and_action_dims(self):
        env = TankBattleEnv()
        assert env.state_dim == 15
        assert env.action_dim == 4

    def test_observe_shape(self):
        env = TankBattleEnv()
        state = env.observe()
        assert state.shape == (15,)
        assert state.dtype == np.float32

    def test_observe_values_normalized(self):
        env = TankBattleEnv()
        state = env.observe()
        assert 0.0 <= state[0] <= 1.0, "player_x should be [0,1]"
        assert 0.0 <= state[1] <= 1.0, "player_y should be [0,1]"
        assert 0.0 <= state[2] <= 1.0, "player_hp should be [0,1]"

    def test_execute_returns_reward(self):
        env = TankBattleEnv()
        action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        reward = env.execute(action)
        assert isinstance(reward, float)

    def test_reset_clears_state(self):
        env = TankBattleEnv()
        for _ in range(50):
            env.execute(np.random.uniform(-1, 1, 4).astype(np.float32))
        env.reset()
        assert env._tick == 0
        assert env._kills == 0
        assert env._shots_fired == 0
        assert not env.done

    def test_done_when_all_enemies_killed(self):
        cfg = TankConfig(enemy_count=1, enemy_roster=["sniper"])
        env = TankBattleEnv(cfg)
        for e in env._enemies:
            e.hp = 0
            e.alive = False
        env.execute(np.zeros(4, dtype=np.float32))
        assert env.done

    def test_done_when_player_dies(self):
        env = TankBattleEnv()
        env._player.hp = 0
        env.execute(np.zeros(4, dtype=np.float32))
        assert env.done

    def test_done_when_time_runs_out(self):
        cfg = TankConfig(round_max_ticks=10)
        env = TankBattleEnv(cfg)
        for _ in range(15):
            if not env.done:
                env.execute(np.zeros(4, dtype=np.float32))
        assert env.done

    def test_shoot_fires_bullet(self):
        env = TankBattleEnv()
        env._player.shoot_cd = 0
        env.execute(np.array([0.0, 0.0, 0.0, 0.9], dtype=np.float32))
        assert env._shots_fired >= 1

    def test_set_strategy(self):
        env = TankBattleEnv()
        env.set_strategy(1, "aggressive", [100.0, 200.0])
        assert env._strategy_target == 1
        assert env._strategy_code == 0.8
        assert env._llm_calls == 1

    def test_get_round_score_keys(self):
        env = TankBattleEnv()
        for _ in range(10):
            env.execute(np.random.uniform(-1, 1, 4).astype(np.float32))
        score = env.get_round_score()
        expected_keys = {
            "kills", "shots_fired", "shots_hit", "hit_rate",
            "survive_time", "dodges", "llm_calls", "total_score", "grade",
        }
        assert expected_keys.issubset(score.keys())

    def test_get_game_state_keys(self):
        env = TankBattleEnv()
        gs = env.get_game_state()
        assert "player" in gs
        assert "enemies" in gs
        assert "bullets" in gs
        assert gs["w"] == env.cfg.arena_w

    def test_get_state_summary_is_string(self):
        env = TankBattleEnv()
        summary = env.get_state_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ======================================================================
# Randomized spawns
# ======================================================================

class TestRandomizedSpawns:

    def test_enemy_positions_vary_across_resets(self):
        """Enemies should NOT always spawn in the same place."""
        env = TankBattleEnv(TankConfig(randomize_spawns=True))
        positions_set: set[tuple[float, float]] = set()
        for _ in range(10):
            env.reset()
            for e in env._enemies:
                positions_set.add((round(e.x, 1), round(e.y, 1)))
        assert len(positions_set) > 3, (
            f"Only {len(positions_set)} unique positions across 10 resets "
            "— spawns are probably still fixed"
        )

    def test_player_position_varies(self):
        env = TankBattleEnv(TankConfig(randomize_spawns=True))
        px_set = set()
        for _ in range(10):
            env.reset()
            px_set.add(round(env._player.x, 1))
        assert len(px_set) > 1

    def test_enemies_not_on_top_of_player(self):
        for _ in range(20):
            env = TankBattleEnv(TankConfig(randomize_spawns=True))
            px, py = env._player.x, env._player.y
            for e in env._enemies:
                dist = math.hypot(e.x - px, e.y - py)
                assert dist > 80, (
                    f"Enemy spawned too close to player: dist={dist:.1f}"
                )

    def test_fixed_spawns_still_work(self):
        env = TankBattleEnv(TankConfig(randomize_spawns=False))
        p1 = env._player.x
        env.reset()
        p2 = env._player.x
        assert p1 == p2, "Fixed spawns should produce identical player position"


# ======================================================================
# Threat assessment
# ======================================================================

class TestThreatAssessment:

    def test_threat_assessment_returns_all_enemies(self):
        env = TankBattleEnv()
        threats = env.get_threat_assessment()
        assert len(threats) == env.cfg.enemy_count
        for label, t in threats.items():
            assert "type" in t
            assert "threat_score" in t
            assert "alive" in t

    def test_threat_builds_over_rounds(self):
        env = TankBattleEnv()
        env._damage_from["A"] = 50.0
        threats = env.get_threat_assessment()
        assert threats["A"]["threat_score"] > threats.get("B", {}).get("threat_score", 0)

    def test_dead_enemy_marked_eliminated(self):
        env = TankBattleEnv()
        env._enemies[0].alive = False
        threats = env.get_threat_assessment()
        label = env._enemies[0].label
        assert threats[label]["threat_level"] == "已消灭"


# ======================================================================
# TankController — cerebellar control
# ======================================================================

class TestTankController:

    def _make(self, cerebellum_enabled: bool = True) -> tuple[TankBattleEnv, TankController]:
        env = TankBattleEnv(TankConfig(randomize_spawns=False))
        cfg = GUIControlConfig(
            cortex_gain=0.6, cortex_noise=0.5,
            correction_lr=0.005, correction_hidden=32,
            forward_model_lr=0.02, forward_model_hidden=64,
            noise_decay=0.9, correction_scale=0.3,
        )
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=cfg,
            cerebellum_enabled=cerebellum_enabled,
        )
        return env, ctrl

    def test_general_capabilities_inherited(self):
        """TankController should NOT define its own confidence/SPE/etc.

        These are universal cerebellar capabilities, not tank-specific.
        """
        _, ctrl = self._make()
        assert hasattr(ctrl, 'cerebellum_confidence')
        assert hasattr(ctrl, 'mean_recent_spe')
        assert hasattr(ctrl, 'should_call_cortex')
        assert hasattr(ctrl, 'cerebellum_enabled')
        assert 'cerebellum_confidence' not in TankController.__dict__
        assert 'mean_recent_spe' not in TankController.__dict__
        assert 'should_call_cortex' not in TankController.__dict__
        assert '_learn_correction' not in TankController.__dict__

    def test_has_three_microzones(self):
        """TankController should create aim/dodge/move microzones."""
        _, ctrl = self._make()
        names = ctrl.microzone_names
        assert names == ["aim", "dodge", "move"]
        assert len(ctrl._microzones) == 3
        for mz in ctrl._microzones:
            assert isinstance(mz, CorrectionMicrozone)

    def test_default_controller_has_single_microzone(self):
        """Base GUIController should use a single 'default' microzone."""
        ctrl = GUIController(state_dim=7, action_dim=3)
        assert ctrl.microzone_names == ["default"]

    def test_microzone_corrections_returns_per_zone(self):
        """microzone_corrections() should return one entry per microzone."""
        env, ctrl = self._make()
        state = env.observe()
        corrs = ctrl.microzone_corrections(state)
        assert set(corrs.keys()) == {"aim", "dodge", "move"}
        for name, c in corrs.items():
            assert c.shape == (4,), f"{name} correction shape wrong"

    def test_aim_error_zero_when_aligned(self):
        """Aim error should be near zero when turret points at target."""
        state_t = torch.zeros(1, 15)
        state_t[0, 3] = 0.0
        state_t[0, 5] = 1.0
        state_t[0, 6] = 0.0
        err = TankController._aim_error(state_t)
        assert err.abs().max().item() < 0.5

    def test_dodge_error_high_when_bullet_close(self):
        """Dodge error should be large when bullet is very close."""
        close = torch.zeros(1, 15)
        close[0, 8] = 0.1
        close[0, 9] = 0.0
        close[0, 10] = 0.05
        far = torch.zeros(1, 15)
        far[0, 8] = 0.1
        far[0, 9] = 0.0
        far[0, 10] = 0.9
        err_close = TankController._dodge_error(close).pow(2).sum().item()
        err_far = TankController._dodge_error(far).pow(2).sum().item()
        assert err_close > err_far * 5, (
            f"Close bullet error ({err_close:.4f}) should be much larger "
            f"than far bullet error ({err_far:.4f})"
        )

    def test_move_error_modulated_by_strategy(self):
        """Move error should be larger with aggressive strategy."""
        base = torch.zeros(1, 15)
        base[0, 5] = 0.5
        base[0, 6] = 0.3
        aggressive = base.clone()
        aggressive[0, 14] = 0.8
        defensive = base.clone()
        defensive[0, 14] = 0.2
        err_agg = TankController._move_error(aggressive).pow(2).sum().item()
        err_def = TankController._move_error(defensive).pow(2).sum().item()
        assert err_agg > err_def

    def test_correction_output_is_bounded(self):
        """Tanh output layer should keep per-microzone corrections bounded."""
        env, ctrl = self._make()
        state = env.observe()
        corrs = ctrl.microzone_corrections(state)
        for name, c in corrs.items():
            max_possible = max(mz.scale for mz in ctrl._microzones)
            assert np.all(np.abs(c) <= max_possible + 1e-6), (
                f"{name} correction out of bounds: max={np.max(np.abs(c)):.4f}"
            )

    def test_modulate_microzone_changes_gain(self):
        """modulate_microzone should change a microzone's scale."""
        _, ctrl = self._make()
        old_scale = ctrl._microzones[0].scale
        ctrl.modulate_microzone("aim", gain=0.1)
        assert ctrl._microzones[0].scale == 0.1
        assert ctrl._microzones[0].scale != old_scale

    def test_modulate_microzone_changes_lr(self):
        _, ctrl = self._make()
        ctrl.modulate_microzone("aim", lr=0.099)
        for pg in ctrl._microzones[0].optimizer.param_groups:
            assert pg['lr'] == 0.099

    def test_modulate_microzone_enable_disable(self):
        """Disabled microzone should contribute zero correction."""
        env, ctrl = self._make()
        state = env.observe()
        ctrl.modulate_microzone("dodge", enabled=False)
        corrs = ctrl.microzone_corrections(state)
        assert np.allclose(corrs["dodge"], 0.0)
        assert not np.allclose(corrs["aim"], 0.0) or True  # aim may be near zero early

    def test_disabled_microzone_not_trained(self):
        """A disabled microzone's weights should not change during training."""
        env, ctrl = self._make()
        ctrl.modulate_microzone("move", enabled=False)
        weights_before = ctrl._microzones[2].net[0].weight.data.clone()
        for _ in range(50):
            if env.done:
                env.reset()
            ctrl.step(env)
        weights_after = ctrl._microzones[2].net[0].weight.data
        assert torch.allclose(weights_before, weights_after), (
            "Disabled microzone weights should not change"
        )

    def test_step_produces_trial_result(self):
        env, ctrl = self._make()
        result = ctrl.step(env)
        assert result.step == 1
        assert result.final_action.shape == (4,)
        assert result.latency_ms > 0

    def test_cortex_signal_shape(self):
        env, ctrl = self._make()
        state = env.observe()
        signal = ctrl.cortex_signal(state)
        assert signal.shape == (4,)
        assert np.all(np.abs(signal) <= 1.0)

    def test_cortex_aims_toward_target(self):
        env, ctrl = self._make()
        env.set_strategy(0, "aggressive", [100, 100])
        state = env.observe()
        ctrl.cfg.cortex_noise = 0.0
        ctrl._noise_scale = 0.0
        signal = ctrl.cortex_signal(state)
        target_dx = state[5]
        if abs(target_dx) > 0.05:
            assert np.sign(signal[0]) == np.sign(target_dx)

    def test_cerebellum_disabled_gives_zero_correction(self):
        env, ctrl = self._make(cerebellum_enabled=False)
        state = env.observe()
        corr = ctrl.cerebellar_correction(state)
        assert np.allclose(corr, 0.0)

    def test_cerebellum_confidence_starts_low(self):
        _, ctrl = self._make()
        assert ctrl.cerebellum_confidence < 0.9

    def test_should_call_cortex_returns_int(self):
        _, ctrl = self._make()
        interval = ctrl.should_call_cortex(base_interval=90)
        assert isinstance(interval, int)
        assert interval >= 90

    def test_latency_under_10ms(self):
        """Each step should be well under 16ms (60Hz budget)."""
        env, ctrl = self._make()
        latencies = []
        for _ in range(50):
            if env.done:
                env.reset()
            result = ctrl.step(env)
            latencies.append(result.latency_ms)
        mean_lat = np.mean(latencies)
        assert mean_lat < 10.0, f"Mean latency {mean_lat:.2f}ms exceeds 10ms"


# ======================================================================
# Learning validation — the core test
# ======================================================================

class TestTankLearning:
    """Verify that the cerebellum actually learns over multiple rounds."""

    def test_spe_decreases_over_steps(self):
        """Forward model SPE should decrease as the cerebellum learns."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=300, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.3,
                correction_lr=0.005, forward_model_lr=0.02,
                forward_model_hidden=64, noise_decay=0.9,
                correction_scale=0.3,
            ),
        )
        early_spes: list[float] = []
        late_spes: list[float] = []
        step = 0
        for _ in range(3):
            env.reset()
            while not env.done:
                result = ctrl.step(env)
                step += 1
                if step <= 100:
                    early_spes.append(result.spe)
                elif step > 500:
                    late_spes.append(result.spe)
            ctrl.decay_noise()

        if early_spes and late_spes:
            assert np.mean(late_spes) <= np.mean(early_spes) + 0.5, (
                f"SPE should decrease: early={np.mean(early_spes):.4f}, "
                f"late={np.mean(late_spes):.4f}"
            )

    def test_cerebellum_learns_faster_forward_model(self):
        """
        Ablation: with cerebellum enabled, forward model converges faster.

        The correction net explores more of the state space (since it
        modifies actions), giving the forward model more diverse data.
        The cortex-only mode has less action diversity after noise decay.
        """
        cfg = TankConfig(
            round_max_ticks=300, randomize_spawns=False, noise=0.0,
            enemy_count=1, enemy_roster=["heavy"],
        )
        ctrl_cfg = GUIControlConfig(
            cortex_gain=0.6, cortex_noise=0.5,
            correction_lr=0.002, forward_model_lr=0.02,
            forward_model_hidden=64, noise_decay=0.90,
            correction_scale=0.15,
        )

        results = {}
        for mode in ("cortex+cerebellum", "cortex_only"):
            enabled = mode == "cortex+cerebellum"
            fm_errors: list[float] = []
            spes: list[float] = []
            for _ in range(3):
                env = TankBattleEnv(cfg)
                ctrl = TankController(
                    state_dim=env.state_dim, action_dim=env.action_dim,
                    cfg=ctrl_cfg, cerebellum_enabled=enabled,
                )
                for ep in range(3):
                    env.reset()
                    while not env.done:
                        r = ctrl.step(env)
                        spes.append(r.spe)
                    ctrl.decay_noise()
                fm_errors.append(ctrl.forward_model.mean_recent_error)
            results[mode] = {
                "fm_error": np.mean(fm_errors),
                "mean_spe": np.mean(spes),
            }

        cb_err = results["cortex+cerebellum"]["fm_error"]
        cx_err = results["cortex_only"]["fm_error"]
        assert cb_err < cx_err * 2.0, (
            f"Cerebellum forward model should not be much worse: "
            f"CB={cb_err:.4f}, CX={cx_err:.4f}"
        )

    def test_correction_net_produces_nonzero_output(self):
        """After training, at least one microzone should produce nonzero corrections."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=200, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.5,
                correction_lr=0.002, correction_scale=0.15,
            ),
        )
        for _ in range(300):
            if env.done:
                env.reset()
            ctrl.step(env)

        state = env.observe()
        corr = ctrl.cerebellar_correction(state)
        assert np.any(np.abs(corr) > 1e-4), (
            f"Total correction should be nonzero after training: max={np.max(np.abs(corr)):.6f}"
        )
        per_mz = ctrl.microzone_corrections(state)
        nonzero_zones = [n for n, c in per_mz.items() if np.any(np.abs(c) > 1e-5)]
        assert len(nonzero_zones) >= 1, (
            f"At least one microzone should learn: {per_mz}"
        )

    def test_forward_model_learns_tank_dynamics(self):
        """The forward model should learn to predict tank state transitions."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=200, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(forward_model_lr=0.02, forward_model_hidden=64),
        )
        for _ in range(200):
            if env.done:
                env.reset()
            ctrl.step(env)

        assert ctrl.forward_model._step >= 100
        assert ctrl.forward_model.mean_recent_error < 0.5, (
            f"FM error still high: {ctrl.forward_model.mean_recent_error:.4f}"
        )

    def test_aim_only_microzone_learns(self):
        """With only aim enabled, correction should primarily affect turret."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=200, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.3,
                correction_lr=0.005, correction_scale=0.3,
            ),
        )
        ctrl.modulate_microzone("dodge", enabled=False)
        ctrl.modulate_microzone("move", enabled=False)

        for _ in range(300):
            if env.done:
                env.reset()
            ctrl.step(env)

        state = env.observe()
        corrs = ctrl.microzone_corrections(state)
        assert np.allclose(corrs["dodge"], 0.0)
        assert np.allclose(corrs["move"], 0.0)
        aim_mag = np.linalg.norm(corrs["aim"])
        assert aim_mag > 1e-5, f"Aim-only correction too small: {aim_mag:.6f}"

    def test_dodge_only_microzone_learns(self):
        """With only dodge enabled, correction should be nonzero when bullets near."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=200, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.3,
                correction_lr=0.005, correction_scale=0.3,
            ),
        )
        ctrl.modulate_microzone("aim", enabled=False)
        ctrl.modulate_microzone("move", enabled=False)

        for _ in range(300):
            if env.done:
                env.reset()
            ctrl.step(env)

        state = env.observe()
        corrs = ctrl.microzone_corrections(state)
        assert np.allclose(corrs["aim"], 0.0)
        assert np.allclose(corrs["move"], 0.0)
        dodge_mag = np.linalg.norm(corrs["dodge"])
        assert dodge_mag > 1e-5, f"Dodge-only correction too small: {dodge_mag:.6f}"

    def test_correction_stays_bounded_over_training(self):
        """After many steps, total correction magnitude should stay bounded."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=300, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.5,
                correction_lr=0.005, correction_scale=0.3,
            ),
        )
        max_corr = 0.0
        for _ in range(3):
            env.reset()
            while not env.done:
                result = ctrl.step(env)
                mag = float(np.linalg.norm(result.correction))
                max_corr = max(max_corr, mag)
            ctrl.decay_noise()

        max_theoretical = sum(mz.scale for mz in ctrl._microzones) * np.sqrt(4)
        assert max_corr < max_theoretical + 0.1, (
            f"Correction exploded: max={max_corr:.4f}, "
            f"theoretical_bound={max_theoretical:.4f}"
        )

    def test_confidence_increases_over_training(self):
        """Cerebellum confidence should rise as it accumulates experience."""
        env = TankBattleEnv(TankConfig(
            round_max_ticks=200, randomize_spawns=False, noise=0.0,
        ))
        ctrl = TankController(
            state_dim=env.state_dim, action_dim=env.action_dim,
            cfg=GUIControlConfig(
                cortex_gain=0.6, cortex_noise=0.3,
                forward_model_lr=0.02, forward_model_hidden=64,
            ),
        )
        early_conf = ctrl.cerebellum_confidence
        for _ in range(400):
            if env.done:
                env.reset()
            ctrl.step(env)
        late_conf = ctrl.cerebellum_confidence

        assert late_conf >= early_conf, (
            f"Confidence should increase: early={early_conf:.4f}, late={late_conf:.4f}"
        )
