"""
GUIController — cerebellar motor control via Sensory Prediction Error.

Biological model (Wolpert, Miall & Kawato 1998; Bastian 2006):
  The cerebellum is NOT a reward-maximizer (that's basal ganglia / RL).
  The cerebellum is a PREDICTION-ERROR-MINIMIZER:

  1. Motor cortex sends a crude movement intention
  2. Cerebellar forward model PREDICTS the sensory consequences
  3. After execution, the ACTUAL sensory feedback is compared with prediction
  4. Sensory Prediction Error (SPE) = actual - predicted
  5. SPE (climbing fiber) drives correction learning via the forward model gradient

  This is fundamentally different from RL:
    RL:         max Σ reward      (externally defined objective)
    Cerebellum: min ||SPE||²      (self-supervised prediction accuracy)

  No reward function needed. The forward model IS the cerebellum.
  The correction IS derived from the forward model gradient.
  The cortex defines WHAT to do; the cerebellum learns HOW precisely.

Architecture (multi-microzone):
  cortex_signal(state) → crude action               # "move right-ish"
  microzone_k.net(state) → correction_k for each k  # parallel learned corrections
  action = cortex + Σ correction_k
  → forward model predicts next state
  → actual next state observed
  → per-microzone climbing fiber error drives each correction_k independently
  → each microzone learns to minimize its own sub-objective

  Biological basis for microzones:
    The cerebellar cortex is divided into parasagittal microzones, each receiving
    a distinct climbing fiber signal from the inferior olive.  Each microzone
    controls a different motor component (e.g., one for saccade accuracy, another
    for head stabilization).  They share the same mossy fiber input (state) but
    learn independently.  Their outputs are summed at the deep cerebellar nuclei.

    Single-objective tasks (aim trainer, GUI click) use one microzone.
    Multi-objective tasks (tank: aim + dodge + move) use parallel microzones,
    each with its own error signal — exactly as in biology.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from digital_cerebellum.core.forward_model import ForwardModel
from digital_cerebellum.micro_ops.engine import Environment


@dataclass
class GUIControlConfig:
    cortex_gain: float = 1.5
    cortex_noise: float = 0.3
    correction_lr: float = 0.01
    correction_hidden: int = 64
    forward_model_lr: float = 0.02
    forward_model_hidden: int = 128
    noise_decay: float = 0.97
    correction_scale: float = 0.5
    alignment_weight: float = 0.3


@dataclass
class TrialResult:
    """Result of one control step."""
    step: int
    cortex_action: np.ndarray
    correction: np.ndarray
    final_action: np.ndarray
    reward: float
    spe: float
    forward_error: float
    latency_ms: float


class CorrectionMicrozone:
    """
    A single cerebellar microzone — independent correction circuit.

    Each microzone has its own population of Purkinje cells (correction net),
    its own climbing fiber input (error signal), and learns independently.
    Multiple microzones operate in parallel on the same mossy fiber input
    (state) and their outputs are summed at the deep cerebellar nuclei.

    Output is bounded via tanh (biological: Purkinje cell firing rate
    saturation) to prevent correction explosion during early training.
    """
    __slots__ = ('name', 'net', 'optimizer', 'scale', 'enabled')

    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.01,
        scale: float = 0.5,
        enabled: bool = True,
    ):
        self.name = name
        self.scale = scale
        self.enabled = enabled
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh(),
        )
        with torch.no_grad():
            self.net[-2].weight.mul_(0.01)
            self.net[-2].bias.zero_()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=1e-4,
        )


class GUIController:
    """
    Cerebellar controller — SPE-driven, task-agnostic.

    The SAME learning mechanism works for any environment:
    aim trainer, tank battle, robotic arm, GUI control.

    Subclasses override ONLY:
      - cortex_signal(state)        — what the cortex wants (mossy fiber)
      - cortex_error_signal(s_t)    — single climbing fiber (simple tasks)
      - cortex_error_signals(s_t)   — per-microzone climbing fibers (multi-obj)
      - _build_microzones()         — define parallel correction circuits

    Everything else — forward model, microzone training, SPE tracking,
    confidence, adaptive cortex consultation — is the universal
    cerebellar circuit, identical across all tasks.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: GUIControlConfig | None = None,
        cerebellum_enabled: bool = True,
    ):
        self.cfg = cfg or GUIControlConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cerebellum_enabled = cerebellum_enabled

        self.forward_model = ForwardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.forward_model_hidden,
            lr=self.cfg.forward_model_lr,
        )

        self._microzones = self._build_microzones()

        self._step = 0
        self._noise_scale = self.cfg.cortex_noise
        self._prev_state: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._history: list[dict] = []
        self._recent_spes: list[float] = []

    def _build_microzones(self) -> list[CorrectionMicrozone]:
        """Factory for correction microzones.

        Default: single microzone driven by cortex_error_signal().
        Override in subclasses to create parallel microzones for
        multi-objective tasks (e.g., aim + dodge + move).
        """
        return [CorrectionMicrozone(
            "default",
            self.state_dim,
            self.action_dim,
            hidden_dim=self.cfg.correction_hidden,
            lr=self.cfg.correction_lr,
            scale=self.cfg.correction_scale,
        )]

    def cortex_signal(self, state: np.ndarray) -> np.ndarray:
        """
        Generate crude motor intention from state.

        For aim trainer: state[4]=dx, state[5]=dy (direction to target).
        The cortex simply says "move in that direction" with noise.
        Subclasses override for different environments.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)

        if len(state) >= 6:
            action[0] = state[4] * self.cfg.cortex_gain
            action[1] = state[5] * self.cfg.cortex_gain

        action += np.random.randn(self.action_dim).astype(np.float32) * self._noise_scale

        if len(state) >= 7:
            dist = state[6]
            if dist < 0.05 and self.action_dim >= 3:
                action[2] = 0.5

        return np.clip(action, -1.0, 1.0)

    def cerebellar_correction(self, state: np.ndarray) -> np.ndarray:
        """Sum corrections from all enabled microzones (deep cerebellar nuclei)."""
        if not self.cerebellum_enabled:
            return np.zeros(self.action_dim, dtype=np.float32)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0)
            total = torch.zeros(1, self.action_dim)
            for mz in self._microzones:
                if mz.enabled:
                    total += mz.net(s) * mz.scale
            return total.squeeze(0).numpy()

    def step(self, env: Environment) -> TrialResult:
        """
        Execute one control step with cerebellar refinement.

        The step mirrors the biological cerebellar loop:
        1. Observe state (mossy fiber input)
        2. Cortex generates crude intention (motor cortex)
        3. Correction net refines it (Purkinje cell output)
        4. Forward model predicts sensory consequence (internal model)
        5. Action executed → actual sensory feedback
        6. SPE = actual - predicted (climbing fiber from inferior olive)
        7. SPE gradient through frozen FM updates correction (LTD at PF→PC)
        8. FM also learns from SPE (granule cell layer adaptation)
        """
        t0 = time.perf_counter()
        self._step += 1

        state = env.observe()

        forward_error = 0.0
        if self._prev_state is not None and self._prev_action is not None:
            forward_error = self.forward_model.learn(
                self._prev_state, self._prev_action, state,
            )

        cortex = self.cortex_signal(state)
        correction = self.cerebellar_correction(state)
        final_action = np.clip(cortex + correction, -1.0, 1.0)

        prediction = self.forward_model.predict(state, final_action)
        predicted_next = prediction.predicted_next_state

        reward = env.execute(final_action)

        actual_next = env.observe()
        spe_vec = self.forward_model.compute_spe(predicted_next, actual_next)
        spe = float(np.linalg.norm(spe_vec))

        self._learn_correction(state, cortex, actual_next, spe)

        self._prev_state = state.copy()
        self._prev_action = final_action.copy()

        latency = (time.perf_counter() - t0) * 1000

        result = TrialResult(
            step=self._step,
            cortex_action=cortex,
            correction=correction,
            final_action=final_action,
            reward=reward,
            spe=spe,
            forward_error=forward_error,
            latency_ms=latency,
        )

        self._history.append({
            "step": self._step,
            "reward": reward,
            "spe": spe,
            "correction_mag": float(np.linalg.norm(correction)),
            "latency_ms": latency,
        })

        from digital_cerebellum.viz.event_bus import event_bus as _eb
        _eb.emit("step", "GUIController", step=self._step,
                 reward=float(reward), spe=float(spe),
                 correction_mag=float(np.linalg.norm(correction)),
                 latency_ms=round(latency, 2),
                 confidence=float(prediction.confidence))
        if spe > 0.5:
            _eb.emit("error", "GUIController", spe=float(spe), step=self._step)

        return result

    def cortex_error_signal(self, state_t: torch.Tensor) -> torch.Tensor:
        """
        Differentiable error signal — the "retinal slip" equivalent.

        Returns a tensor whose magnitude = how far from the cortex's goal.
        When this is zero, the task is accomplished.

        For aim trainer: [dx_to_target, dy_to_target]
        Override in subclasses for different environments.
        """
        if state_t.shape[-1] >= 7:
            return state_t[..., 4:6] * self.cfg.cortex_gain
        return state_t[..., :min(self.action_dim, state_t.shape[-1])] * 0.1

    def cortex_error_signals(
        self, state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-microzone climbing fiber errors.

        Default wraps cortex_error_signal() for the single "default" microzone.
        Override in subclasses that define multiple microzones to return
        a dict mapping microzone name → differentiable error tensor.
        """
        return {"default": self.cortex_error_signal(state_t)}

    def _learn_correction(
        self,
        state: np.ndarray,
        cortex: np.ndarray,
        actual_next: np.ndarray,
        spe_scalar: float,
    ):
        """
        Train each microzone independently via its own climbing fiber error.

        For each microzone k:
          1. Freeze all OTHER microzones and the forward model
          2. Compute action = tanh(cortex + mz_k(state) + Σ_{j≠k} detach(mz_j(state)))
          3. Predict next state through the frozen FM
          4. Compute microzone k's error from the predicted next state
          5. Backprop only through mz_k's correction net

        This mirrors the biological independence of microzones: each has its own
        Purkinje cell population and climbing fiber, learning in parallel without
        interfering with other microzones' synaptic weights.
        """
        self._recent_spes.append(spe_scalar)
        if len(self._recent_spes) > 100:
            self._recent_spes.pop(0)

        if not self.cerebellum_enabled:
            return

        s_t = torch.from_numpy(state).float().unsqueeze(0)
        c_t = torch.from_numpy(cortex).float().unsqueeze(0)

        for p in self.forward_model.parameters():
            p.requires_grad_(False)

        modulation = 1.0 + min(spe_scalar, 3.0)

        for mz in self._microzones:
            if not mz.enabled:
                continue

            correction = mz.net(s_t) * mz.scale

            other_corr = sum(
                (omz.net(s_t) * omz.scale).detach()
                for omz in self._microzones
                if omz is not mz and omz.enabled
            )
            if not isinstance(other_corr, torch.Tensor):
                other_corr = torch.zeros_like(correction)

            action = torch.tanh(c_t + correction + other_corr)
            sa = torch.cat([s_t, action], dim=-1)
            delta = self.forward_model.net(sa)
            predicted_next = s_t + delta if self.forward_model._residual_mode else delta

            errors = self.cortex_error_signals(predicted_next)
            if mz.name not in errors:
                continue

            future_error = errors[mz.name]
            error_loss = future_error.pow(2).sum()
            reg = correction.pow(2).mean() * 0.02
            loss = (error_loss + reg) * modulation

            mz.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mz.net.parameters(), 1.0)
            mz.optimizer.step()

        for p in self.forward_model.parameters():
            p.requires_grad_(True)

    def decay_noise(self):
        """Call once per episode to reduce exploration noise."""
        self._noise_scale *= self.cfg.noise_decay
        self._noise_scale = max(self._noise_scale, 0.02)

    def run_episode(
        self,
        env: Environment,
        n_steps: int = 500,
    ) -> dict[str, Any]:
        """Run one episode and return summary metrics."""
        rewards = []
        spes = []
        latencies = []
        corrections = []

        for _ in range(n_steps):
            result = self.step(env)
            rewards.append(result.reward)
            spes.append(result.spe)
            latencies.append(result.latency_ms)
            corrections.append(float(np.linalg.norm(result.correction)))

        self.decay_noise()

        rewards_arr = np.array(rewards)
        spes_arr = np.array(spes)

        half = len(rewards) // 2
        return {
            "total_reward": round(float(rewards_arr.sum()), 2),
            "mean_reward": round(float(rewards_arr.mean()), 4),
            "mean_spe": round(float(spes_arr.mean()), 4),
            "spe_first_half": round(float(spes_arr[:half].mean()), 4),
            "spe_second_half": round(float(spes_arr[half:].mean()), 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "mean_correction_mag": round(float(np.mean(corrections)), 4),
            "noise_scale": round(self._noise_scale, 4),
        }

    # ── General cerebellar metrics (task-agnostic) ──────────────────

    @property
    def mean_recent_spe(self) -> float:
        """Average SPE over the last 100 steps."""
        if not self._recent_spes:
            return float("inf")
        return sum(self._recent_spes) / len(self._recent_spes)

    @property
    def cerebellum_confidence(self) -> float:
        """0-1 score: how well the cerebellum has learned the current task."""
        spe = self.mean_recent_spe
        fm_err = self.forward_model.mean_recent_error
        spe_conf = 1.0 / (1.0 + spe * 2.0)
        fm_conf = 1.0 / (1.0 + fm_err * 10.0)
        return min(spe_conf, fm_conf)

    def should_call_cortex(self, base_interval: int = 90) -> int:
        """Adaptive cortex consultation interval based on confidence.

        High confidence → longer interval → less cortex (LLM) dependency.
        This is the digital equivalent of motor skill automation:
        well-learned tasks need less conscious (cortex) involvement.
        """
        conf = self.cerebellum_confidence
        multiplier = 1.0 + conf * 4.0
        return int(base_interval * multiplier)

    def modulate_microzone(
        self,
        name: str,
        gain: float | None = None,
        lr: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Cortex-driven gain modulation (cortico-rubro-olivary pathway).

        Allows the cortex (LLM) to adjust microzone priorities in real-time.
        Biology: motor cortex -> red nucleus -> inferior olive -> climbing
        fiber gain.  Higher gain = stronger error signal = faster learning
        + larger correction for that sub-objective.
        """
        for mz in self._microzones:
            if mz.name == name:
                if gain is not None:
                    mz.scale = gain
                if lr is not None:
                    for pg in mz.optimizer.param_groups:
                        pg['lr'] = lr
                if enabled is not None:
                    mz.enabled = enabled
                return

    @property
    def microzone_names(self) -> list[str]:
        return [mz.name for mz in self._microzones]

    def microzone_corrections(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Per-microzone correction vectors (for diagnostics)."""
        if not self.cerebellum_enabled:
            return {mz.name: np.zeros(self.action_dim, dtype=np.float32)
                    for mz in self._microzones}
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0)
            return {
                mz.name: (mz.net(s).squeeze(0) * mz.scale).numpy()
                if mz.enabled else np.zeros(self.action_dim, dtype=np.float32)
                for mz in self._microzones
            }

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "noise_scale": round(self._noise_scale, 4),
            "cerebellum_enabled": self.cerebellum_enabled,
            "cerebellum_confidence": round(self.cerebellum_confidence, 4),
            "mean_recent_spe": round(self.mean_recent_spe, 6)
                if self._recent_spes else None,
            "microzones": self.microzone_names,
            "forward_model": self.forward_model.stats,
        }
