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

Architecture:
  cortex_signal(state) → crude action          # "move right-ish"
  correction_net(state) → learned correction   # "adjust -3px, +1px"
  action = cortex + correction
  → forward model predicts next state
  → actual next state observed
  → SPE gradient flows THROUGH frozen FM BACK to correction_net
  → correction_net learns to reduce SPE
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


class GUIController:
    """
    Cerebellar controller — SPE-driven, task-agnostic.

    The SAME learning mechanism works for any environment:
    aim trainer, tank battle, robotic arm, GUI control.
    Only cortex_signal() needs task-specific override.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: GUIControlConfig | None = None,
    ):
        self.cfg = cfg or GUIControlConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.forward_model = ForwardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.forward_model_hidden,
            lr=self.cfg.forward_model_lr,
        )

        self._correction_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, self.cfg.correction_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.cfg.correction_hidden, self.cfg.correction_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.cfg.correction_hidden, action_dim),
        )
        with torch.no_grad():
            self._correction_net[-1].weight.mul_(0.01)
            self._correction_net[-1].bias.zero_()
        self._correction_opt = torch.optim.Adam(
            self._correction_net.parameters(), lr=self.cfg.correction_lr,
        )

        self._step = 0
        self._noise_scale = self.cfg.cortex_noise
        self._prev_state: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._history: list[dict] = []

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
        """Generate learned correction from the cerebellar network."""
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0)
            correction = self._correction_net(s).squeeze(0).numpy()
        return correction * self.cfg.correction_scale

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

    def _learn_correction(
        self,
        state: np.ndarray,
        cortex: np.ndarray,
        actual_next: np.ndarray,
        spe_scalar: float,
    ):
        """
        Train correction via FUTURE ERROR MINIMIZATION through the FM.

        Biological model (retinal slip / motor error):
          1. Cortex says "aim at target" (crude command)
          2. Cerebellum adds correction, action is executed
          3. Forward model predicts the NEXT state
          4. At the predicted next state, we compute the RESIDUAL ERROR
             (how far from target we'll still be)
          5. Correction learns to MINIMIZE this future error

          This is exactly "retinal slip" in eye movements:
            - Move eyes toward target
            - After saccade, retinal slip = target offset on retina
            - Cerebellum adjusts so next saccade lands closer

        Why this enables precise aiming:
          - future_error includes turret angle offset, distance to target
          - Gradient flows: correction_net → FM → predicted_next → error
          - The gradient tells correction: "if you increased turret action
            by 0.1, the predicted turret error would decrease by 0.05"
          - Over time, correction learns the exact mapping

        Not reward-based: the error signal is the SENSORY state itself
        (observable, not externally defined). SPE modulates learning rate.
        """
        s_t = torch.from_numpy(state).float().unsqueeze(0)
        c_t = torch.from_numpy(cortex).float().unsqueeze(0)

        for p in self.forward_model.parameters():
            p.requires_grad_(False)

        correction = self._correction_net(s_t)
        action = torch.tanh(c_t + correction * self.cfg.correction_scale)

        sa = torch.cat([s_t, action], dim=-1)
        delta = self.forward_model.net(sa)
        predicted_next = s_t + delta if self.forward_model._residual_mode else delta

        future_error = self.cortex_error_signal(predicted_next)
        error_loss = future_error.pow(2).sum()

        reg = correction.pow(2).mean() * 0.005

        modulation = 1.0 + min(spe_scalar, 3.0)
        loss = (error_loss + reg) * modulation

        self._correction_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._correction_net.parameters(), 1.0)
        self._correction_opt.step()

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

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "noise_scale": round(self._noise_scale, 4),
            "forward_model": self.forward_model.stats,
        }
