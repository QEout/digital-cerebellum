"""
Micro-Operation Engine — continuous real-time control via the cerebellum.

This is the core of Phase 6.  Instead of the request-response pattern
(evaluate → result), the micro-op engine runs a continuous loop:

    observe → predict → act → compare → learn → repeat (60Hz+)

Biological basis:
  The cerebellum runs continuously, not on-demand.  It constantly
  receives sensory input, maintains a forward model of the body's
  state, generates corrective motor signals, and updates its model
  from prediction errors — all at ~200Hz in biology.

  The three key principles (Tsay & Ivry 2025):
    1. PREDICTION — forward model predicts sensory consequences
    2. TIMESCALE — operates at millisecond precision
    3. CONTINUITY — never stops, always refining

Architecture::

    Environment
        │ state (60Hz)
        ▼
    ┌─ MicroOpEngine ──────────────────────────────────┐
    │                                                    │
    │  StateEncoder  →  encode state as vector            │
    │       │                                            │
    │  PatternSeparator  →  RFF sparse code              │
    │       │                                            │
    │  PredictionEngine  →  K-head action prediction     │
    │       │                                            │
    │  ForwardModel  →  predict next state               │
    │       │                                            │
    │  ActionEncoder  →  decode action vector             │
    │       │                                            │
    │  SkillStore  →  check for learned action sequences  │
    │       │                                            │
    │  Learn from (predicted vs actual) SPE              │
    │                                                    │
    └────────────────────────────────────────────────────┘
        │ action
        ▼
    Environment.execute(action)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from digital_cerebellum.core.state_encoder import StateEncoder
from digital_cerebellum.core.forward_model import ForwardModel
from digital_cerebellum.core.action_encoder import ActionEncoder
from digital_cerebellum.core.pattern_separator import PatternSeparator

log = logging.getLogger(__name__)


# ======================================================================
# Environment protocol
# ======================================================================

class Environment(Protocol):
    """Interface for any controllable environment."""

    def observe(self) -> np.ndarray:
        """Return current state vector."""
        ...

    def execute(self, action: np.ndarray) -> float:
        """Execute action, return reward."""
        ...

    @property
    def state_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...


# ======================================================================
# Step result
# ======================================================================

@dataclass
class StepResult:
    """Output of one micro-operation step."""
    step: int
    state: np.ndarray
    action: np.ndarray
    reward: float
    predicted_next_state: np.ndarray | None = None
    actual_next_state: np.ndarray | None = None
    spe: float = 0.0
    forward_model_error: float = 0.0
    latency_ms: float = 0.0
    used_skill: bool = False


# ======================================================================
# Micro-Operation Engine
# ======================================================================

@dataclass
class MicroOpConfig:
    """Configuration for the micro-operation engine."""
    rff_dim: int = 2048
    rff_gamma: float = 1.0
    rff_sparsity: float = 0.1
    hidden_dim: int = 128
    forward_model_lr: float = 0.01
    action_lr: float = 0.005
    target_hz: float = 60.0
    max_steps: int = 10000


class MicroOpEngine:
    """
    Continuous cerebellar controller for real-time micro-operations.

    Unlike DigitalCerebellum (request-response), this engine runs
    a tight loop: observe → predict → act → learn, at target_hz.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: MicroOpConfig | None = None,
    ):
        self.cfg = cfg or MicroOpConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        encoder_target_dim = state_dim + 4
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            target_dim=encoder_target_dim,
            mode="direct",
        )

        self.pattern_separator = PatternSeparator(
            input_dim=encoder_target_dim,
            rff_dim=self.cfg.rff_dim,
            gamma=self.cfg.rff_gamma,
            sparsity=self.cfg.rff_sparsity,
        )

        self.forward_model = ForwardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.hidden_dim,
            lr=self.cfg.forward_model_lr,
        )

        self.action_encoder = ActionEncoder(action_dim=action_dim)

        self._action_net = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.rff_dim, self.cfg.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.cfg.hidden_dim, action_dim),
            torch.nn.Tanh(),
        )
        self._action_optimizer = torch.optim.Adam(
            self._action_net.parameters(), lr=self.cfg.action_lr,
        )

        self._step = 0
        self._total_reward = 0.0
        self._history: list[dict] = []
        self._prev_state: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None

    # ==================================================================
    # Single step
    # ==================================================================

    def step(self, env: Environment) -> StepResult:
        """
        Execute one control step.

        1. Observe current state
        2. Encode state → RFF sparse code
        3. Generate action from learned policy
        4. Predict next state via forward model
        5. Execute action in environment
        6. Compare prediction vs reality → learn
        """
        t0 = time.perf_counter()
        self._step += 1

        state = env.observe()

        # Learn from previous step's prediction
        forward_error = 0.0
        if self._prev_state is not None and self._prev_action is not None:
            forward_error = self.forward_model.learn(
                self._prev_state, self._prev_action, state,
            )

        encoded = self.state_encoder.encode(state)
        z = self.pattern_separator.encode_event(encoded)

        with torch.no_grad():
            z_t = torch.from_numpy(z).float().unsqueeze(0)
            action_raw = self._action_net(z_t).squeeze(0).numpy()

        prediction = self.forward_model.predict(state, action_raw)
        predicted_next = prediction.predicted_next_state

        reward = env.execute(action_raw)
        self._total_reward += reward

        actual_next = env.observe()
        spe_vec = self.forward_model.compute_spe(predicted_next, actual_next)
        spe_mag = float(np.linalg.norm(spe_vec))

        self._learn_action(z, reward, spe_mag)

        self._prev_state = state.copy()
        self._prev_action = action_raw.copy()

        latency = (time.perf_counter() - t0) * 1000

        result = StepResult(
            step=self._step,
            state=state,
            action=action_raw,
            reward=reward,
            predicted_next_state=predicted_next,
            actual_next_state=actual_next,
            spe=spe_mag,
            forward_model_error=forward_error,
            latency_ms=latency,
        )

        self._history.append({
            "step": self._step,
            "reward": reward,
            "spe": spe_mag,
            "forward_error": forward_error,
            "latency_ms": latency,
        })

        return result

    def _learn_action(self, z: np.ndarray, reward: float, spe: float):
        """
        Update the action policy using reward signal.

        Simple policy gradient: action that got positive reward → reinforce.
        SPE modulates learning rate: high prediction error → learn faster.
        """
        z_t = torch.from_numpy(z).float().unsqueeze(0)
        action = self._action_net(z_t)

        reward_signal = torch.tensor([[reward]], dtype=torch.float32)
        modulation = 1.0 + min(spe, 2.0)
        loss = -(action * reward_signal * modulation).mean()

        self._action_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._action_net.parameters(), 1.0)
        self._action_optimizer.step()

    # ==================================================================
    # Run loop
    # ==================================================================

    def run(
        self,
        env: Environment,
        n_steps: int | None = None,
        target_hz: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the continuous control loop.

        Parameters
        ----------
        env : Environment to control
        n_steps : number of steps (default: cfg.max_steps)
        target_hz : target frequency (default: cfg.target_hz)

        Returns
        -------
        Summary dict with performance metrics.
        """
        n = n_steps or self.cfg.max_steps
        hz = target_hz or self.cfg.target_hz
        frame_time = 1.0 / hz

        t_start = time.perf_counter()
        rewards = []
        spes = []
        latencies = []

        for i in range(n):
            t_frame = time.perf_counter()

            result = self.step(env)
            rewards.append(result.reward)
            spes.append(result.spe)
            latencies.append(result.latency_ms)

            elapsed = time.perf_counter() - t_frame
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_time = time.perf_counter() - t_start
        actual_hz = n / total_time if total_time > 0 else 0

        rewards_arr = np.array(rewards)
        spes_arr = np.array(spes)
        latencies_arr = np.array(latencies)

        half = len(rewards) // 2
        early_reward = float(rewards_arr[:half].mean()) if half > 0 else 0
        late_reward = float(rewards_arr[half:].mean()) if half > 0 else 0
        early_spe = float(spes_arr[:half].mean()) if half > 0 else 0
        late_spe = float(spes_arr[half:].mean()) if half > 0 else 0

        return {
            "total_steps": n,
            "total_time_s": round(total_time, 2),
            "actual_hz": round(actual_hz, 1),
            "total_reward": round(float(rewards_arr.sum()), 4),
            "mean_reward": round(float(rewards_arr.mean()), 4),
            "mean_spe": round(float(spes_arr.mean()), 4),
            "mean_latency_ms": round(float(latencies_arr.mean()), 2),
            "max_latency_ms": round(float(latencies_arr.max()), 2),
            "improvement": {
                "early_reward": round(early_reward, 4),
                "late_reward": round(late_reward, 4),
                "reward_improvement": round(late_reward - early_reward, 4),
                "early_spe": round(early_spe, 4),
                "late_spe": round(late_spe, 4),
                "spe_reduction": round(early_spe - late_spe, 4),
            },
            "forward_model": self.forward_model.stats,
        }

    # ==================================================================
    # Diagnostics
    # ==================================================================

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "total_reward": round(self._total_reward, 4),
            "forward_model": self.forward_model.stats,
            "state_encoder": self.state_encoder.stats,
            "action_encoder": self.action_encoder.stats,
        }
