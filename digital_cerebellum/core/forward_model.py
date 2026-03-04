"""
Forward Model — the cerebellum's core computational principle.

Biological basis:
  The cerebellum's primary function is maintaining an internal forward
  model: given the current state and a planned action, predict what
  the sensory consequences will be.  When the prediction doesn't match
  reality, the climbing fibre error signal drives learning.

  This is why the cerebellum is essential for motor control:
  - It predicts "if I move my arm this way, I'll feel X"
  - If the actual sensation differs → error → adjust the model
  - Over time, predictions become accurate → movements become smooth

  References:
    - Wolpert, Miall & Kawato (1998): "Internal models in the cerebellum"
    - Bastian (2006): "Learning to predict the future"
    - Tsay & Ivry (2025): Prediction, Timescale, Continuity

Digital implementation:
  ForwardModel takes (current_state, action) and predicts next_state.
  The prediction error (predicted - actual) is the Sensory Prediction
  Error (SPE) that drives online learning.

  Architecture: shallow MLP (2 hidden layers) — deliberately simple,
  matching the biological constraint that cerebellar circuits are
  feedforward, not recurrent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ForwardPrediction:
    """Output of the forward model."""
    predicted_next_state: np.ndarray
    confidence: float
    _raw_tensor: torch.Tensor | None = None


class ForwardModel(nn.Module):
    """
    Cerebellar forward model: (state, action) → predicted_next_state.

    Shallow feedforward network (no recurrence — matching biology).
    Learns online from sensory prediction errors.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self._residual_mode = True
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

        self._step = 0
        self._cumulative_loss = 0.0
        self._recent_errors: list[float] = []

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> ForwardPrediction:
        """
        Predict next state given current state and action.

        In residual mode (default), predicts the DELTA:
          next_state = state + model(state, action)
        This makes learning easier (predicting changes, not absolutes).
        """
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            a = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
            sa = torch.cat([s, a], dim=-1)

            delta = self.net(sa)

            if self._residual_mode:
                pred = s + delta
            else:
                pred = delta

            pred_np = pred.squeeze(0).numpy()

            confidence = 1.0 / (1.0 + float(delta.abs().mean()))

        return ForwardPrediction(
            predicted_next_state=pred_np,
            confidence=confidence,
            _raw_tensor=pred,
        )

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        actual_next_state: np.ndarray,
    ) -> float:
        """
        Learn from a single state transition (online learning).

        Returns the prediction error magnitude (SPE).
        """
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        a = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(np.asarray(actual_next_state, dtype=np.float32)).unsqueeze(0)

        sa = torch.cat([s, a], dim=-1)
        delta = self.net(sa)

        if self._residual_mode:
            pred = s + delta
        else:
            pred = delta

        loss = self._loss_fn(pred, target)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        error = float(loss.item())
        self._step += 1
        self._cumulative_loss += error
        self._recent_errors.append(error)
        if len(self._recent_errors) > 100:
            self._recent_errors.pop(0)

        return error

    def compute_spe(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> np.ndarray:
        """Compute Sensory Prediction Error vector."""
        return actual - predicted

    @property
    def mean_recent_error(self) -> float:
        if not self._recent_errors:
            return 0.0
        return sum(self._recent_errors) / len(self._recent_errors)

    @property
    def is_improving(self) -> bool:
        if len(self._recent_errors) < 20:
            return True
        half = len(self._recent_errors) // 2
        old = sum(self._recent_errors[:half]) / half
        new = sum(self._recent_errors[half:]) / (len(self._recent_errors) - half)
        return new < old

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "mean_recent_error": round(self.mean_recent_error, 6),
            "cumulative_loss": round(self._cumulative_loss, 4),
            "is_improving": self.is_improving,
            "residual_mode": self._residual_mode,
        }
