"""
Step Forward Model — predicts outcomes of agent steps.

Unlike the MicroOpEngine's ForwardModel (which works at 285Hz with
raw numeric vectors), this model works at the step level with
text-encoded embeddings.  It predicts: given the current state and
an intended action, what should the outcome embedding look like?

The architecture is deliberately shallow (2-layer MLP) to match
the biological constraint that cerebellar circuits are feedforward.

This model is universal: it doesn't know whether the agent is
clicking a button, executing a shell command, or moving a game
character.  It only sees embedding vectors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class StepForwardModel(nn.Module):
    """
    Predict outcome embedding from (state_embedding, action_embedding).

    Input:  concat(state_emb, action_emb)  →  dim = 2 * embedding_dim
    Output: predicted_outcome_emb           →  dim = embedding_dim

    Uses residual prediction: outcome ≈ state + delta, since outcomes
    are often similar to the prior state (small changes per step).
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        lr: float = 0.005,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

        self._step = 0
        self._cumulative_loss = 0.0
        self._recent_errors: list[float] = []
        self._max_recent = 100

    def predict(
        self,
        state_emb: np.ndarray,
        action_emb: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Predict the outcome embedding.

        Returns (predicted_outcome, confidence).
        """
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state_emb, dtype=np.float32))
            a = torch.from_numpy(np.asarray(action_emb, dtype=np.float32))
            sa = torch.cat([s, a]).unsqueeze(0)

            delta = self.net(sa)
            pred = s.unsqueeze(0) + delta

            pred_np = pred.squeeze(0).numpy()
            confidence = 1.0 / (1.0 + float(delta.abs().mean()))

        return pred_np, confidence

    def learn(
        self,
        state_emb: np.ndarray,
        action_emb: np.ndarray,
        actual_outcome_emb: np.ndarray,
    ) -> float:
        """
        Learn from one observed transition.

        Returns the prediction error (MSE loss).
        """
        s = torch.from_numpy(np.asarray(state_emb, dtype=np.float32)).unsqueeze(0)
        a = torch.from_numpy(np.asarray(action_emb, dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(
            np.asarray(actual_outcome_emb, dtype=np.float32)
        ).unsqueeze(0)

        sa = torch.cat([s, a], dim=-1)
        delta = self.net(sa)
        pred = s + delta

        loss = self._loss_fn(pred, target)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        error = float(loss.item())
        self._step += 1
        self._cumulative_loss += error
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._max_recent:
            self._recent_errors.pop(0)

        return error

    @staticmethod
    def compute_spe(
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Compute Sensory Prediction Error magnitude."""
        diff = actual - predicted
        return float(np.linalg.norm(diff))

    @property
    def mean_recent_error(self) -> float:
        if not self._recent_errors:
            return 0.0
        return sum(self._recent_errors) / len(self._recent_errors)

    @property
    def is_improving(self) -> bool:
        if len(self._recent_errors) < 10:
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
        }
