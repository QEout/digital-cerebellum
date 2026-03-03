"""
Online Learner — SGD + Elastic Weight Consolidation.

Drives multi-site plasticity:
  Site 1: prediction engine weights  ← sensory_error   (Phase 0)
  Site 2: router weights             ← reward_error    (Phase 1)
  Site 3: frequency filter params    ← sensory_error   (Phase 2)
  Site 4: Golgi gate params          ← sensory_error   (Phase 2)

Phase 0 implements Site 1 only.

References
----------
- Kirkpatrick et al. 2017 "Overcoming catastrophic forgetting in neural networks"
- tanmay1024/EWC-Implementation (PyTorch)
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn
import numpy as np

from src.core.types import ErrorSignal, ErrorType


class EWCRegularizer:
    """
    Diagonal Fisher-based Elastic Weight Consolidation.

    Maintains a running diagonal Fisher information matrix and a snapshot
    of important parameters.  The regularisation term penalises movement
    away from θ* on dimensions that matter to old tasks.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0):
        self.model = model
        self.ewc_lambda = ewc_lambda

        # θ* — parameter snapshot (updated periodically)
        self._param_snapshot: dict[str, torch.Tensor] = {}
        # F  — diagonal Fisher (running average of squared gradients)
        self._fisher: dict[str, torch.Tensor] = {}
        self._fisher_count = 0

        self._take_snapshot()

    # ------------------------------------------------------------------
    def _take_snapshot(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._param_snapshot[name] = param.data.clone()
                if name not in self._fisher:
                    self._fisher[name] = torch.zeros_like(param.data)

    # ------------------------------------------------------------------
    def update_fisher(self, model: nn.Module):
        """Accumulate Fisher diagonal from the current gradient."""
        self._fisher_count += 1
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Incremental moving average of grad²
                self._fisher[name] += (
                    param.grad.data.pow(2) - self._fisher[name]
                ) / self._fisher_count

    # ------------------------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """EWC regularisation loss: λ/2 · Σ F_i · (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._fisher:
                diff = param - self._param_snapshot[name]
                loss = loss + (self._fisher[name] * diff.pow(2)).sum()
        return (self.ewc_lambda / 2.0) * loss

    # ------------------------------------------------------------------
    def consolidate(self):
        """Snapshot current params as new θ* (call periodically)."""
        self._take_snapshot()


class OnlineLearner:
    """
    Drives online learning for the prediction engine (Site 1, Phase 0).

    Usage::

        learner = OnlineLearner(prediction_engine)

        # after every prediction + actual-outcome pair:
        loss = learner.learn(sparse_z, actual_action_emb, actual_outcome_emb)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        ewc_lambda: float = 400.0,
    ):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.ewc = EWCRegularizer(model, ewc_lambda)
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.step_count = 0

    # ------------------------------------------------------------------
    def learn(
        self,
        z: np.ndarray,
        actual_action_emb: np.ndarray,
        actual_outcome_emb: np.ndarray,
    ) -> float:
        """
        One SGD step on the prediction engine.

        Returns the total loss (task + EWC).
        """
        self.model.train()

        z_t = torch.from_numpy(z).float().unsqueeze(0)
        target_a = torch.from_numpy(actual_action_emb).float().unsqueeze(0)
        target_o = torch.from_numpy(actual_outcome_emb).float().unsqueeze(0)
        ones = torch.ones(1)

        # Forward through all heads
        pred = self.model(z_t)

        # Loss: average cosine loss across all heads
        task_loss = torch.tensor(0.0)
        for head in self.model.heads:
            a, o = head(z_t)
            task_loss = task_loss + self.cos_loss(a, target_a, ones)
            task_loss = task_loss + self.cos_loss(o, target_o, ones)
        task_loss = task_loss / (len(self.model.heads) * 2)

        ewc_loss = self.ewc.penalty()
        total = task_loss + ewc_loss

        self.optimizer.zero_grad()
        total.backward()

        # Update Fisher estimate after backward
        self.ewc.update_fisher(self.model)

        self.optimizer.step()
        self.step_count += 1

        return total.item()

    # ------------------------------------------------------------------
    def consolidate(self):
        """Snapshot current weights (call after a domain 'graduates')."""
        self.ewc.consolidate()

    # ------------------------------------------------------------------
    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
