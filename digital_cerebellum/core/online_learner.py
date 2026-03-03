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

from digital_cerebellum.core.types import ErrorSignal, ErrorType


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
                if name not in self._fisher:
                    continue
                self._fisher[name] += (
                    param.grad.data.pow(2) - self._fisher[name]
                ) / self._fisher_count

    # ------------------------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """EWC regularisation loss: λ/2 · Σ F_i · (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._param_snapshot:
                diff = param - self._param_snapshot[name]
                loss = loss + (self._fisher[name] * diff.pow(2)).sum()
        return (self.ewc_lambda / 2.0) * loss

    # ------------------------------------------------------------------
    def consolidate(self):
        """Snapshot current params as new θ* (call periodically)."""
        self._take_snapshot()


class OnlineLearner:
    """
    Drives online learning for the prediction engine.

    The prediction heads (Purkinje analogues) are trained with SGD + EWC.
    Task heads (DCN readouts) each have independent Adam optimizers.
    A replay buffer stores recent training examples for rehearsal,
    analogous to cerebellar offline replay during consolidation.

    Usage::

        learner = OnlineLearner(prediction_engine)
        loss = learner.learn(z, action_emb, outcome_emb,
                             task_labels={"safety": 1.0, "risk": 0.2})
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        ewc_lambda: float = 400.0,
        task_lr: float = 0.005,
        replay_size: int = 64,
        replay_per_step: int = 4,
    ):
        self.model = model

        head_params = list(model.heads.parameters())
        self.optimizer = torch.optim.SGD(head_params, lr=lr)

        self._task_lr = task_lr
        self.task_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._sync_task_optimizers()

        self.ewc = EWCRegularizer(model, ewc_lambda)
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.step_count = 0

        self._replay_buf: list[dict] = []
        self._replay_size = replay_size
        self._replay_per_step = replay_per_step

    def _sync_task_optimizers(self):
        """Create an Adam optimizer for each registered task head."""
        for name, head in self.model.task_heads.items():
            if name not in self.task_optimizers:
                self.task_optimizers[name] = torch.optim.Adam(
                    head.parameters(), lr=self._task_lr,
                )

    # ------------------------------------------------------------------
    def learn(
        self,
        z: np.ndarray,
        actual_action_emb: np.ndarray,
        actual_outcome_emb: np.ndarray,
        task_labels: dict[str, float] | None = None,
        safe_label: bool | None = None,
    ) -> float:
        """
        One SGD step on prediction heads + one Adam step per task head,
        plus replay of recent examples from the buffer.

        ``task_labels``: dict mapping task head name → target value.
        ``safe_label``:  backward-compatible shorthand for {"safety": 0/1}.
        """
        labels = dict(task_labels or {})
        if safe_label is not None and "safety" not in labels:
            labels["safety"] = 1.0 if safe_label else 0.0

        loss = self._learn_one(z, actual_action_emb, actual_outcome_emb, labels)

        # Store in replay buffer
        self._replay_buf.append({
            "z": z, "a": actual_action_emb, "o": actual_outcome_emb,
            "labels": labels,
        })
        if len(self._replay_buf) > self._replay_size:
            self._replay_buf.pop(0)

        # Replay recent examples
        import random as _rng
        n_replay = min(self._replay_per_step, len(self._replay_buf) - 1)
        if n_replay > 0:
            samples = _rng.sample(self._replay_buf[:-1], n_replay)
            for s in samples:
                self._learn_one(s["z"], s["a"], s["o"], s["labels"])

        self.step_count += 1
        return loss

    def _learn_one(
        self,
        z: np.ndarray,
        actual_action_emb: np.ndarray,
        actual_outcome_emb: np.ndarray,
        labels: dict[str, float],
    ) -> float:
        """Single training step (used by learn + replay)."""
        self.model.train()
        self._sync_task_optimizers()

        z_t = torch.from_numpy(z).float().unsqueeze(0)
        target_a = torch.from_numpy(actual_action_emb).float().unsqueeze(0)
        target_o = torch.from_numpy(actual_outcome_emb).float().unsqueeze(0)
        ones = torch.ones(1)

        # --- Prediction heads (EWC-regularised) ---
        pred_loss = torch.tensor(0.0)
        for head in self.model.heads:
            a, o = head(z_t)
            pred_loss = pred_loss + self.cos_loss(a, target_a, ones)
            pred_loss = pred_loss + self.cos_loss(o, target_o, ones)
        pred_loss = pred_loss / (len(self.model.heads) * 2)

        ewc_loss = self.ewc.penalty()
        head_total = pred_loss + ewc_loss

        self.optimizer.zero_grad()
        head_total.backward()
        self.ewc.update_fisher(self.model)
        self.optimizer.step()

        # --- Task heads (independent, no EWC) ---
        task_loss_total = 0.0
        for name, target_val in labels.items():
            if name not in self.model.task_heads or name not in self.task_optimizers:
                continue
            th = self.model.task_heads[name]
            opt = self.task_optimizers[name]
            target_t = torch.tensor([[target_val]])
            pred_t = th(z_t)
            if th.activation == "sigmoid":
                loss = nn.functional.binary_cross_entropy(pred_t, target_t)
            else:
                loss = nn.functional.mse_loss(pred_t, target_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            task_loss_total += loss.item()

        return head_total.item() + task_loss_total

    # ------------------------------------------------------------------
    def consolidate(self):
        """Snapshot current weights (call after a domain 'graduates')."""
        self.ewc.consolidate()

    # ------------------------------------------------------------------
    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
