"""
Prediction Engine — Purkinje cell population analogue.

K independent linear heads read the same sparse RFF code and each produce
an action + outcome prediction.  Confidence emerges from population agreement
(variance), not from a trained sigmoid.

Biology: Purkinje cell population → spatial correlation encodes behaviour
Digital:  K linear heads           → agreement = confidence
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from digital_cerebellum.core.types import HeadPrediction, PredictionOutput


@dataclass
class EngineConfig:
    rff_dim: int = 4096
    action_dim: int = 128
    outcome_dim: int = 128
    num_heads: int = 4          # K — population size
    temperature: float = 1.0    # τ for confidence = exp(-var/τ)


class PredictionHead(nn.Module):
    """
    One Purkinje cell — a linear readout with a unique dendritic mask.

    Each head sees a random subset of the RFF features, mimicking the
    biological fact that each Purkinje cell's dendritic tree connects
    to a different subset of parallel fibres.  This forces genuine
    diversity: agreement across masked views = real confidence.
    """

    def __init__(self, rff_dim: int, action_dim: int, outcome_dim: int,
                 mask_ratio: float = 0.5):
        super().__init__()
        self.action_proj = nn.Linear(rff_dim, action_dim)
        self.outcome_proj = nn.Linear(rff_dim, outcome_dim)

        mask = torch.ones(rff_dim)
        drop_idx = torch.randperm(rff_dim)[:int(rff_dim * mask_ratio)]
        mask[drop_idx] = 0.0
        self.register_buffer("feature_mask", mask)
        self._scale = 1.0 / max(1.0 - mask_ratio, 0.1)

    def forward(self, z: torch.Tensor):
        z_masked = z * self.feature_mask * self._scale
        return self.action_proj(z_masked), self.outcome_proj(z_masked)


class TaskHead(nn.Module):
    """
    Generic task readout head — registered per microzone.

    Maps RFF representation → task-specific scalar or vector output.
    Biologically: a DCN readout neuron for a specific functional domain.
    """

    def __init__(self, rff_dim: int, output_dim: int = 1, activation: str = "sigmoid"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rff_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.activation = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        if self.activation == "softmax":
            return torch.softmax(out, dim=-1)
        return out


class PredictionEngine(nn.Module):
    """
    K-head population predictor + pluggable task heads.

    The prediction heads (Purkinje population) are universal.
    Task heads (DCN readouts) are registered per microzone.
    """

    def __init__(self, cfg: EngineConfig | None = None):
        super().__init__()
        cfg = cfg or EngineConfig()
        self.cfg = cfg

        self.heads = nn.ModuleList([
            PredictionHead(cfg.rff_dim, cfg.action_dim, cfg.outcome_dim)
            for _ in range(cfg.num_heads)
        ])
        self.task_heads = nn.ModuleDict()

    # ------------------------------------------------------------------
    def register_task_head(self, name: str, output_dim: int = 1, activation: str = "sigmoid"):
        """Register a new task-specific readout head (one per microzone task)."""
        self.task_heads[name] = TaskHead(self.cfg.rff_dim, output_dim, activation)

    @property
    def safety_head(self):
        """Backward-compatible accessor."""
        return self.task_heads["safety"] if "safety" in self.task_heads else None

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> PredictionOutput:
        """
        Parameters
        ----------
        z : (rff_dim,) or (batch, rff_dim)

        Returns
        -------
        PredictionOutput  (numpy arrays; gradients kept in _raw_tensor)
        """
        squeezed = z.dim() == 1
        if squeezed:
            z = z.unsqueeze(0)

        all_actions: list[torch.Tensor] = []
        all_outcomes: list[torch.Tensor] = []

        for head in self.heads:
            a, o = head(z)
            all_actions.append(a)
            all_outcomes.append(o)

        # Stack: (K, batch, dim)
        actions_stack = torch.stack(all_actions, dim=0)
        outcomes_stack = torch.stack(all_outcomes, dim=0)

        mean_action = actions_stack.mean(dim=0)     # (batch, action_dim)
        mean_outcome = outcomes_stack.mean(dim=0)

        # Confidence from pairwise cosine agreement across heads.
        # High agreement among masked views → genuine confidence.
        K = len(self.heads)
        normed_a = nn.functional.normalize(actions_stack, dim=-1)   # (K, batch, dim)
        normed_o = nn.functional.normalize(outcomes_stack, dim=-1)

        cos_sum = torch.zeros(normed_a.shape[1])  # (batch,)
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                cos_sum += (normed_a[i] * normed_a[j]).sum(dim=-1)
                cos_sum += (normed_o[i] * normed_o[j]).sum(dim=-1)
                pairs += 2
        mean_cos = cos_sum / max(pairs, 1)  # range [-1, 1]
        confidence = (mean_cos + 1.0) / 2.0  # map to [0, 1]

        # Task heads (DCN readouts)
        task_outputs: dict[str, float] = {}
        task_tensors: list[torch.Tensor] = []
        for name, task_head in self.task_heads.items():
            t_out = task_head(z)  # (batch, output_dim)
            task_tensors.append(t_out)
            if squeezed and t_out.shape[-1] == 1:
                task_outputs[name] = t_out[0, 0].item()
            elif squeezed:
                task_outputs[name] = t_out[0].detach().cpu().numpy()

        # Build per-head predictions (detached numpy for inspection)
        head_preds: list[HeadPrediction] = []
        for k in range(len(self.heads)):
            head_preds.append(HeadPrediction(
                action_embedding=all_actions[k][0].detach().cpu().numpy(),
                outcome_embedding=all_outcomes[k][0].detach().cpu().numpy(),
            ))

        raw_parts = [mean_action, mean_outcome, confidence.unsqueeze(-1)] + task_tensors
        raw = torch.cat(raw_parts, dim=-1)

        output = PredictionOutput(
            action_embedding=mean_action[0].detach().cpu().numpy() if squeezed else mean_action.detach().cpu().numpy(),
            outcome_embedding=mean_outcome[0].detach().cpu().numpy() if squeezed else mean_outcome.detach().cpu().numpy(),
            confidence=confidence[0].item() if squeezed else confidence.detach().cpu().numpy(),
            head_predictions=head_preds,
            task_outputs=task_outputs,
            _raw_tensor=raw,
        )
        return output

    # ------------------------------------------------------------------
    def predict_numpy(self, z: np.ndarray) -> PredictionOutput:
        """Convenience: numpy in, PredictionOutput out."""
        with torch.no_grad():
            return self.forward(torch.from_numpy(z).float())
