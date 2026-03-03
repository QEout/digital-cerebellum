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

from src.core.types import HeadPrediction, PredictionOutput


@dataclass
class EngineConfig:
    rff_dim: int = 4096
    action_dim: int = 128
    outcome_dim: int = 128
    num_heads: int = 4          # K — population size
    temperature: float = 1.0    # τ for confidence = exp(-var/τ)


class PredictionHead(nn.Module):
    """One Purkinje cell — a single linear readout."""

    def __init__(self, rff_dim: int, action_dim: int, outcome_dim: int):
        super().__init__()
        self.action_proj = nn.Linear(rff_dim, action_dim)
        self.outcome_proj = nn.Linear(rff_dim, outcome_dim)

    def forward(self, z: torch.Tensor):
        return self.action_proj(z), self.outcome_proj(z)


class PredictionEngine(nn.Module):
    """
    K-head population predictor.

    forward(z) → PredictionOutput with emergent confidence.
    """

    def __init__(self, cfg: EngineConfig | None = None):
        super().__init__()
        cfg = cfg or EngineConfig()
        self.cfg = cfg

        self.heads = nn.ModuleList([
            PredictionHead(cfg.rff_dim, cfg.action_dim, cfg.outcome_dim)
            for _ in range(cfg.num_heads)
        ])

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

        # Confidence = exp(-variance / τ)
        action_var = actions_stack.var(dim=0).mean(dim=-1)   # (batch,)
        outcome_var = outcomes_stack.var(dim=0).mean(dim=-1)
        total_var = (action_var + outcome_var) / 2.0
        confidence = torch.exp(-total_var / self.cfg.temperature)

        # Build per-head predictions (detached numpy for inspection)
        head_preds: list[HeadPrediction] = []
        for k in range(len(self.heads)):
            head_preds.append(HeadPrediction(
                action_embedding=all_actions[k][0].detach().cpu().numpy(),
                outcome_embedding=all_outcomes[k][0].detach().cpu().numpy(),
            ))

        raw = torch.cat([mean_action, mean_outcome, confidence.unsqueeze(-1)], dim=-1)

        output = PredictionOutput(
            action_embedding=mean_action[0].detach().cpu().numpy() if squeezed else mean_action.detach().cpu().numpy(),
            outcome_embedding=mean_outcome[0].detach().cpu().numpy() if squeezed else mean_outcome.detach().cpu().numpy(),
            confidence=confidence[0].item() if squeezed else confidence.detach().cpu().numpy(),
            head_predictions=head_preds,
            _raw_tensor=raw,
        )
        return output

    # ------------------------------------------------------------------
    def predict_numpy(self, z: np.ndarray) -> PredictionOutput:
        """Convenience: numpy in, PredictionOutput out."""
        with torch.no_grad():
            return self.forward(torch.from_numpy(z).float())
