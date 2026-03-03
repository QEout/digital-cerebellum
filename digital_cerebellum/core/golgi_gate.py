"""
Golgi Feedback Gate — Golgi Cell analogue.

Biology: Golgi cells form an inhibitory feedback loop in the granule cell layer.
They monitor granule cell population activity and dynamically regulate sparsity:
when too many granule cells fire, Golgi inhibition increases; when too few fire,
it decreases.  This maintains optimal pattern separation performance.

Reference: "A computational model of the cerebellar granular layer calibrated
to experimental data" — medium inhibition levels are optimal.

Digital: A learnable, feedback-modulated gate applied after the RFF expansion.
Unlike the simple sigmoid gate in Phase 0, this version tracks population
activity statistics and uses them to adaptively control sparsity.

z_gated = z * gate(z, activity_stats)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GolgiGate(nn.Module):
    """
    Adaptive feedback gate for RFF outputs.

    Parameters
    ----------
    dim : int
        Feature dimension (rff_dim).
    target_sparsity : float
        Desired fraction of active (non-zero) features. The gate learns to
        push population activity toward this target.
    feedback_lr : float
        How quickly the gate adapts to activity deviations.
    """

    def __init__(
        self,
        dim: int = 4096,
        target_sparsity: float = 0.1,
        feedback_lr: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.target_sparsity = target_sparsity
        self.feedback_lr = feedback_lr

        self.gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.01)

        self.register_buffer("_activity_ema", torch.full((dim,), target_sparsity))
        self.register_buffer("_inhibition_bias", torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, dim) or (dim,)

        Returns
        -------
        z_gated : same shape as z
        """
        squeezed = z.dim() == 1
        if squeezed:
            z = z.unsqueeze(0)

        activity = (z.abs() > 1e-6).float().mean(dim=0)
        self._activity_ema = 0.95 * self._activity_ema + 0.05 * activity.detach()

        sparsity_error = self._activity_ema - self.target_sparsity
        self._inhibition_bias = self._inhibition_bias + self.feedback_lr * sparsity_error

        gate_input = self.gate_proj(z) - self._inhibition_bias.unsqueeze(0)
        gate = torch.sigmoid(gate_input)

        z_gated = z * gate

        if squeezed:
            z_gated = z_gated.squeeze(0)
        return z_gated

    @property
    def stats(self) -> dict:
        return {
            "mean_activity": self._activity_ema.mean().item(),
            "mean_inhibition": self._inhibition_bias.mean().item(),
            "target_sparsity": self.target_sparsity,
        }
