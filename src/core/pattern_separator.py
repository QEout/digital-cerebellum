"""
Pattern Separator — Granule Cell Layer analogue.

Maps low-dimensional input vectors into a high-dimensional sparse space using
Random Fourier Features (RFF), approximating an RBF kernel.  This makes
downstream linear prediction heads powerful enough (the kernel trick).

Biology: mossy fibre input → granule cells (dimension explosion) → sparse code
Digital:  event embedding    → RFF projection                     → top-k sparse
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PatternSeparator(nn.Module):
    """
    Random Fourier Features with optional Golgi feedback gating.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming feature vector.
    rff_dim : int
        Dimensionality of the RFF output (D >> input_dim).
    gamma : float
        RBF kernel bandwidth.  Larger γ → sharper kernel.
    sparsity : float
        Fraction of dimensions to keep after top-k (0 < sparsity ≤ 1).
    enable_golgi : bool
        If True, apply a learnable sigmoid gate after RFF (Phase 2).
    """

    def __init__(
        self,
        input_dim: int = 512,
        rff_dim: int = 4096,
        gamma: float = 1.0,
        sparsity: float = 0.1,
        enable_golgi: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.sparsity = sparsity
        self.enable_golgi = enable_golgi
        self.top_k = max(1, int(rff_dim * sparsity))

        # Fixed random projection (not trained — mirrors frozen granule-cell wiring)
        W = torch.randn(rff_dim, input_dim) * gamma
        b = torch.empty(rff_dim).uniform_(0, 2 * math.pi)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        self.scale = math.sqrt(2.0 / rff_dim)

        # Golgi feedback gate (Phase 2 — learnable gain per dimension)
        if enable_golgi:
            self.golgi_linear = nn.Linear(rff_dim, rff_dim)
        else:
            self.golgi_linear = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)   or   (input_dim,)

        Returns
        -------
        z : same leading dims, but last dim = rff_dim  (sparse)
        """
        squeezed = x.dim() == 1
        if squeezed:
            x = x.unsqueeze(0)

        # RFF: z = sqrt(2/D) * cos(Wx + b)
        z = torch.cos(x @ self.W.T + self.b) * self.scale    # (batch, rff_dim)

        # Golgi gate (Phase 2)
        if self.golgi_linear is not None:
            gate = torch.sigmoid(self.golgi_linear(z))
            z = z * gate

        # Top-k sparsification
        z = self._top_k_sparse(z)

        if squeezed:
            z = z.squeeze(0)
        return z

    # ------------------------------------------------------------------
    def _top_k_sparse(self, z: torch.Tensor) -> torch.Tensor:
        """Keep only the top-k absolute-value dimensions; zero the rest."""
        _, indices = torch.topk(z.abs(), self.top_k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, indices, 1.0)
        return z * mask

    # ------------------------------------------------------------------
    def encode_event(self, feature_vector: np.ndarray) -> np.ndarray:
        """Convenience: numpy in → numpy out (no grad)."""
        with torch.no_grad():
            x = torch.from_numpy(feature_vector).float()
            z = self.forward(x)
            return z.numpy()
