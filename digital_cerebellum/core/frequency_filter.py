"""
Frequency Filter — Molecular Layer Interneuron analogue.

Biology: basket cells filter low-frequency signals, stellate cells filter
high-frequency signals. Together they decompose the temporal dynamics of
cerebellar input into complementary frequency bands.

Digital: Inserted between Pattern Separator and Prediction Engine, this module
splits the RFF representation into low-pass and high-pass components, enabling
the downstream predictor to distinguish transient events from sustained trends.

Implementation uses an exponential moving average (EMA) as the low-pass filter;
the high-pass residual is the original signal minus the EMA.  Both bands are
concatenated (or selected) before feeding into the prediction heads.

Reference: 2025 Nature — basket cells filter low freq, stellate cells filter high freq.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyFilter(nn.Module):
    """
    Dual-band frequency decomposition on RFF feature streams.

    Parameters
    ----------
    dim : int
        Feature dimension (should equal rff_dim).
    alpha : float
        EMA smoothing factor. Smaller = smoother low-pass (more memory).
        Biologically: the time-constant of basket-cell inhibition.
    mode : str
        "concat" — output is [low; high] (dim*2).
        "gate"   — learnable mixing gate decides per-dimension blend.
    """

    def __init__(self, dim: int = 4096, alpha: float = 0.1, mode: str = "gate"):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.mode = mode

        self.register_buffer("_ema", torch.zeros(dim))
        self._initialized = False

        if mode == "gate":
            self.mix_gate = nn.Linear(dim * 2, dim)
        else:
            self.mix_gate = None

    @property
    def output_dim(self) -> int:
        return self.dim if self.mode == "gate" else self.dim * 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, dim) or (dim,)

        Returns
        -------
        filtered : (batch, output_dim) or (output_dim,)
        """
        squeezed = z.dim() == 1
        if squeezed:
            z = z.unsqueeze(0)

        batch_size = z.shape[0]
        outputs = []

        for i in range(batch_size):
            zi = z[i]

            if not self._initialized:
                self._ema.copy_(zi.detach())
                self._initialized = True

            self._ema = (1 - self.alpha) * self._ema + self.alpha * zi.detach()

            low_pass = self._ema.clone()
            high_pass = zi - low_pass

            if self.mode == "gate":
                combined = torch.cat([low_pass, high_pass], dim=-1)
                gate = torch.sigmoid(self.mix_gate(combined))
                out = gate * low_pass + (1 - gate) * high_pass
            else:
                out = torch.cat([low_pass, high_pass], dim=-1)

            outputs.append(out)

        result = torch.stack(outputs, dim=0)
        if squeezed:
            result = result.squeeze(0)
        return result

    def reset(self):
        """Reset the EMA state (e.g. between episodes)."""
        self._ema.zero_()
        self._initialized = False
