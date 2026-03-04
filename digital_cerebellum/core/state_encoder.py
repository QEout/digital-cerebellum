"""
State Encoder — numeric state vector encoder for continuous control.

Biological basis:
  Mossy fibres carry proprioceptive and sensory signals as firing rate
  vectors, NOT as language.  The granule cell layer performs dimension
  expansion on these raw numeric signals.

  For Phase 0-5, we used sentence-transformers because the input was
  text (tool names, commands).  For Phase 6 (micro-operations), the
  input is a numeric state vector (game state, robot joint angles,
  UI component positions) that must bypass text encoding entirely.

  This encoder normalises numeric state vectors and projects them
  into the same dimensionality expected by the PatternSeparator,
  enabling the full cerebellar pipeline at 60Hz+ without the ~5ms
  sentence-transformer overhead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Encodes raw numeric state vectors for the cerebellar pipeline.

    Maps variable-length state vectors into a fixed-size representation
    compatible with the PatternSeparator (RFF layer).

    Supports two modes:
      - direct: pad/truncate to target_dim (zero overhead, <0.1ms)
      - projected: learned linear projection (trainable, <0.5ms)
    """

    def __init__(
        self,
        state_dim: int,
        target_dim: int = 388,
        mode: str = "direct",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.mode = mode

        if mode == "projected":
            self.projection = nn.Sequential(
                nn.Linear(state_dim, target_dim),
                nn.LayerNorm(target_dim),
            )
        else:
            self.projection = None

        self._running_mean = np.zeros(state_dim, dtype=np.float32)
        self._running_var = np.ones(state_dim, dtype=np.float32)
        self._count = 0
        self._warmup = 10

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Encode a numeric state vector.

        Parameters
        ----------
        state : (state_dim,) numeric array

        Returns
        -------
        (target_dim,) normalised feature vector
        """
        state = np.asarray(state, dtype=np.float32)

        self._update_stats(state)
        normed = self._normalise(state)

        if self.mode == "projected":
            with torch.no_grad():
                t = torch.from_numpy(normed).float().unsqueeze(0)
                out = self.projection(t).squeeze(0).numpy()
            return out

        return self._pad_or_truncate(normed)

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """Encode a batch of state vectors. (batch, state_dim) → (batch, target_dim)."""
        results = np.stack([self.encode(s) for s in states])
        return results

    def _normalise(self, state: np.ndarray) -> np.ndarray:
        """Running z-score normalisation (online, no full dataset needed)."""
        if self._count < self._warmup:
            return state / (np.abs(state).max() + 1e-8)
        std = np.sqrt(self._running_var + 1e-8)
        return (state - self._running_mean) / std

    def _update_stats(self, state: np.ndarray):
        """Welford's online algorithm for running mean/variance."""
        self._count += 1
        delta = state - self._running_mean
        self._running_mean += delta / self._count
        delta2 = state - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / self._count

    def _pad_or_truncate(self, vec: np.ndarray) -> np.ndarray:
        if len(vec) == self.target_dim:
            return vec
        if len(vec) > self.target_dim:
            return vec[:self.target_dim]
        return np.pad(vec, (0, self.target_dim - len(vec)))

    @property
    def stats(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "target_dim": self.target_dim,
            "mode": self.mode,
            "observations": self._count,
        }
