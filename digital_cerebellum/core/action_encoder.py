"""
Action Encoder — maps continuous actions to/from vector representations.

Biological basis:
  The motor cortex encodes movements as population activity vectors.
  The cerebellum receives an efference copy of these motor commands
  and learns to predict their sensory consequences.

  Actions in games, UIs, and robotics are continuous: mouse positions,
  button timings, force magnitudes.  The ActionEncoder standardises
  these heterogeneous action spaces into uniform vectors that the
  cerebellar pipeline can process.
"""

from __future__ import annotations

import numpy as np


class ActionEncoder:
    """
    Encodes and decodes continuous actions for the cerebellar pipeline.

    Handles:
      - Normalisation to [-1, 1] range
      - Padding/truncation to fixed dimension
      - Action space metadata for decoding
    """

    def __init__(
        self,
        action_dim: int,
        action_names: list[str] | None = None,
        action_ranges: list[tuple[float, float]] | None = None,
    ):
        self.action_dim = action_dim
        self.action_names = action_names or [f"a{i}" for i in range(action_dim)]
        self.action_ranges = action_ranges or [(-1.0, 1.0)] * action_dim

        self._lows = np.array([r[0] for r in self.action_ranges], dtype=np.float32)
        self._highs = np.array([r[1] for r in self.action_ranges], dtype=np.float32)
        self._ranges = self._highs - self._lows
        self._ranges = np.where(self._ranges < 1e-8, 1.0, self._ranges)

    def encode(self, action: np.ndarray | list | dict) -> np.ndarray:
        """
        Encode a raw action into a normalised vector.

        Accepts: numpy array, list, or dict with action_names as keys.
        Returns: (action_dim,) float32 array in [-1, 1] range.
        """
        if isinstance(action, dict):
            raw = np.array(
                [action.get(name, 0.0) for name in self.action_names],
                dtype=np.float32,
            )
        else:
            raw = np.asarray(action, dtype=np.float32)

        if len(raw) > self.action_dim:
            raw = raw[:self.action_dim]
        elif len(raw) < self.action_dim:
            raw = np.pad(raw, (0, self.action_dim - len(raw)))

        normalised = 2.0 * (raw - self._lows) / self._ranges - 1.0
        return np.clip(normalised, -1.0, 1.0)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode a normalised vector back to raw action space."""
        return ((encoded + 1.0) / 2.0) * self._ranges + self._lows

    def decode_to_dict(self, encoded: np.ndarray) -> dict[str, float]:
        """Decode to a named dict."""
        raw = self.decode(encoded)
        return {name: float(val) for name, val in zip(self.action_names, raw)}

    @property
    def stats(self) -> dict:
        return {
            "action_dim": self.action_dim,
            "action_names": self.action_names,
        }
