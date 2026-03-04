"""
Temporal Pattern Detector — determines if inputs have sequential structure.

Biology: The cerebellum's temporal processing circuits (molecular layer
interneurons, Golgi cells) don't fire uniformly — they are modulated by
task demands. When stimuli are independent (no temporal correlation), these
circuits contribute little. When sequences have temporal structure, they
become critical.

Digital: This module monitors the input stream and computes a
`temporal_strength` score in [0, 1]:
  - 0.0 → inputs are i.i.d. (no sequential pattern) → bypass Phase 2
  - 1.0 → strong temporal autocorrelation → Phase 2 fully engaged

The score drives adaptive gating of FrequencyFilter, GolgiGate, and
StateEstimator, ensuring Phase 2 never hurts on static tasks while
providing full benefit on sequential tasks.

Method: Exponentially-weighted autocorrelation of consecutive input
features + input variance tracking.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class TemporalPatternDetector:
    """
    Detects temporal structure in the input stream.

    Parameters
    ----------
    window : int
        Number of recent inputs to consider.
    ema_alpha : float
        Smoothing for the temporal_strength EMA.
    min_samples : int
        Minimum samples before producing a non-zero score.
    """

    def __init__(
        self,
        window: int = 20,
        ema_alpha: float = 0.1,
        min_samples: int = 5,
    ):
        self.window = window
        self.ema_alpha = ema_alpha
        self.min_samples = min_samples

        self._recent: deque[np.ndarray] = deque(maxlen=window)
        self._autocorr_ema: float = 0.0
        self._variance_ema: float = 0.0
        self._temporal_strength: float = 0.0
        self._n: int = 0

    def observe(self, z: np.ndarray) -> float:
        """
        Feed in a new RFF feature vector and return temporal_strength.

        Parameters
        ----------
        z : (rff_dim,) array — RFF features for the current input.

        Returns
        -------
        temporal_strength : float in [0, 1]
        """
        self._n += 1
        z_flat = z.ravel().astype(np.float32)

        if len(self._recent) >= 1:
            prev = self._recent[-1]
            norm_z = np.linalg.norm(z_flat)
            norm_p = np.linalg.norm(prev)
            if norm_z > 1e-8 and norm_p > 1e-8:
                cosine_sim = float(np.dot(z_flat, prev) / (norm_z * norm_p))
            else:
                cosine_sim = 0.0

            self._autocorr_ema = (
                (1 - self.ema_alpha) * self._autocorr_ema
                + self.ema_alpha * abs(cosine_sim)
            )

        if len(self._recent) >= 3:
            stack = np.stack(list(self._recent)[-5:], axis=0)
            var = float(np.mean(np.var(stack, axis=0)))
            self._variance_ema = (
                (1 - self.ema_alpha) * self._variance_ema
                + self.ema_alpha * var
            )

        self._recent.append(z_flat.copy())

        if self._n < self.min_samples:
            self._temporal_strength = 0.0
            return 0.0

        autocorr_signal = self._autocorr_ema
        var_signal = min(self._variance_ema * 10.0, 1.0)

        raw_strength = 0.7 * autocorr_signal + 0.3 * (1.0 - var_signal)
        raw_strength = max(0.0, min(1.0, raw_strength))

        self._temporal_strength = (
            (1 - self.ema_alpha) * self._temporal_strength
            + self.ema_alpha * raw_strength
        )

        return self._temporal_strength

    @property
    def temporal_strength(self) -> float:
        return self._temporal_strength

    @property
    def stats(self) -> dict:
        return {
            "temporal_strength": round(self._temporal_strength, 4),
            "autocorrelation": round(self._autocorr_ema, 4),
            "variance": round(self._variance_ema, 6),
            "observations": self._n,
        }

    def reset(self):
        self._recent.clear()
        self._autocorr_ema = 0.0
        self._variance_ema = 0.0
        self._temporal_strength = 0.0
        self._n = 0
