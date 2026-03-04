"""
Somatic Marker — gut-feeling system inspired by Damasio's hypothesis.

Biological basis:
  Past experiences leave "somatic markers" — associations between
  situation patterns and emotional valence — that bias future decisions
  before conscious reasoning.  The cerebellum contributes by sending
  prediction error signals via the cerebello-thalamo-cortical pathway
  (J. Neurosci. 2025).  The PATTERN of Purkinje cell population
  disagreement (not just its magnitude) carries information about
  what kind of situation this is.

Digital implementation:
  1. Extract a "divergence fingerprint" from the K prediction heads:
     which heads agree with which, the structure of their disagreement.
  2. Store fingerprints alongside their outcome valence (+/-).
  3. On new predictions, compare the current fingerprint against the
     valence library → produce a GutFeeling with direction and intensity.
  4. Strong negative gut feelings can override normal routing
     (force slow path even when confidence is high).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from digital_cerebellum.core.types import HeadPrediction


@dataclass
class GutFeeling:
    """The output of the somatic marker system."""

    valence: float        # -1.0 (danger) to +1.0 (safe)
    intensity: float      # 0.0 (no feeling) to 1.0 (overwhelming)
    trigger_pattern: str  # closest stored marker that triggered this
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_override(self) -> bool:
        """True when the gut feeling is strong enough to force slow-path."""
        return self.intensity > 0.6 and self.valence < -0.3

    @property
    def label(self) -> str:
        if self.intensity < 0.2:
            return "neutral"
        if self.valence > 0.3:
            return "positive"
        if self.valence < -0.3:
            return "alarm"
        return "uneasy"


@dataclass
class _ValenceMarker:
    """One stored somatic marker — a fingerprint + associated valence."""

    fingerprint: np.ndarray    # divergence pattern vector
    valence: float             # outcome: +1 success, -1 failure
    domain: str = ""
    created_step: int = 0
    strength: float = 1.0      # decays over time


class SomaticMarker:
    """
    Builds and queries a library of somatic markers (valence patterns).

    The core insight: when prediction heads disagree in a specific pattern,
    that pattern is informative.  If similar patterns previously led to bad
    outcomes, the system should feel uneasy — even if average confidence
    looks acceptable.
    """

    def __init__(
        self,
        max_markers: int = 500,
        similarity_threshold: float = 0.75,
        decay_rate: float = 0.995,
    ):
        self._markers: deque[_ValenceMarker] = deque(maxlen=max_markers)
        self._similarity_threshold = similarity_threshold
        self._decay_rate = decay_rate
        self._step = 0

    @staticmethod
    def extract_fingerprint(head_predictions: list[HeadPrediction]) -> np.ndarray:
        """
        Extract a divergence fingerprint from K head predictions.

        The fingerprint encodes pairwise cosine similarities between heads,
        capturing the *structure* of agreement/disagreement rather than
        just its magnitude.

        For K=4 heads, this produces a (K*(K-1)/2 * 2) = 12-dim vector
        (6 action pairs + 6 outcome pairs).
        """
        K = len(head_predictions)
        if K < 2:
            return np.zeros(2, dtype=np.float32)

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        pairs_action = []
        pairs_outcome = []
        for i in range(K):
            for j in range(i + 1, K):
                pairs_action.append(
                    _cosine(head_predictions[i].action_embedding,
                            head_predictions[j].action_embedding)
                )
                pairs_outcome.append(
                    _cosine(head_predictions[i].outcome_embedding,
                            head_predictions[j].outcome_embedding)
                )

        return np.array(pairs_action + pairs_outcome, dtype=np.float32)

    def record(
        self,
        head_predictions: list[HeadPrediction],
        valence: float,
        domain: str = "",
    ) -> None:
        """
        Store a new somatic marker after observing an outcome.

        Parameters
        ----------
        head_predictions : from PredictionOutput.head_predictions
        valence : +1.0 = good outcome, -1.0 = bad outcome
        domain : microzone name
        """
        self._step += 1
        fp = self.extract_fingerprint(head_predictions)
        self._markers.append(_ValenceMarker(
            fingerprint=fp,
            valence=np.clip(valence, -1.0, 1.0),
            domain=domain,
            created_step=self._step,
            strength=1.0,
        ))

    def feel(
        self,
        head_predictions: list[HeadPrediction],
        domain: str = "",
    ) -> GutFeeling:
        """
        Produce a gut feeling by comparing the current divergence
        fingerprint against stored somatic markers.

        Uses similarity-weighted valence aggregation: markers with
        fingerprints similar to the current situation contribute
        more to the gut feeling.
        """
        if not self._markers:
            return GutFeeling(valence=0.0, intensity=0.0,
                              trigger_pattern="none")

        current_fp = self.extract_fingerprint(head_predictions)
        fp_norm = np.linalg.norm(current_fp)
        if fp_norm < 1e-9:
            return GutFeeling(valence=0.0, intensity=0.0,
                              trigger_pattern="zero_fingerprint")

        weighted_valence = 0.0
        total_weight = 0.0
        max_sim = -1.0
        trigger = "none"

        for marker in self._markers:
            if domain and marker.domain and marker.domain != domain:
                continue

            mn = np.linalg.norm(marker.fingerprint)
            if mn < 1e-9:
                continue
            sim = float(np.dot(current_fp, marker.fingerprint) / (fp_norm * mn))

            if sim < self._similarity_threshold:
                continue

            w = sim * marker.strength
            weighted_valence += w * marker.valence
            total_weight += w

            if sim > max_sim:
                max_sim = sim
                trigger = f"step_{marker.created_step}_{marker.domain}"

        if total_weight < 1e-9:
            return GutFeeling(valence=0.0, intensity=0.0,
                              trigger_pattern="no_match")

        avg_valence = weighted_valence / total_weight
        intensity = min(1.0, total_weight / max(len(self._markers) * 0.1, 1.0))

        return GutFeeling(
            valence=float(np.clip(avg_valence, -1.0, 1.0)),
            intensity=float(intensity),
            trigger_pattern=trigger,
            details={
                "matched_markers": int(total_weight > 0),
                "max_similarity": float(max_sim),
                "fingerprint_norm": float(fp_norm),
            },
        )

    def decay(self) -> None:
        """Apply time-based decay to all markers (called during sleep)."""
        for marker in self._markers:
            marker.strength *= self._decay_rate

    @property
    def stats(self) -> dict[str, Any]:
        if not self._markers:
            return {"count": 0, "mean_valence": 0.0, "mean_strength": 0.0}
        valences = [m.valence for m in self._markers]
        strengths = [m.strength for m in self._markers]
        return {
            "count": len(self._markers),
            "mean_valence": float(np.mean(valences)),
            "std_valence": float(np.std(valences)),
            "mean_strength": float(np.mean(strengths)),
            "positive_ratio": float(sum(1 for v in valences if v > 0) / len(valences)),
        }
