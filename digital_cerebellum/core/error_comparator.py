"""
Error Comparator — Climbing fibre analogue.

Three independent error channels, each driving different learning:

  SPE (sensory prediction error)
      predicted vs actual outcome → updates prediction heads
  TPE (temporal prediction error)
      predicted vs actual timing → updates rhythm / scheduling
  RPE (reward prediction error)
      expected reward vs actual feedback → updates decision router thresholds

Biology:
  - SPE ≈ complex spike from climbing fibres (Marr-Albus-Ito model)
  - TPE ≈ timing-sensitive cerebellar learning (eyeblink conditioning)
  - RPE ≈ dopaminergic modulation of cerebellar plasticity (Heffley et al. 2018)
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from digital_cerebellum.core.types import ErrorSignal, ErrorType, PredictionOutput


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity.  Returns 0 when identical, 2 when opposite."""
    dot = np.dot(a, b)
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(1.0 - dot / norm)


class ErrorComparator:
    """
    Computes error signals by comparing predictions with actual outcomes.

    All three channels maintain running statistics for adaptive thresholds.
    """

    def __init__(self, window_size: int = 100):
        self._spe_history: deque[float] = deque(maxlen=window_size)
        self._tpe_history: deque[float] = deque(maxlen=window_size)
        self._rpe_history: deque[float] = deque(maxlen=window_size)
        self._timing_model = _TimingModel()

    # ------------------------------------------------------------------
    # Channel 1: Sensory Prediction Error (SPE)
    # ------------------------------------------------------------------
    def compute_sensory_error(
        self,
        prediction: PredictionOutput,
        actual_action_emb: np.ndarray,
        actual_outcome_emb: np.ndarray,
        event_id: str = "",
    ) -> ErrorSignal:
        action_err = cosine_distance(prediction.action_embedding, actual_action_emb)
        outcome_err = cosine_distance(prediction.outcome_embedding, actual_outcome_emb)

        error_vec = np.array([action_err, outcome_err], dtype=np.float32)
        magnitude = float((action_err + outcome_err) / 2.0)
        self._spe_history.append(magnitude)

        return ErrorSignal(
            error_type=ErrorType.SENSORY,
            value=magnitude,
            vector=error_vec,
            source_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Channel 2: Temporal Prediction Error (TPE)
    # ------------------------------------------------------------------
    def compute_temporal_error(
        self,
        predicted_time: float,
        actual_time: float,
        event_id: str = "",
    ) -> ErrorSignal:
        """
        Temporal prediction error — mismatch between when we expected
        an event and when it actually arrived.

        Uses Weber's law: the error is relative to the expected interval,
        matching the scalar property of cerebellar timing.
        """
        raw_delta = actual_time - predicted_time
        weber_denom = max(abs(predicted_time), 0.001)
        normalised = raw_delta / weber_denom

        self._tpe_history.append(abs(normalised))
        self._timing_model.update(actual_time)

        return ErrorSignal(
            error_type=ErrorType.TEMPORAL,
            value=float(normalised),
            vector=np.array([raw_delta, normalised, predicted_time, actual_time],
                            dtype=np.float32),
            source_event_id=event_id,
        )

    def predict_next_time(self) -> float:
        """Predict when the next event will occur based on recent intervals."""
        return self._timing_model.predict()

    # ------------------------------------------------------------------
    # Channel 3: Reward Prediction Error (RPE)
    # ------------------------------------------------------------------
    def compute_reward_error(
        self,
        expected_reward: float,
        actual_reward: float,
        event_id: str = "",
    ) -> ErrorSignal:
        """
        Reward prediction error — mismatch between expected and actual value.

        Parameters
        ----------
        expected_reward : float
            The cerebellum's expectation (e.g. predicted confidence).
        actual_reward : float
            Actual outcome value. +1 = success, -1 = failure, 0 = neutral.
            Can also be continuous (e.g. user satisfaction score).
        """
        delta = actual_reward - expected_reward
        self._rpe_history.append(delta)

        return ErrorSignal(
            error_type=ErrorType.REWARD,
            value=float(delta),
            vector=np.array([expected_reward, actual_reward, delta],
                            dtype=np.float32),
            source_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, dict[str, float]]:
        def _summarise(hist: deque) -> dict[str, float]:
            if not hist:
                return {"mean": 0.0, "std": 0.0, "count": 0}
            arr = np.array(hist)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "count": len(hist),
                "recent_mean": float(arr[-min(10, len(arr)):].mean()),
            }
        return {
            "spe": _summarise(self._spe_history),
            "tpe": _summarise(self._tpe_history),
            "rpe": _summarise(self._rpe_history),
        }

    def is_improving(self, channel: str = "spe", window: int = 20) -> bool:
        """Check if recent errors are lower than older errors."""
        hist = getattr(self, f"_{channel}_history")
        if len(hist) < window * 2:
            return False
        arr = np.array(hist)
        old_mean = float(np.abs(arr[-window * 2:-window]).mean())
        new_mean = float(np.abs(arr[-window:]).mean())
        return new_mean < old_mean


class _TimingModel:
    """
    Simple exponential smoothing model for event timing.

    Tracks inter-event intervals and predicts the next one.
    Cerebellar timing uses a similar mechanism — adaptive temporal
    expectations based on recent experience.
    """

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._last_time: float | None = None
        self._smoothed_interval: float = 1.0
        self._intervals: deque[float] = deque(maxlen=50)

    def update(self, timestamp: float):
        if self._last_time is not None:
            interval = timestamp - self._last_time
            if interval > 0:
                self._intervals.append(interval)
                self._smoothed_interval = (
                    self._alpha * interval
                    + (1 - self._alpha) * self._smoothed_interval
                )
        self._last_time = timestamp

    def predict(self) -> float:
        if self._last_time is None:
            return time.time() + self._smoothed_interval
        return self._last_time + self._smoothed_interval
