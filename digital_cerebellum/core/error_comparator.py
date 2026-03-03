"""
Error Comparator — Climbing fibre analogue.

Generates three independent error channels:
  SPE  (sensory prediction error)  — predicted vs actual outcome
  TPE  (temporal prediction error)  — predicted vs actual timing  (Phase 1)
  RPE  (reward prediction error)    — expected value vs user feedback (Phase 1)

Phase 0 implements SPE only; the other channels expose stub interfaces.
"""

from __future__ import annotations

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

    Phase 0: only ``compute_sensory_error`` is fully implemented.
    """

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

        return ErrorSignal(
            error_type=ErrorType.SENSORY,
            value=magnitude,
            vector=error_vec,
            source_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Channel 2: Temporal Prediction Error (TPE) — Phase 1 stub
    # ------------------------------------------------------------------
    def compute_temporal_error(
        self,
        predicted_time: float,
        actual_time: float,
        event_id: str = "",
    ) -> ErrorSignal:
        delta = abs(predicted_time - actual_time)
        return ErrorSignal(
            error_type=ErrorType.TEMPORAL,
            value=delta,
            vector=None,
            source_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Channel 3: Reward Prediction Error (RPE) — Phase 1 stub
    # ------------------------------------------------------------------
    def compute_reward_error(
        self,
        user_feedback: float,
        event_id: str = "",
    ) -> ErrorSignal:
        """
        Parameters
        ----------
        user_feedback : float
            +1 = positive, -1 = negative, 0 = neutral.
        """
        return ErrorSignal(
            error_type=ErrorType.REWARD,
            value=user_feedback,
            vector=None,
            source_event_id=event_id,
        )
