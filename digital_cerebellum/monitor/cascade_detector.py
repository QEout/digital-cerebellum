"""
Error Cascade Detector — the core of agent reliability.

Biological basis:
  When the cerebellum detects sustained high SPE (sensory prediction
  error), it triggers a climbing fibre burst that interrupts the
  ongoing motor program.  This prevents a small error from propagating
  into a catastrophic failure.

  In agents, the equivalent is: if several steps in a row produce
  unexpected outcomes (high SPE), the task is going off the rails
  and should be paused before more damage is done.

Detection strategy:
  1. Track SPE over a sliding window
  2. Count consecutive steps with SPE above threshold
  3. Detect upward trend in SPE (errors getting worse)
  4. Combine into a cascade risk score (0-1)
  5. When risk exceeds threshold → recommend pause
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from digital_cerebellum.monitor.types import CascadeStatus


class ErrorCascadeDetector:
    """
    Tracks SPE across agent steps to detect error accumulation.

    Framework-agnostic: only sees SPE numbers, doesn't know what
    the agent is doing.
    """

    def __init__(
        self,
        window_size: int = 10,
        spe_threshold: float = 0.9,
        consecutive_limit: int = 3,
        cascade_risk_threshold: float = 0.7,
        trend_weight: float = 0.3,
    ):
        self._window_size = window_size
        self._spe_threshold = spe_threshold
        self._consecutive_limit = consecutive_limit
        self._cascade_risk_threshold = cascade_risk_threshold
        self._trend_weight = trend_weight

        self._spe_history: deque[float] = deque(maxlen=window_size)
        self._consecutive_high: int = 0
        self._total_steps: int = 0
        self._total_pauses: int = 0

    def observe(self, spe: float) -> CascadeStatus:
        """
        Record a new SPE observation and assess cascade risk.

        Call this after every agent step with the SPE from
        StepForwardModel.compute_spe().
        """
        self._total_steps += 1
        self._spe_history.append(spe)

        if spe > self._spe_threshold:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0

        risk = float(self._compute_risk())
        is_cascading = bool(risk >= self._cascade_risk_threshold)
        trend = self._compute_trend()

        if is_cascading:
            self._total_pauses += 1

        return CascadeStatus(
            risk=risk,
            is_cascading=is_cascading,
            consecutive_high=self._consecutive_high,
            trend=trend,
            mean_recent_spe=self._mean_spe(),
            details={
                "spe": round(spe, 4),
                "threshold": self._spe_threshold,
                "window_fill": len(self._spe_history),
            },
        )

    def _compute_risk(self) -> float:
        """
        Combine multiple signals into a single cascade risk score.

        Three factors:
        1. Consecutive high-SPE steps (most important)
        2. Mean recent SPE relative to threshold
        3. Upward trend in SPE
        """
        if not self._spe_history:
            return 0.0

        consecutive_ratio = min(
            self._consecutive_high / self._consecutive_limit, 1.0
        )

        mean_spe = self._mean_spe()
        magnitude_ratio = min(mean_spe / max(self._spe_threshold, 1e-9), 1.5)

        trend_score = 0.0
        if len(self._spe_history) >= 4:
            half = len(self._spe_history) // 2
            old = np.mean(list(self._spe_history)[:half])
            new = np.mean(list(self._spe_history)[half:])
            if old > 1e-9:
                trend_score = max(0.0, (new - old) / old)
                trend_score = min(trend_score, 1.0)

        risk = (
            0.5 * consecutive_ratio
            + 0.2 * min(magnitude_ratio, 1.0)
            + self._trend_weight * trend_score
        )
        return min(risk, 1.0)

    def _compute_trend(self) -> str:
        if len(self._spe_history) < 4:
            return "insufficient_data"
        half = len(self._spe_history) // 2
        old = np.mean(list(self._spe_history)[:half])
        new = np.mean(list(self._spe_history)[half:])
        if new > old * 1.1:
            return "increasing"
        if new < old * 0.9:
            return "decreasing"
        return "stable"

    def _mean_spe(self) -> float:
        if not self._spe_history:
            return 0.0
        return float(np.mean(list(self._spe_history)))

    def reset(self) -> None:
        """Reset for a new task/episode."""
        self._spe_history.clear()
        self._consecutive_high = 0

    @property
    def stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "total_pauses": self._total_pauses,
            "consecutive_high": self._consecutive_high,
            "mean_recent_spe": round(self._mean_spe(), 4),
            "window_fill": len(self._spe_history),
        }
