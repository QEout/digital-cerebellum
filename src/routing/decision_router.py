"""
Decision Router — Deep Cerebellar Nuclei (DCN) analogue.

Phase 0: threshold-based routing with RPE-adaptive thresholds.
Phase 1: replace with ActiveDCN (learnable smoother + delay compensator).
"""

from __future__ import annotations

from src.core.types import ErrorSignal, ErrorType, PredictionOutput, RouteDecision, RoutingResult


class DecisionRouter:
    """
    Adaptive threshold router.

    Each microzone maintains its own router instance with independent
    thresholds that shift via reward prediction error.
    """

    def __init__(
        self,
        threshold_high: float = 0.95,
        threshold_low: float = 0.5,
        rpe_lr: float = 0.05,
        min_high: float = 0.70,
        max_high: float = 0.99,
    ):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.rpe_lr = rpe_lr
        self.min_high = min_high
        self.max_high = max_high

        self._cumulative_rpe = 0.0
        self._total_routed = 0

    # ------------------------------------------------------------------
    def route(self, prediction: PredictionOutput) -> RoutingResult:
        c = prediction.confidence
        self._total_routed += 1

        if c < self.threshold_low:
            return RoutingResult(RouteDecision.SLOW, c, "below low threshold")
        if c >= self.threshold_high:
            return RoutingResult(RouteDecision.FAST, c, "above high threshold")
        return RoutingResult(RouteDecision.SHADOW, c, "mid-range → shadow execution")

    # ------------------------------------------------------------------
    def update_from_reward(self, error: ErrorSignal):
        """
        Adapt thresholds based on reward prediction error.

        Positive RPE → trust cerebellum more → lower threshold.
        Negative RPE → trust less → raise threshold.
        """
        if error.error_type is not ErrorType.REWARD:
            return

        self._cumulative_rpe += error.value

        delta = -self.rpe_lr * error.value   # positive RPE → negative delta → lower threshold
        self.threshold_high = max(self.min_high, min(self.max_high, self.threshold_high + delta))

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "threshold_high": round(self.threshold_high, 4),
            "threshold_low": round(self.threshold_low, 4),
            "cumulative_rpe": round(self._cumulative_rpe, 4),
            "total_routed": self._total_routed,
        }
