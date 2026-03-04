"""
Self-Model — metacognitive competency awareness.

Biological basis:
  During motor learning, metacognitive confidence judgments reflect the
  integration of error history through recency-weighted averaging
  (Metacognitive Judgments during Visuomotor Learning, 2023).
  The cerebellum's multiple microzones each track different functional
  domains, and the aggregate of their error/confidence patterns forms
  an implicit self-model: "I'm good at catching balls but bad at
  threading needles."

  Recent AI work (EGPO 2026, HTC 2026, meta-d' 2025) formalises this
  as confidence calibration — the ability to predict one's own accuracy.

Digital implementation:
  Per-microzone tracking of:
    - Accuracy (rolling window)
    - Confidence calibration error (|predicted_conf - actual_accuracy| per bin)
    - Fast-path success rate
    - Learning trend (improving / stable / declining)
  Aggregated into a SelfReport that answers:
    "What am I good at?  Where should I defer to the brain?"
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CompetencyProfile:
    """Competency assessment for one domain."""

    domain: str
    skill_level: str         # "novice" | "learning" | "competent" | "expert"
    accuracy: float          # rolling accuracy [0, 1]
    calibration_error: float # Expected Calibration Error
    fast_path_ratio: float   # fraction routed fast
    fast_path_accuracy: float
    learning_trend: str      # "improving" | "stable" | "declining"
    total_observations: int
    confidence_mean: float


@dataclass
class SelfReport:
    """Full metacognitive self-assessment."""

    competencies: dict[str, CompetencyProfile]
    overall_calibration: float
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str

    def to_prompt(self) -> str:
        """Format as a natural-language string for LLM context injection."""
        lines = ["[Self-Assessment]"]
        for name, cp in self.competencies.items():
            lines.append(
                f"  {name}: {cp.skill_level} "
                f"(acc={cp.accuracy:.0%}, cal_err={cp.calibration_error:.3f}, "
                f"fast={cp.fast_path_ratio:.0%})"
            )
        if self.strengths:
            lines.append(f"  Strengths: {', '.join(self.strengths)}")
        if self.weaknesses:
            lines.append(f"  Weaknesses: {', '.join(self.weaknesses)}")
        lines.append(f"  Overall calibration: {self.overall_calibration:.3f}")
        lines.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(lines)


class _DomainProfile:
    """Raw tracking data for one domain."""

    def __init__(self, window: int = 100, cal_bins: int = 10):
        self._window = window
        self._cal_bins = cal_bins

        self._outcomes: deque[bool] = deque(maxlen=window)
        self._confidences: deque[float] = deque(maxlen=window)
        self._routes: deque[str] = deque(maxlen=window)
        self._fast_outcomes: deque[bool] = deque(maxlen=window)
        self.total: int = 0

    def record(
        self,
        correct: bool,
        confidence: float,
        route: str,
    ) -> None:
        self.total += 1
        self._outcomes.append(correct)
        self._confidences.append(confidence)
        self._routes.append(route)
        if route == "fast":
            self._fast_outcomes.append(correct)

    @property
    def accuracy(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def confidence_mean(self) -> float:
        if not self._confidences:
            return 0.0
        return float(np.mean(list(self._confidences)))

    @property
    def fast_path_ratio(self) -> float:
        if not self._routes:
            return 0.0
        return sum(1 for r in self._routes if r == "fast") / len(self._routes)

    @property
    def fast_path_accuracy(self) -> float:
        if not self._fast_outcomes:
            return 0.0
        return sum(self._fast_outcomes) / len(self._fast_outcomes)

    @property
    def calibration_error(self) -> float:
        """Expected Calibration Error (ECE) across confidence bins."""
        if len(self._outcomes) < 5:
            return 0.0

        outcomes = np.array(list(self._outcomes), dtype=float)
        confs = np.array(list(self._confidences))

        bin_edges = np.linspace(0, 1, self._cal_bins + 1)
        ece = 0.0
        for i in range(self._cal_bins):
            mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            bin_acc = outcomes[mask].mean()
            bin_conf = confs[mask].mean()
            ece += (n_bin / len(outcomes)) * abs(bin_acc - bin_conf)
        return float(ece)

    @property
    def learning_trend(self) -> str:
        n = len(self._outcomes)
        if n < 20:
            return "learning"
        half = n // 2
        old_acc = sum(list(self._outcomes)[:half]) / half
        new_acc = sum(list(self._outcomes)[half:]) / (n - half)
        delta = new_acc - old_acc
        if delta > 0.05:
            return "improving"
        if delta < -0.05:
            return "declining"
        return "stable"

    @property
    def skill_level(self) -> str:
        acc = self.accuracy
        if self.total < 10:
            return "novice"
        if acc < 0.5:
            return "learning"
        if acc < 0.8:
            return "competent"
        return "expert"


class SelfModel:
    """
    Metacognitive self-model that tracks competency across domains.

    Feed it outcome observations; query it for self-reports and
    adaptive threshold recommendations.
    """

    def __init__(self, window: int = 100, cal_bins: int = 10):
        self._domains: dict[str, _DomainProfile] = defaultdict(
            lambda: _DomainProfile(window=window, cal_bins=cal_bins)
        )
        self._window = window
        self._cal_bins = cal_bins

    def record(
        self,
        domain: str,
        correct: bool,
        confidence: float,
        route: str = "fast",
    ) -> None:
        """Record one observation for a domain."""
        self._domains[domain].record(correct, confidence, route)

    def introspect(self, domain: str | None = None) -> SelfReport:
        """
        Generate a comprehensive self-report.

        If domain is specified, only that domain is included;
        otherwise all tracked domains are reported.
        """
        domains = (
            {domain: self._domains[domain]}
            if domain and domain in self._domains
            else dict(self._domains)
        )

        competencies: dict[str, CompetencyProfile] = {}
        for name, dp in domains.items():
            competencies[name] = CompetencyProfile(
                domain=name,
                skill_level=dp.skill_level,
                accuracy=dp.accuracy,
                calibration_error=dp.calibration_error,
                fast_path_ratio=dp.fast_path_ratio,
                fast_path_accuracy=dp.fast_path_accuracy,
                learning_trend=dp.learning_trend,
                total_observations=dp.total,
                confidence_mean=dp.confidence_mean,
            )

        strengths = [n for n, cp in competencies.items() if cp.skill_level in ("competent", "expert")]
        weaknesses = [n for n, cp in competencies.items() if cp.skill_level in ("novice", "learning")]

        all_ece = [cp.calibration_error for cp in competencies.values()]
        overall_cal = float(np.mean(all_ece)) if all_ece else 0.0

        if weaknesses and not strengths:
            rec = "Defer most decisions to the brain (LLM); actively practice all domains."
        elif weaknesses:
            rec = f"Handle {', '.join(strengths)} autonomously; defer {', '.join(weaknesses)} to brain."
        elif overall_cal > 0.15:
            rec = "Good accuracy but poor calibration — confidence scores are unreliable."
        else:
            rec = "Operating well across all domains. Consider expanding to new microzones."

        return SelfReport(
            competencies=competencies,
            overall_calibration=overall_cal,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=rec,
        )

    def suggest_thresholds(self, domain: str) -> dict[str, float]:
        """
        Suggest adaptive routing thresholds based on self-assessed competency.

        Expert domains → lower thresholds (trust the fast path more).
        Novice domains → higher thresholds (be cautious, ask the brain).
        """
        if domain not in self._domains:
            return {"threshold_high": 0.95, "threshold_low": 0.5}

        dp = self._domains[domain]
        level = dp.skill_level
        cal_err = dp.calibration_error

        base = {
            "novice":    {"threshold_high": 0.98, "threshold_low": 0.6},
            "learning":  {"threshold_high": 0.92, "threshold_low": 0.5},
            "competent": {"threshold_high": 0.85, "threshold_low": 0.4},
            "expert":    {"threshold_high": 0.75, "threshold_low": 0.3},
        }[level]

        # Poor calibration → widen the slow-path zone
        if cal_err > 0.2:
            base["threshold_high"] = min(base["threshold_high"] + 0.05, 0.99)
            base["threshold_low"] = max(base["threshold_low"] - 0.05, 0.1)

        return base

    @property
    def stats(self) -> dict[str, Any]:
        return {
            domain: {
                "skill": dp.skill_level,
                "accuracy": dp.accuracy,
                "ece": dp.calibration_error,
                "fast_ratio": dp.fast_path_ratio,
                "trend": dp.learning_trend,
                "n": dp.total,
            }
            for domain, dp in self._domains.items()
        }
