"""
Curiosity Drive — intrinsic motivation from learning progress.

Biological basis:
  Dopaminergic neurons fire not just for rewards but for NOVELTY and
  PREDICTION ERRORS.  Crucially, the drive is strongest when errors
  are large AND decreasing — the system is actively learning something
  new.  Pure noise (high error, no decrease) should NOT be rewarding
  ("noisy TV problem").

  The cerebellum provides the key signal: prediction error trajectories
  per domain.  Learning progress = negative slope of recent errors.

  References:
    - Schmidhuber (1991): Curiosity as learning progress
    - Pathak et al. (2017): Curiosity-driven exploration via self-supervision
    - Colas et al. (2022): Intrinsic motivation and autotelic learning
    - CDE (2025, arXiv 2509.09675): Curiosity for LLM reasoning
    - LPM (2025): Learning Progress Monitoring — reward improvements, not error

Digital implementation:
  1. Track per-domain error trajectories (SPE, RPE sliding windows).
  2. Compute learning_progress = mean(old_errors) - mean(new_errors).
  3. Compute novelty = distance of current input from memory.
  4. intrinsic_reward = learning_progress * novelty_bonus.
  5. Recommend: explore (learnable novelty), exploit (known), or
     abandon (unlearnable noise).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CuriositySignal:
    """Output of the curiosity drive for a single evaluation."""

    domain: str
    novelty: float               # 0-1, how novel is the current input
    learning_progress: float     # positive = improving, negative = forgetting
    intrinsic_reward: float      # combined curiosity score
    recommendation: str          # "explore" | "exploit" | "abandon"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_interesting(self) -> bool:
        return self.recommendation == "explore"


class _DomainTracker:
    """Tracks error trajectory for one domain (microzone)."""

    def __init__(self, window: int = 50, progress_window: int = 15):
        self.errors: deque[float] = deque(maxlen=window)
        self._progress_window = progress_window
        self.observation_count: int = 0

    def record(self, error: float) -> None:
        self.errors.append(error)
        self.observation_count += 1

    @property
    def learning_progress(self) -> float:
        """
        Learning progress = mean(older half) - mean(newer half).

        Positive → errors are decreasing → we're learning.
        Negative → errors are increasing → forgetting or noise.
        """
        n = len(self.errors)
        pw = self._progress_window
        if n < pw * 2:
            return 0.0
        arr = np.array(self.errors)
        old_mean = float(np.abs(arr[-pw * 2:-pw]).mean())
        new_mean = float(np.abs(arr[-pw:]).mean())
        return old_mean - new_mean

    @property
    def mean_error(self) -> float:
        if not self.errors:
            return 0.0
        return float(np.mean(np.abs(list(self.errors))))

    @property
    def error_variance(self) -> float:
        if len(self.errors) < 3:
            return 0.0
        return float(np.var(list(self.errors)))


class CuriosityDrive:
    """
    Generates intrinsic motivation signals based on learning progress.

    High error + high learning progress → "This is fascinating, keep exploring"
    High error + zero progress          → "This is noise, don't waste resources"
    Low error + stable                  → "I've mastered this, time to move on"
    """

    def __init__(
        self,
        progress_window: int = 15,
        novelty_decay: float = 0.98,
        explore_threshold: float = 0.1,
        abandon_threshold: float = -0.05,
    ):
        self._trackers: dict[str, _DomainTracker] = defaultdict(
            lambda: _DomainTracker(progress_window=progress_window)
        )
        self._novelty_decay = novelty_decay
        self._explore_threshold = explore_threshold
        self._abandon_threshold = abandon_threshold
        self._seen_fingerprints: deque[np.ndarray] = deque(maxlen=200)

    def record_error(self, domain: str, error: float) -> None:
        """Record a prediction error for a domain."""
        self._trackers[domain].record(error)

    def compute_novelty(self, feature_vec: np.ndarray) -> float:
        """
        Compute novelty of the current input relative to recently seen inputs.

        Uses mean cosine distance to recent feature vectors.
        High distance → novel; low distance → familiar.
        """
        if not self._seen_fingerprints:
            self._seen_fingerprints.append(feature_vec.copy())
            return 1.0

        fn = np.linalg.norm(feature_vec)
        if fn < 1e-9:
            return 0.0

        distances = []
        for stored in self._seen_fingerprints:
            sn = np.linalg.norm(stored)
            if sn < 1e-9:
                distances.append(1.0)
                continue
            cos_sim = np.dot(feature_vec, stored) / (fn * sn)
            distances.append(1.0 - cos_sim)

        self._seen_fingerprints.append(feature_vec.copy())

        mean_dist = float(np.mean(distances))
        return float(np.clip(mean_dist, 0.0, 1.0))

    def assess(
        self,
        domain: str,
        error: float,
        feature_vec: np.ndarray | None = None,
    ) -> CuriositySignal:
        """
        Assess the curiosity signal for the current evaluation.

        Parameters
        ----------
        domain : microzone name
        error : current prediction error (SPE or combined)
        feature_vec : optional feature vector for novelty computation
        """
        self.record_error(domain, error)
        tracker = self._trackers[domain]

        lp = tracker.learning_progress
        novelty = self.compute_novelty(feature_vec) if feature_vec is not None else 0.5

        # Intrinsic reward: learning progress scaled by novelty
        # High novelty amplifies the reward; familiar domains get a smaller bonus
        novelty_bonus = 0.5 + 0.5 * novelty
        intrinsic_reward = lp * novelty_bonus

        # Classification
        if lp > self._explore_threshold and error > 0.1:
            recommendation = "explore"
        elif lp < self._abandon_threshold and error > 0.3:
            recommendation = "abandon"
        else:
            recommendation = "exploit"

        return CuriositySignal(
            domain=domain,
            novelty=novelty,
            learning_progress=lp,
            intrinsic_reward=float(intrinsic_reward),
            recommendation=recommendation,
            details={
                "mean_error": tracker.mean_error,
                "error_variance": tracker.error_variance,
                "observation_count": tracker.observation_count,
                "novelty_bonus": float(novelty_bonus),
            },
        )

    def get_exploration_ranking(self) -> list[tuple[str, float]]:
        """
        Rank domains by learning potential (curiosity).

        Returns a list of (domain, intrinsic_reward) sorted descending.
        Useful for the DigitalBrain to decide which tools to practice.
        """
        ranking = []
        for domain, tracker in self._trackers.items():
            lp = tracker.learning_progress
            err = tracker.mean_error
            score = lp * (0.5 + 0.5 * min(err, 1.0))
            ranking.append((domain, float(score)))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_exploration_requests(self) -> list[dict[str, Any]]:
        """
        Generate active exploration requests.

        Instead of passively reporting curiosity, this method produces
        actionable requests that the brain can act on: "Try domain X
        with pattern Y to learn faster."

        Biological basis: dopaminergic signals don't just report novelty,
        they DRIVE exploratory behavior.
        """
        requests = []
        for domain, tracker in self._trackers.items():
            lp = tracker.learning_progress
            err = tracker.mean_error

            if lp > self._explore_threshold and err > 0.1:
                requests.append({
                    "domain": domain,
                    "action": "explore",
                    "urgency": float(lp * err),
                    "reason": f"Active learning: progress={lp:.3f}, error={err:.3f}",
                })
            elif err > 0.4 and tracker.observation_count < 20:
                requests.append({
                    "domain": domain,
                    "action": "practice",
                    "urgency": float(err * 0.5),
                    "reason": f"Needs practice: error={err:.3f}, observations={tracker.observation_count}",
                })

        requests.sort(key=lambda r: r["urgency"], reverse=True)
        return requests

    @property
    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for domain, tracker in self._trackers.items():
            result[domain] = {
                "observations": tracker.observation_count,
                "mean_error": tracker.mean_error,
                "learning_progress": tracker.learning_progress,
                "error_variance": tracker.error_variance,
            }
        return result
