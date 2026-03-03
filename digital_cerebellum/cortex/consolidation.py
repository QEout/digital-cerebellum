"""
Task Consolidation Pipeline — the cerebellum's "graduation" mechanism.

Biology (Boven et al. 2024, Nature Communications):
    Cerebellum rapidly learns a new task → gradually transfers knowledge
    to the cortex → cortex consolidates → cerebellum frees capacity.

Digital equivalent:
    Prediction engine learns from LLM → when a task pattern matures
    (high accuracy, many repetitions, stable error) → the pattern is
    "graduated" and the engine becomes more confident on that class,
    eventually handling it entirely on the fast path.

The pipeline tracks per-microzone task maturity and manages the
graduation lifecycle:

    Stage 0: Unfamiliar  — all slow path
    Stage 1: Accumulating — shadow execution, learning
    Stage 2: Shadow       — cerebellum predicts alongside LLM
    Stage 3: Graduated    — fast path only
    Stage 4: Crystallised — pattern compiled to rule (future)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TaskPattern:
    """Tracks a recurring task pattern's maturity."""
    pattern_id: str
    microzone: str
    description: str = ""
    stage: int = 0
    total_seen: int = 0
    fast_correct: int = 0
    fast_incorrect: int = 0
    slow_total: int = 0
    created_at: float = field(default_factory=time.time)
    graduated_at: float | None = None
    last_seen: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        total = self.fast_correct + self.fast_incorrect
        return self.fast_correct / total if total > 0 else 0.0

    @property
    def fast_total(self) -> int:
        return self.fast_correct + self.fast_incorrect


class ConsolidationPipeline:
    """
    Manages task pattern maturity and graduation across microzones.

    Graduation thresholds (configurable):
        - ACCUMULATE_MIN: minimum observations before shadow stage
        - SHADOW_ACCURACY: minimum accuracy in shadow to graduate
        - SHADOW_MIN_FAST: minimum fast-path attempts before graduation
        - DEGRADE_ERROR_RATE: error rate that triggers demotion
    """

    ACCUMULATE_MIN = 5
    SHADOW_ACCURACY = 0.90
    SHADOW_MIN_FAST = 10
    DEGRADE_ERROR_RATE = 0.20

    def __init__(self):
        self._patterns: dict[str, TaskPattern] = {}
        self._history: list[dict[str, Any]] = []

    def get_or_create(self, pattern_id: str, microzone: str,
                      description: str = "") -> TaskPattern:
        if pattern_id not in self._patterns:
            self._patterns[pattern_id] = TaskPattern(
                pattern_id=pattern_id,
                microzone=microzone,
                description=description,
            )
        return self._patterns[pattern_id]

    def record_observation(
        self,
        pattern_id: str,
        microzone: str,
        was_fast_path: bool,
        was_correct: bool | None = None,
        description: str = "",
    ):
        """Record one observation and potentially advance the stage."""
        p = self.get_or_create(pattern_id, microzone, description)
        p.total_seen += 1
        p.last_seen = time.time()

        if was_fast_path:
            if was_correct is True:
                p.fast_correct += 1
            elif was_correct is False:
                p.fast_incorrect += 1
        else:
            p.slow_total += 1

        old_stage = p.stage
        self._evaluate_stage(p)

        if p.stage != old_stage:
            self._history.append({
                "pattern_id": pattern_id,
                "microzone": microzone,
                "old_stage": old_stage,
                "new_stage": p.stage,
                "timestamp": time.time(),
                "accuracy": p.accuracy,
                "total_seen": p.total_seen,
            })
            log.info(
                "Pattern '%s' (%s): stage %d → %d  "
                "(seen=%d, fast_acc=%.1f%%)",
                pattern_id, microzone, old_stage, p.stage,
                p.total_seen, p.accuracy * 100,
            )

    def _evaluate_stage(self, p: TaskPattern):
        """State machine for task maturity."""
        if p.stage == 0 and p.total_seen >= self.ACCUMULATE_MIN:
            p.stage = 1

        if p.stage == 1 and p.total_seen >= self.ACCUMULATE_MIN * 2:
            p.stage = 2

        if p.stage == 2:
            if (p.fast_total >= self.SHADOW_MIN_FAST
                    and p.accuracy >= self.SHADOW_ACCURACY):
                p.stage = 3
                p.graduated_at = time.time()
            elif p.fast_total >= self.SHADOW_MIN_FAST and p.accuracy < 0.5:
                p.stage = 1

        if p.stage == 3:
            recent_total = p.fast_correct + p.fast_incorrect
            if recent_total > 5 and p.accuracy < (1.0 - self.DEGRADE_ERROR_RATE):
                p.stage = 2
                p.graduated_at = None
                log.warning(
                    "Pattern '%s' DEGRADED: accuracy dropped to %.1f%%",
                    p.pattern_id, p.accuracy * 100,
                )

    def get_stage(self, pattern_id: str) -> int:
        """Get the current maturity stage for a pattern. 0 if unknown."""
        if pattern_id in self._patterns:
            return self._patterns[pattern_id].stage
        return 0

    def is_graduated(self, pattern_id: str) -> bool:
        return self.get_stage(pattern_id) >= 3

    @property
    def stats(self) -> dict[str, Any]:
        by_stage = {i: 0 for i in range(5)}
        by_microzone: dict[str, int] = {}
        for p in self._patterns.values():
            by_stage[p.stage] = by_stage.get(p.stage, 0) + 1
            by_microzone[p.microzone] = by_microzone.get(p.microzone, 0) + 1
        return {
            "total_patterns": len(self._patterns),
            "by_stage": by_stage,
            "by_microzone": by_microzone,
            "graduated": sum(1 for p in self._patterns.values() if p.stage >= 3),
            "transitions": len(self._history),
        }

    def graduated_patterns(self, microzone: str | None = None) -> list[TaskPattern]:
        """Return all graduated patterns, optionally filtered by microzone."""
        return [
            p for p in self._patterns.values()
            if p.stage >= 3
            and (microzone is None or p.microzone == microzone)
        ]
