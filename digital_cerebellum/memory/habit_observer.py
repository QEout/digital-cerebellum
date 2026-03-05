"""
HabitObserver — extracts temporal behaviour patterns from agent interactions.

Biological basis:
  The suprachiasmatic nucleus (SCN) is the brain's master clock.
  But the cerebellum maintains its OWN timing circuits — Purkinje cells
  encode temporal intervals, and the granule-cell layer acts as a delay
  line / temporal basis set.

  The cerebellum doesn't just react to rhythms — it PREDICTS them.
  A drummer doesn't wait for the metronome click; the cerebellum fires
  motor commands *before* the beat, based on learned temporal patterns.

Digital implementation:
  HabitObserver watches every action the agent takes (via monitor_after_step)
  and extracts three kinds of temporal patterns:

  1. **Daily rhythms** — "user checks email at ~09:00 on weekdays"
  2. **Sequential patterns** — "after checking email, user opens Slack"
  3. **Frequency patterns** — "user deploys code ~2x per week"

  These patterns feed the RhythmEngine for predictive awakening:
  instead of polling every 30 minutes (OpenClaw's cron), the cerebellum
  predicts WHEN the next action will be needed and prepares in advance.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """A single observed agent action with timestamp."""

    timestamp: float = field(default_factory=time.time)
    action: str = ""
    domain: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hour_of_day(self) -> float:
        import datetime
        dt = datetime.datetime.fromtimestamp(self.timestamp)
        return dt.hour + dt.minute / 60.0

    @property
    def weekday(self) -> int:
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp).weekday()


@dataclass
class HabitPattern:
    """An extracted temporal behaviour pattern."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action_pattern: str = ""
    domain: str = ""
    pattern_type: str = "daily"  # "daily" | "sequential" | "frequency"

    # Daily / weekly timing
    hour_mean: float = 0.0
    hour_std: float = 24.0
    weekdays: list[int] = field(default_factory=lambda: list(range(7)))

    # Sequential: what typically precedes this action
    predecessor: str | None = None
    delay_mean_seconds: float = 0.0
    delay_std_seconds: float = 0.0

    # Frequency
    occurrences_per_day: float = 0.0

    # Confidence
    confidence: float = 0.0
    sample_count: int = 0
    last_seen: float = field(default_factory=time.time)

    def matches_time(self, hour: float, weekday: int, tolerance_std: float = 1.5) -> float:
        """How well does a given time match this pattern? Returns 0-1 score."""
        if weekday not in self.weekdays:
            return 0.0

        if self.hour_std < 0.1:
            return 1.0 if abs(hour - self.hour_mean) < 0.5 else 0.0

        diff = abs(hour - self.hour_mean)
        if diff > 12:
            diff = 24 - diff
        z = diff / max(self.hour_std, 0.1)
        if z > tolerance_std:
            return 0.0
        return float(math.exp(-0.5 * z * z)) * self.confidence


@dataclass
class HabitPrediction:
    """A prediction of what the user is likely to do next."""

    action: str
    domain: str
    confidence: float
    pattern: HabitPattern
    reason: str = ""
    preparation_hint: str = ""


class HabitObserver:
    """
    Watches agent actions and extracts temporal patterns.

    Integrates with StepMonitor — every after_step automatically
    feeds into the observer. Patterns are extracted during sleep
    cycles or on demand.
    """

    MAX_RECORDS = 10000
    MIN_SAMPLES_FOR_PATTERN = 3
    SEQUENCE_WINDOW_SECONDS = 1800  # 30 min: actions within this window are "sequential"

    def __init__(self):
        self._records: list[ActionRecord] = []
        self._patterns: dict[str, HabitPattern] = {}
        self._action_clusters: dict[str, list[ActionRecord]] = defaultdict(list)

    def record(
        self,
        action: str,
        domain: str = "",
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record an observed action. Called from monitor_after_step."""
        rec = ActionRecord(
            timestamp=timestamp or time.time(),
            action=action,
            domain=domain,
            success=success,
            metadata=metadata or {},
        )
        self._records.append(rec)
        self._action_clusters[self._normalize_action(action)].append(rec)

        if len(self._records) > self.MAX_RECORDS:
            self._records = self._records[-self.MAX_RECORDS:]

    def extract_patterns(self, min_occurrences: int | None = None) -> list[HabitPattern]:
        """
        Extract temporal patterns from all recorded actions.

        This is the core pattern-recognition step — typically called
        during a sleep cycle, but can be called on demand.
        """
        min_occ = min_occurrences or self.MIN_SAMPLES_FOR_PATTERN
        new_patterns: list[HabitPattern] = []

        for action_key, records in self._action_clusters.items():
            if len(records) < min_occ:
                continue

            daily = self._extract_daily_pattern(action_key, records)
            if daily:
                new_patterns.append(daily)

            seq = self._extract_sequential_patterns(action_key, records)
            new_patterns.extend(seq)

        for p in new_patterns:
            self._patterns[p.id] = p

        return new_patterns

    def get_predictions(
        self,
        current_time: float | None = None,
        top_k: int = 5,
    ) -> list[HabitPrediction]:
        """
        Predict what the user is likely to do next.

        Based on current time and known patterns, returns ranked
        predictions with confidence scores.
        """
        import datetime

        now = current_time or time.time()
        dt = datetime.datetime.fromtimestamp(now)
        hour = dt.hour + dt.minute / 60.0
        weekday = dt.weekday()

        predictions: list[HabitPrediction] = []

        for pattern in self._patterns.values():
            if pattern.pattern_type == "daily":
                score = pattern.matches_time(hour, weekday)
                if score > 0.1:
                    predictions.append(HabitPrediction(
                        action=pattern.action_pattern,
                        domain=pattern.domain,
                        confidence=score,
                        pattern=pattern,
                        reason=f"You usually do this at ~{pattern.hour_mean:.0f}:00 "
                               f"({pattern.sample_count} observations)",
                        preparation_hint=f"Prepare for '{pattern.action_pattern}'?",
                    ))

            elif pattern.pattern_type == "sequential":
                recent = self._get_recent_action(now, window_seconds=600)
                if recent and self._normalize_action(recent.action) == pattern.predecessor:
                    elapsed = now - recent.timestamp
                    expected = pattern.delay_mean_seconds
                    if elapsed < expected + 2 * pattern.delay_std_seconds:
                        score = pattern.confidence * 0.8
                        predictions.append(HabitPrediction(
                            action=pattern.action_pattern,
                            domain=pattern.domain,
                            confidence=score,
                            pattern=pattern,
                            reason=f"After '{pattern.predecessor}', you usually do this "
                                   f"within {pattern.delay_mean_seconds:.0f}s",
                            preparation_hint=f"Following up with '{pattern.action_pattern}'?",
                        ))

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]

    def get_suggestions(
        self,
        current_time: float | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get proactive suggestions formatted for the agent.

        Returns dicts ready to be injected into monitor_before_step context.
        """
        predictions = self.get_predictions(current_time, top_k=top_k)
        return [
            {
                "type": "habit_suggestion",
                "action": p.action,
                "domain": p.domain,
                "confidence": round(p.confidence, 3),
                "reason": p.reason,
                "hint": p.preparation_hint,
                "pattern_type": p.pattern.pattern_type,
                "pattern_id": p.pattern.id,
            }
            for p in predictions
        ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "action_clusters": len(self._action_clusters),
            "patterns": len(self._patterns),
            "patterns_by_type": self._count_by_type(),
        }

    # ------------------------------------------------------------------
    # Pattern extraction internals
    # ------------------------------------------------------------------

    def _extract_daily_pattern(
        self, action_key: str, records: list[ActionRecord],
    ) -> HabitPattern | None:
        """Extract a daily time-of-day pattern from records."""
        hours = [r.hour_of_day for r in records]
        weekdays_seen: dict[int, int] = defaultdict(int)
        for r in records:
            weekdays_seen[r.weekday] += 1

        hour_mean = float(np.mean(hours))
        hour_std = float(np.std(hours))

        if hour_std > 6.0:
            return None

        active_weekdays = [
            wd for wd, count in weekdays_seen.items()
            if count >= max(1, len(records) * 0.1)
        ]

        total_days = self._count_unique_days(records)
        freq = len(records) / max(total_days, 1)

        confidence = min(1.0, len(records) / 10.0) * max(0.3, 1.0 - hour_std / 6.0)

        domain = records[0].domain if records else ""

        return HabitPattern(
            action_pattern=action_key,
            domain=domain,
            pattern_type="daily",
            hour_mean=hour_mean,
            hour_std=hour_std,
            weekdays=sorted(active_weekdays),
            occurrences_per_day=freq,
            confidence=confidence,
            sample_count=len(records),
            last_seen=records[-1].timestamp,
        )

    def _extract_sequential_patterns(
        self, action_key: str, records: list[ActionRecord],
    ) -> list[HabitPattern]:
        """Find actions that typically precede this one."""
        predecessor_delays: dict[str, list[float]] = defaultdict(list)

        for rec in records:
            preceding = self._find_preceding_action(rec.timestamp)
            if preceding is None:
                continue
            pred_key = self._normalize_action(preceding.action)
            if pred_key == action_key:
                continue
            delay = rec.timestamp - preceding.timestamp
            if 0 < delay < self.SEQUENCE_WINDOW_SECONDS:
                predecessor_delays[pred_key].append(delay)

        patterns: list[HabitPattern] = []
        for pred_key, delays in predecessor_delays.items():
            if len(delays) < self.MIN_SAMPLES_FOR_PATTERN:
                continue

            delay_mean = float(np.mean(delays))
            delay_std = float(np.std(delays))
            confidence = min(1.0, len(delays) / 8.0) * max(0.3, 1.0 - delay_std / delay_mean if delay_mean > 0 else 0.5)

            domain = records[0].domain if records else ""

            patterns.append(HabitPattern(
                action_pattern=action_key,
                domain=domain,
                pattern_type="sequential",
                predecessor=pred_key,
                delay_mean_seconds=delay_mean,
                delay_std_seconds=delay_std,
                confidence=confidence,
                sample_count=len(delays),
                last_seen=records[-1].timestamp,
            ))

        return patterns

    def _find_preceding_action(self, timestamp: float) -> ActionRecord | None:
        """Find the most recent action before the given timestamp."""
        best: ActionRecord | None = None
        for rec in reversed(self._records):
            if rec.timestamp < timestamp:
                best = rec
                break
        return best

    def _get_recent_action(self, now: float, window_seconds: float) -> ActionRecord | None:
        """Get the most recent action within the time window."""
        for rec in reversed(self._records):
            if now - rec.timestamp <= window_seconds:
                return rec
            if now - rec.timestamp > window_seconds:
                break
        return None

    @staticmethod
    def _normalize_action(action: str) -> str:
        """Normalize action text for clustering. Simple keyword extraction."""
        action = action.lower().strip()
        if len(action) > 100:
            action = action[:100]
        for prefix in ("use ", "run ", "execute ", "call ", "invoke "):
            if action.startswith(prefix):
                action = action[len(prefix):]
        return action

    @staticmethod
    def _count_unique_days(records: list[ActionRecord]) -> int:
        import datetime
        days = set()
        for r in records:
            dt = datetime.datetime.fromtimestamp(r.timestamp)
            days.add((dt.year, dt.month, dt.day))
        return len(days)

    def _count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for p in self._patterns.values():
            counts[p.pattern_type] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = ".digital-cerebellum/habits") -> None:
        """Persist records and patterns to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        records_data = [
            {
                "timestamp": r.timestamp,
                "action": r.action,
                "domain": r.domain,
                "success": r.success,
                "metadata": r.metadata,
            }
            for r in self._records[-self.MAX_RECORDS:]
        ]

        patterns_data = [
            {
                "id": p.id,
                "action_pattern": p.action_pattern,
                "domain": p.domain,
                "pattern_type": p.pattern_type,
                "hour_mean": p.hour_mean,
                "hour_std": p.hour_std,
                "weekdays": p.weekdays,
                "predecessor": p.predecessor,
                "delay_mean_seconds": p.delay_mean_seconds,
                "delay_std_seconds": p.delay_std_seconds,
                "occurrences_per_day": p.occurrences_per_day,
                "confidence": p.confidence,
                "sample_count": p.sample_count,
                "last_seen": p.last_seen,
            }
            for p in self._patterns.values()
        ]

        with open(path / "habits.json", "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "records": records_data,
                "patterns": patterns_data,
            }, f, indent=2, ensure_ascii=False)

        log.info("Saved %d records, %d patterns to %s",
                 len(records_data), len(patterns_data), path)

    def load(self, path: str | Path = ".digital-cerebellum/habits") -> int:
        """Load records and patterns from disk. Returns total items loaded."""
        path = Path(path)
        json_path = path / "habits.json"

        if not json_path.exists():
            return 0

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        self._records.clear()
        self._action_clusters.clear()
        self._patterns.clear()

        for rd in data.get("records", []):
            rec = ActionRecord(
                timestamp=rd["timestamp"],
                action=rd["action"],
                domain=rd.get("domain", ""),
                success=rd.get("success", True),
                metadata=rd.get("metadata", {}),
            )
            self._records.append(rec)
            self._action_clusters[self._normalize_action(rec.action)].append(rec)

        for pd in data.get("patterns", []):
            pattern = HabitPattern(
                id=pd["id"],
                action_pattern=pd["action_pattern"],
                domain=pd.get("domain", ""),
                pattern_type=pd["pattern_type"],
                hour_mean=pd.get("hour_mean", 0.0),
                hour_std=pd.get("hour_std", 24.0),
                weekdays=pd.get("weekdays", list(range(7))),
                predecessor=pd.get("predecessor"),
                delay_mean_seconds=pd.get("delay_mean_seconds", 0.0),
                delay_std_seconds=pd.get("delay_std_seconds", 0.0),
                occurrences_per_day=pd.get("occurrences_per_day", 0.0),
                confidence=pd.get("confidence", 0.0),
                sample_count=pd.get("sample_count", 0),
                last_seen=pd.get("last_seen", 0.0),
            )
            self._patterns[pattern.id] = pattern

        total = len(self._records) + len(self._patterns)
        log.info("Loaded %d records, %d patterns from %s",
                 len(self._records), len(self._patterns), path)
        return total
