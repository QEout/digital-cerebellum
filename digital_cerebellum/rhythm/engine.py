"""
Rhythm Engine — predictive awakening, not polling.

Biological basis:
  OpenClaw polls every 30 minutes (a cron job).
  Biological organisms don't poll — they PREDICT.

  The suprachiasmatic nucleus maintains circadian rhythms.
  The cerebellum maintains sub-second to minute-scale timing.
  Together they create an event-driven + predictive system:

  - Event-driven: file change → instant response (no polling delay)
  - Predictive: "user usually checks email at 9:00" → prepare at 8:55
  - Adaptive: busy period → check more often; quiet period → check less

Digital implementation:
  RhythmEngine sits on top of HabitObserver and provides:
  1. Proactive suggestions — "you usually do X around now"
  2. Next-wakeup prediction — when to next check (not fixed interval)
  3. Preparation signals — pre-compute / pre-fetch before the user asks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from digital_cerebellum.memory.habit_observer import HabitObserver

log = logging.getLogger(__name__)


@dataclass
class WakeupEvent:
    """A predicted future event the system should prepare for."""

    timestamp: float
    action: str
    domain: str
    confidence: float
    preparation: str
    pattern_id: str = ""


@dataclass
class RhythmState:
    """Current state of the rhythm engine."""

    mode: str = "normal"  # "active" | "normal" | "quiet"
    check_interval_seconds: float = 300.0
    last_check: float = field(default_factory=time.time)
    pending_wakeups: list[WakeupEvent] = field(default_factory=list)


class RhythmEngine:
    """
    Predictive awakening system.

    Instead of polling at fixed intervals, predicts when the next
    action will be needed and schedules preparation accordingly.
    """

    DEFAULT_INTERVAL = 300.0
    MIN_INTERVAL = 30.0
    MAX_INTERVAL = 7200.0
    ACTIVITY_DECAY = 0.95
    PREPARATION_LEAD_MINUTES = 5

    def __init__(self, habit_observer: HabitObserver):
        self._habits = habit_observer
        self._state = RhythmState()
        self._activity_level: float = 0.5
        self._last_activity: float = time.time()

    def get_proactive_suggestions(
        self,
        current_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get suggestions for what the agent should prepare NOW.

        Combines habit predictions with current activity level.
        """
        now = current_time or time.time()
        self._update_activity_level(now)
        self._state.last_check = now

        suggestions = self._habits.get_suggestions(now)

        upcoming = self._get_upcoming_wakeups(now)
        for wu in upcoming:
            if wu.confidence > 0.3:
                suggestions.append({
                    "type": "rhythm_preparation",
                    "action": wu.action,
                    "domain": wu.domain,
                    "confidence": round(wu.confidence, 3),
                    "reason": f"Predicted in ~{(wu.timestamp - now) / 60:.0f} minutes",
                    "hint": wu.preparation,
                    "pattern_type": "predictive",
                    "pattern_id": wu.pattern_id,
                })

        suggestions.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        return suggestions

    def get_next_wakeup(self, current_time: float | None = None) -> float:
        """
        When should the system next check for work?

        Returns a timestamp. Adapts based on activity level,
        known patterns, and time of day.
        """
        now = current_time or time.time()
        self._update_activity_level(now)

        interval = self.DEFAULT_INTERVAL
        if self._activity_level > 0.7:
            interval = self.MIN_INTERVAL
            self._state.mode = "active"
        elif self._activity_level < 0.2:
            interval = self.MAX_INTERVAL
            self._state.mode = "quiet"
        else:
            ratio = 1.0 - self._activity_level
            interval = self.MIN_INTERVAL + ratio * (self.MAX_INTERVAL - self.MIN_INTERVAL)
            self._state.mode = "normal"

        predictions = self._habits.get_predictions(now, top_k=3)
        for pred in predictions:
            if pred.confidence > 0.5:
                import datetime
                dt = datetime.datetime.fromtimestamp(now)
                pred_hour = pred.pattern.hour_mean
                current_hour = dt.hour + dt.minute / 60.0

                hours_until = pred_hour - current_hour
                if hours_until < 0:
                    hours_until += 24
                if hours_until > 12:
                    continue

                seconds_until = hours_until * 3600 - self.PREPARATION_LEAD_MINUTES * 60
                if 0 < seconds_until < interval:
                    interval = max(self.MIN_INTERVAL, seconds_until)

        next_time = now + interval
        self._state.check_interval_seconds = interval
        return next_time

    def record_activity(self) -> None:
        """Call this after every agent action to update activity level."""
        self._activity_level = min(1.0, self._activity_level + 0.2)
        self._last_activity = time.time()

    @property
    def state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode,
            "check_interval_seconds": round(self._state.check_interval_seconds, 1),
            "activity_level": round(self._activity_level, 3),
            "pending_wakeups": len(self._state.pending_wakeups),
            "patterns_known": len(self._habits._patterns),
        }

    def _update_activity_level(self, now: float) -> None:
        elapsed_minutes = (now - self._last_activity) / 60.0
        if elapsed_minutes > 0:
            self._activity_level *= self.ACTIVITY_DECAY ** elapsed_minutes
            self._activity_level = max(0.0, self._activity_level)

    def _get_upcoming_wakeups(self, now: float) -> list[WakeupEvent]:
        import datetime

        wakeups: list[WakeupEvent] = []
        predictions = self._habits.get_predictions(now, top_k=5)

        for pred in predictions:
            if pred.pattern.pattern_type == "daily" and pred.confidence > 0.3:
                dt = datetime.datetime.fromtimestamp(now)
                pred_hour = pred.pattern.hour_mean
                current_hour = dt.hour + dt.minute / 60.0

                diff = pred_hour - current_hour
                if -0.5 < diff < self.PREPARATION_LEAD_MINUTES / 60.0:
                    target = now + diff * 3600
                    wakeups.append(WakeupEvent(
                        timestamp=target,
                        action=pred.action,
                        domain=pred.domain,
                        confidence=pred.confidence,
                        preparation=pred.preparation_hint,
                        pattern_id=pred.pattern.id,
                    ))

        self._state.pending_wakeups = wakeups
        return wakeups
