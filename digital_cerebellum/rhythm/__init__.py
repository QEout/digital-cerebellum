"""
Rhythm system — predictive awakening instead of polling.

The biological alternative to cron jobs: event-driven + predictive timing,
adaptive check intervals based on activity level and learned patterns.
"""

from digital_cerebellum.rhythm.engine import RhythmEngine, WakeupEvent, RhythmState

__all__ = ["RhythmEngine", "WakeupEvent", "RhythmState"]
