"""Tests for HabitObserver and RhythmEngine."""

from __future__ import annotations

import datetime
import time

import pytest

from digital_cerebellum.memory.habit_observer import (
    ActionRecord,
    HabitObserver,
    HabitPattern,
)
from digital_cerebellum.rhythm.engine import RhythmEngine


# ── HabitObserver basics ──────────────────────────────────────────────


class TestHabitObserverRecord:
    def test_record_stores_action(self):
        obs = HabitObserver()
        obs.record("check email", domain="email")
        assert len(obs._records) == 1
        assert obs._records[0].action == "check email"
        assert obs._records[0].domain == "email"

    def test_record_clusters_by_normalized_action(self):
        obs = HabitObserver()
        obs.record("check email")
        obs.record("check email")
        obs.record("open slack")
        assert len(obs._action_clusters) == 2
        assert len(obs._action_clusters["check email"]) == 2

    def test_record_normalizes_action(self):
        obs = HabitObserver()
        obs.record("Use check_email tool")
        obs.record("use check_email tool")
        assert len(obs._action_clusters["check_email tool"]) == 2

    def test_max_records_enforced(self):
        obs = HabitObserver()
        obs.MAX_RECORDS = 10
        for i in range(20):
            obs.record(f"action_{i}")
        assert len(obs._records) <= 10


# ── Pattern extraction ────────────────────────────────────────────────


class TestDailyPatternExtraction:
    def _make_daily_observations(
        self, obs: HabitObserver, action: str, hour: float, count: int,
    ):
        """Create `count` observations at roughly the same hour across days."""
        import random
        base = datetime.datetime(2026, 3, 1, int(hour), int((hour % 1) * 60))
        for day in range(count):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            ts += random.uniform(-600, 600)  # ±10min jitter
            obs.record(action, domain="email", timestamp=ts)

    def test_extracts_daily_pattern(self):
        obs = HabitObserver()
        self._make_daily_observations(obs, "check email", hour=9.0, count=7)
        patterns = obs.extract_patterns()
        daily = [p for p in patterns if p.pattern_type == "daily"]
        assert len(daily) >= 1
        p = daily[0]
        assert abs(p.hour_mean - 9.0) < 1.0
        assert p.confidence > 0.3
        assert p.sample_count == 7

    def test_no_pattern_from_random_times(self):
        obs = HabitObserver()
        import random
        for i in range(10):
            hour = random.uniform(0, 24)
            ts = datetime.datetime(2026, 3, 1 + i, int(hour), 0).timestamp()
            obs.record("random task", timestamp=ts)
        patterns = obs.extract_patterns()
        daily = [p for p in patterns if p.pattern_type == "daily"]
        for p in daily:
            assert p.confidence < 0.5 or p.hour_std > 4.0

    def test_min_samples_required(self):
        obs = HabitObserver()
        obs.record("rare task", timestamp=time.time())
        obs.record("rare task", timestamp=time.time() + 86400)
        patterns = obs.extract_patterns(min_occurrences=3)
        assert len(patterns) == 0


class TestSequentialPatternExtraction:
    def test_extracts_sequential_pattern(self):
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0).timestamp()
        for day in range(5):
            t = base + day * 86400
            obs.record("check email", domain="email", timestamp=t)
            obs.record("open slack", domain="chat", timestamp=t + 600)  # 10min later

        patterns = obs.extract_patterns()
        seq = [p for p in patterns if p.pattern_type == "sequential"]
        assert len(seq) >= 1
        slack_seq = [p for p in seq if "slack" in p.action_pattern]
        assert len(slack_seq) >= 1
        p = slack_seq[0]
        assert p.predecessor == "check email"
        assert 300 < p.delay_mean_seconds < 900


# ── Predictions ───────────────────────────────────────────────────────


class TestPredictions:
    def test_predicts_daily_action(self):
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0)
        for day in range(10):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            obs.record("check email", domain="email", timestamp=ts)

        obs.extract_patterns()

        query_time = datetime.datetime(2026, 3, 15, 9, 5).timestamp()
        preds = obs.get_predictions(current_time=query_time)
        assert len(preds) >= 1
        assert preds[0].action == "check email"
        assert preds[0].confidence > 0.3

    def test_no_prediction_at_wrong_time(self):
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0)
        for day in range(10):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            obs.record("check email", domain="email", timestamp=ts)

        obs.extract_patterns()

        query_time = datetime.datetime(2026, 3, 15, 22, 0).timestamp()
        preds = obs.get_predictions(current_time=query_time)
        email_preds = [p for p in preds if p.action == "check email"]
        if email_preds:
            assert email_preds[0].confidence < 0.2

    def test_get_suggestions_format(self):
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0)
        for day in range(5):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            obs.record("check email", domain="email", timestamp=ts)

        obs.extract_patterns()
        query_time = datetime.datetime(2026, 3, 10, 9, 0).timestamp()
        suggestions = obs.get_suggestions(current_time=query_time)
        if suggestions:
            s = suggestions[0]
            assert "type" in s
            assert "action" in s
            assert "confidence" in s
            assert s["type"] == "habit_suggestion"


# ── Persistence ───────────────────────────────────────────────────────


class TestHabitPersistence:
    def test_save_and_load(self, tmp_path):
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0)
        for day in range(5):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            obs.record("check email", domain="email", timestamp=ts)
        obs.extract_patterns()

        save_path = tmp_path / "habits"
        obs.save(save_path)

        obs2 = HabitObserver()
        loaded = obs2.load(save_path)
        assert loaded > 0
        assert len(obs2._records) == 5
        assert len(obs2._patterns) >= 1

    def test_load_nonexistent(self, tmp_path):
        obs = HabitObserver()
        loaded = obs.load(tmp_path / "nonexistent")
        assert loaded == 0


# ── RhythmEngine ──────────────────────────────────────────────────────


class TestRhythmEngine:
    def _build_rhythm(self) -> tuple[HabitObserver, RhythmEngine]:
        obs = HabitObserver()
        base = datetime.datetime(2026, 3, 1, 9, 0)
        for day in range(10):
            ts = (base + datetime.timedelta(days=day)).timestamp()
            obs.record("check email", domain="email", timestamp=ts)
        obs.extract_patterns()
        return obs, RhythmEngine(obs)

    def test_suggestions_at_matching_time(self):
        obs, rhythm = self._build_rhythm()
        now = datetime.datetime(2026, 3, 15, 8, 57).timestamp()
        suggestions = rhythm.get_proactive_suggestions(current_time=now)
        assert len(suggestions) >= 1

    def test_next_wakeup_adapts(self):
        obs, rhythm = self._build_rhythm()
        now = datetime.datetime(2026, 3, 15, 3, 0).timestamp()
        next_wake = rhythm.get_next_wakeup(current_time=now)
        assert next_wake > now
        interval_1 = next_wake - now

        rhythm.record_activity()
        rhythm.record_activity()
        rhythm.record_activity()
        next_wake_2 = rhythm.get_next_wakeup(current_time=now)
        interval_2 = next_wake_2 - now
        assert interval_2 <= interval_1  # more active → shorter interval

    def test_state_report(self):
        obs, rhythm = self._build_rhythm()
        state = rhythm.state
        assert "mode" in state
        assert "activity_level" in state
        assert "patterns_known" in state
        assert state["patterns_known"] >= 1


# ── Integration with StepMonitor ──────────────────────────────────────


class TestStepMonitorHabitIntegration:
    def test_after_step_records_habit(self):
        from digital_cerebellum.monitor import StepMonitor

        monitor = StepMonitor()
        monitor.before_step(action="check email", state="desktop idle")
        monitor.after_step(outcome="inbox opened with 5 new emails", success=True)

        assert len(monitor.habit_observer._records) == 1
        rec = monitor.habit_observer._records[0]
        assert "check email" in rec.action

    def test_multiple_steps_build_cluster(self):
        from digital_cerebellum.monitor import StepMonitor

        monitor = StepMonitor()
        for _ in range(5):
            monitor.before_step(action="deploy staging", state="terminal")
            monitor.after_step(outcome="deployed successfully", success=True)

        assert len(monitor.habit_observer._records) == 5
        clusters = monitor.habit_observer._action_clusters
        assert len(clusters) >= 1
