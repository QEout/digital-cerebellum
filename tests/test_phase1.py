"""
Phase 1 component tests: Sleep Cycle, Task Consolidation, full TPE/RPE.

Run with:  pytest tests/test_phase1.py -v
"""

import time
import numpy as np
import pytest

from digital_cerebellum.core.types import MemorySlot, ErrorType
from digital_cerebellum.core.error_comparator import ErrorComparator, cosine_distance
from digital_cerebellum.core.prediction_engine import EngineConfig, PredictionEngine
from digital_cerebellum.memory.fluid_memory import FluidMemory
from digital_cerebellum.memory.sleep_cycle import SleepCycle, SleepReport
from digital_cerebellum.cortex.consolidation import ConsolidationPipeline, TaskPattern


# ======================================================================
# Sleep Cycle
# ======================================================================

class TestSleepCycle:
    def _make_slot(self, content: str, layer: str = "short_term",
                   strength: float = 0.5, access_count: int = 0,
                   embedding: np.ndarray | None = None) -> MemorySlot:
        emb = embedding if embedding is not None else np.random.randn(64).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return MemorySlot(
            content=content, embedding=emb, strength=strength,
            layer=layer, access_count=access_count,
        )

    def test_basic_run(self):
        mem = FluidMemory()
        for i in range(5):
            mem.store(self._make_slot(f"mem-{i}"))
        cycle = SleepCycle()
        report = cycle.run(mem)
        assert isinstance(report, SleepReport)
        assert report.duration_ms > 0

    def test_consolidation_promotes(self):
        mem = FluidMemory()
        slot = self._make_slot("frequently accessed", access_count=5, strength=0.8)
        mem.store(slot)
        cycle = SleepCycle()
        report = cycle.run(mem)
        assert report.consolidated >= 1
        assert slot.layer == "long_term"

    def test_pattern_abstraction(self):
        mem = FluidMemory()
        base_emb = np.random.randn(64).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        for i in range(4):
            noise = np.random.randn(64).astype(np.float32) * 0.02
            emb = base_emb + noise
            emb = emb / np.linalg.norm(emb)
            slot = self._make_slot(
                f"similar-{i}", layer="long_term",
                strength=0.8, embedding=emb,
            )
            mem.store(slot)

        cycle = SleepCycle(cluster_threshold=0.90, min_cluster_size=3)
        report = cycle.run(mem)
        assert report.abstracted >= 1
        assert len(report.patterns) >= 1
        assert report.patterns[0].metadata.get("is_prototype") is True

    def test_conflict_resolution(self):
        mem = FluidMemory()
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        s1 = self._make_slot("strong", layer="long_term", strength=0.9, embedding=emb.copy())
        s2 = self._make_slot("weak", layer="long_term", strength=0.3, embedding=emb.copy())
        mem.store(s1)
        mem.store(s2)

        before = len(mem)
        cycle = SleepCycle()
        report = cycle.run(mem)
        assert report.conflicts_resolved >= 1
        assert len(mem) < before

    def test_distill_flagging(self):
        mem = FluidMemory()
        slot = self._make_slot(
            "mature", layer="long_term",
            strength=0.8, access_count=10,
        )
        mem.store(slot)
        cycle = SleepCycle()
        report = cycle.run(mem)
        assert report.distill_candidates >= 1
        assert slot.metadata.get("distill_ready") is True


# ======================================================================
# Task Consolidation Pipeline
# ======================================================================

class TestConsolidationPipeline:
    def test_stage_progression(self):
        pipe = ConsolidationPipeline()
        pid = "test_pattern"

        for _ in range(5):
            pipe.record_observation(pid, "tool_call", was_fast_path=False)
        assert pipe.get_stage(pid) == 1

        for _ in range(5):
            pipe.record_observation(pid, "tool_call", was_fast_path=False)
        assert pipe.get_stage(pid) == 2

    def test_graduation(self):
        pipe = ConsolidationPipeline()
        pid = "grad_test"

        for _ in range(10):
            pipe.record_observation(pid, "tool_call", was_fast_path=False)

        for _ in range(12):
            pipe.record_observation(pid, "tool_call",
                                    was_fast_path=True, was_correct=True)

        assert pipe.is_graduated(pid)
        assert pipe.get_stage(pid) == 3

    def test_degradation(self):
        pipe = ConsolidationPipeline()
        pid = "degrade_test"

        for _ in range(10):
            pipe.record_observation(pid, "tool_call", was_fast_path=False)
        for _ in range(12):
            pipe.record_observation(pid, "tool_call",
                                    was_fast_path=True, was_correct=True)
        assert pipe.is_graduated(pid)

        for _ in range(10):
            pipe.record_observation(pid, "tool_call",
                                    was_fast_path=True, was_correct=False)

        assert not pipe.is_graduated(pid)

    def test_stats(self):
        pipe = ConsolidationPipeline()
        for i in range(3):
            pipe.record_observation(f"p{i}", "tool_call", was_fast_path=False)
        stats = pipe.stats
        assert stats["total_patterns"] == 3
        assert stats["by_microzone"]["tool_call"] == 3

    def test_graduated_patterns_filter(self):
        pipe = ConsolidationPipeline()
        for _ in range(10):
            pipe.record_observation("p1", "tool_call", was_fast_path=False)
            pipe.record_observation("p2", "payment", was_fast_path=False)
        for _ in range(12):
            pipe.record_observation("p1", "tool_call",
                                    was_fast_path=True, was_correct=True)

        assert len(pipe.graduated_patterns("tool_call")) == 1
        assert len(pipe.graduated_patterns("payment")) == 0


# ======================================================================
# TPE / RPE (full implementation)
# ======================================================================

class TestTemporalPredictionError:
    def test_basic_tpe(self):
        comp = ErrorComparator()
        err = comp.compute_temporal_error(
            predicted_time=1.0, actual_time=1.2, event_id="e1",
        )
        assert err.error_type == ErrorType.TEMPORAL
        assert err.value > 0
        assert err.vector is not None
        assert len(err.vector) == 4

    def test_weber_law_scaling(self):
        comp = ErrorComparator()
        err_short = comp.compute_temporal_error(predicted_time=0.1, actual_time=0.2)
        err_long = comp.compute_temporal_error(predicted_time=10.0, actual_time=10.1)
        assert abs(err_short.value) > abs(err_long.value)

    def test_timing_prediction(self):
        comp = ErrorComparator()
        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            comp.compute_temporal_error(predicted_time=t, actual_time=t)
        predicted = comp.predict_next_time()
        assert predicted > 5.0


class TestRewardPredictionError:
    def test_basic_rpe(self):
        comp = ErrorComparator()
        err = comp.compute_reward_error(
            expected_reward=0.8, actual_reward=1.0, event_id="e1",
        )
        assert err.error_type == ErrorType.REWARD
        assert err.value == pytest.approx(0.2)
        assert err.vector is not None

    def test_negative_rpe(self):
        comp = ErrorComparator()
        err = comp.compute_reward_error(expected_reward=0.9, actual_reward=-1.0)
        assert err.value < 0

    def test_stats_tracking(self):
        comp = ErrorComparator()
        for _ in range(5):
            comp.compute_sensory_error(
                PredictionEngine(EngineConfig(rff_dim=64, action_dim=16, outcome_dim=16))
                    .predict_numpy(np.random.randn(64).astype(np.float32)),
                np.random.randn(16).astype(np.float32),
                np.random.randn(16).astype(np.float32),
            )
            comp.compute_temporal_error(1.0, 1.1)
            comp.compute_reward_error(0.5, 1.0)

        stats = comp.stats
        assert stats["spe"]["count"] == 5
        assert stats["tpe"]["count"] == 5
        assert stats["rpe"]["count"] == 5

    def test_is_improving(self):
        comp = ErrorComparator(window_size=100)
        for i in range(40):
            err_val = 1.0 - i * 0.02
            comp._spe_history.append(max(err_val, 0.1))
        assert comp.is_improving("spe", window=10)
