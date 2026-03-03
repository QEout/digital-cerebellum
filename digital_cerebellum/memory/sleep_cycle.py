"""
Sleep Cycle — offline memory maintenance.

Biological cerebellum consolidates memories during sleep through
replay, abstraction, and pruning.  This module performs the digital
equivalent on the FluidMemory store:

    1. Decay sweep      — remove memories below strength floor
    2. Consolidation    — promote qualifying short-term → long-term
    3. Pattern abstract — cluster similar memories → extract prototype
    4. Distill check    — flag mature patterns for prediction engine
    5. Conflict resolve — deduplicate contradictory memories
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from digital_cerebellum.core.types import MemorySlot

if TYPE_CHECKING:
    from digital_cerebellum.memory.fluid_memory import FluidMemory

log = logging.getLogger(__name__)


@dataclass
class SleepReport:
    """Summary of what happened during one sleep cycle."""
    decayed: int = 0
    consolidated: int = 0
    abstracted: int = 0
    distill_candidates: int = 0
    conflicts_resolved: int = 0
    duration_ms: float = 0.0
    patterns: list[MemorySlot] = field(default_factory=list)


class SleepCycle:
    """
    Periodic offline maintenance for the fluid memory system.

    Trigger manually (``sleep_cycle.run(memory)``) or on a schedule.
    Each run performs all five maintenance stages.
    """

    CLUSTER_THRESHOLD = 0.80
    MIN_CLUSTER_SIZE = 3
    DISTILL_MIN_ACCESS = 5
    DISTILL_MIN_STRENGTH = 0.6

    def __init__(
        self,
        cluster_threshold: float = 0.80,
        min_cluster_size: int = 3,
    ):
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.cycle_count = 0

    def run(self, memory: FluidMemory) -> SleepReport:
        """Execute one full sleep cycle. Returns a report."""
        t0 = time.perf_counter()
        report = SleepReport()

        self._decay_sweep(memory, report)
        self._consolidate(memory, report)
        self._pattern_abstract(memory, report)
        self._distill_check(memory, report)
        self._conflict_resolve(memory, report)

        report.duration_ms = (time.perf_counter() - t0) * 1000
        self.cycle_count += 1
        log.info(
            "Sleep cycle #%d: decayed=%d, consolidated=%d, "
            "abstracted=%d, distill=%d, conflicts=%d (%.1fms)",
            self.cycle_count, report.decayed, report.consolidated,
            report.abstracted, report.distill_candidates,
            report.conflicts_resolved, report.duration_ms,
        )
        return report

    # -- Stage 1: Decay sweep ------------------------------------------------

    def _decay_sweep(self, memory: FluidMemory, report: SleepReport):
        memory._apply_decay()
        dead = [
            sid for sid, s in memory._slots.items()
            if s.strength < memory.STRENGTH_FLOOR
        ]
        for sid in dead:
            del memory._slots[sid]
        report.decayed = len(dead)

    # -- Stage 2: Consolidation (short-term → long-term) ---------------------

    def _consolidate(self, memory: FluidMemory, report: SleepReport):
        for s in list(memory._slots.values()):
            if s.layer != "short_term":
                continue
            if s.access_count >= 3 or s.strength > 0.7:
                s.layer = "long_term"
                report.consolidated += 1

    # -- Stage 3: Pattern abstraction ----------------------------------------

    def _pattern_abstract(self, memory: FluidMemory, report: SleepReport):
        """
        Cluster similar long-term memories and create prototype memories.

        Uses greedy single-linkage clustering with cosine similarity.
        Original memories get weaker; the prototype inherits their strength.
        """
        lt_slots = [s for s in memory._slots.values() if s.layer == "long_term"]
        if len(lt_slots) < self.min_cluster_size:
            return

        used = set()
        clusters: list[list[MemorySlot]] = []

        for i, anchor in enumerate(lt_slots):
            if anchor.id in used:
                continue
            cluster = [anchor]
            used.add(anchor.id)
            for j, candidate in enumerate(lt_slots):
                if candidate.id in used:
                    continue
                sim = self._cosine_sim(anchor.embedding, candidate.embedding)
                if sim >= self.cluster_threshold:
                    cluster.append(candidate)
                    used.add(candidate.id)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        for cluster in clusters:
            prototype = self._make_prototype(cluster)
            memory.store(prototype)
            for s in cluster:
                s.strength *= 0.5
            report.abstracted += 1
            report.patterns.append(prototype)

    def _make_prototype(self, cluster: list[MemorySlot]) -> MemorySlot:
        """Create a prototype memory from a cluster of similar memories."""
        embeddings = np.stack([s.embedding for s in cluster])
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        total_access = sum(s.access_count for s in cluster)
        max_strength = max(s.strength for s in cluster)

        source_ids = [s.id for s in cluster]
        contents = [s.content for s in cluster]
        abstract_content = f"[pattern from {len(cluster)} memories] {contents[0][:80]}..."

        return MemorySlot(
            content=abstract_content,
            embedding=centroid,
            strength=min(max_strength * 1.2, 1.0),
            layer="long_term",
            access_count=total_access,
            source_ids=source_ids,
            metadata={"is_prototype": True, "cluster_size": len(cluster)},
        )

    # -- Stage 4: Distill check ----------------------------------------------

    def _distill_check(self, memory: FluidMemory, report: SleepReport):
        """
        Flag mature patterns that could be compiled into the prediction engine.

        A pattern is "distill-ready" when it has been accessed many times
        and remains strong — meaning it encodes a reliable regularity.
        """
        for s in memory._slots.values():
            if s.layer != "long_term":
                continue
            if s.metadata.get("distill_ready"):
                continue
            if (s.access_count >= self.DISTILL_MIN_ACCESS
                    and s.strength >= self.DISTILL_MIN_STRENGTH):
                s.metadata["distill_ready"] = True
                s.metadata["distill_flagged_at"] = time.time()
                report.distill_candidates += 1

    # -- Stage 5: Conflict resolution ----------------------------------------

    def _conflict_resolve(self, memory: FluidMemory, report: SleepReport):
        """
        Find pairs of highly similar memories and keep the stronger one.

        Contradictory memories arise when the same input led to different
        outcomes at different times.  We keep the more recent / stronger.
        """
        slots = list(memory._slots.values())
        to_remove = set()

        for i in range(len(slots)):
            if slots[i].id in to_remove:
                continue
            for j in range(i + 1, len(slots)):
                if slots[j].id in to_remove:
                    continue
                if slots[i].layer != slots[j].layer:
                    continue
                sim = self._cosine_sim(slots[i].embedding, slots[j].embedding)
                if sim >= 0.95:
                    weaker = slots[j] if slots[i].strength >= slots[j].strength else slots[i]
                    to_remove.add(weaker.id)

        for sid in to_remove:
            if sid in memory._slots:
                del memory._slots[sid]
        report.conflicts_resolved = len(to_remove)

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
