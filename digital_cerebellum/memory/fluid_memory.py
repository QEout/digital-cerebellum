"""
Fluid Memory v0 — hippocampal + cortical memory analogue.

Phase 0 implements:
  - MemorySlot storage with strength-based decay
  - Retrieval with reconsolidation (recall reshapes the memory)
  - Basic consolidation (promote strong short-term → long-term)

Phase 1 adds the full Sleep Cycle (see sleep_cycle.py).
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from digital_cerebellum.core.types import MemorySlot


class FluidMemory:
    """
    In-memory implementation of the fluid memory system.

    Phase 0 uses plain Python dicts.  Phase 1 migrates to SQLite + FAISS.
    """

    LAMBDA_SHORT = 0.1    # fast exponential decay
    LAMBDA_LONG = 0.01    # slow sub-linear decay
    RECONSOLIDATION_ALPHA_SHORT = 0.05
    RECONSOLIDATION_ALPHA_LONG = 0.02
    STRENGTH_FLOOR = 0.05
    SHORT_TERM_CAPACITY = 50

    def __init__(self):
        self._slots: dict[str, MemorySlot] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def store(self, slot: MemorySlot) -> str:
        """Insert or update a memory slot.  Returns the slot id."""
        self._slots[slot.id] = slot
        self._enforce_capacity()
        return slot.id

    # ------------------------------------------------------------------
    # Read (with reconsolidation)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_strength: float = 0.1,
    ) -> list[MemorySlot]:
        """
        Semantic nearest-neighbour retrieval.

        Every retrieved memory is *reconsolidated*:
        - strength re-activated to max(strength, 0.8)
        - embedding micro-shifted toward the query
        """
        self._apply_decay()

        candidates = [
            s for s in self._slots.values()
            if s.strength >= min_strength
        ]
        if not candidates:
            return []

        scored = []
        for s in candidates:
            sim = self._cosine_sim(query_embedding, s.embedding)
            scored.append((sim, s))
        scored.sort(key=lambda t: t[0], reverse=True)

        results = []
        for sim, s in scored[:top_k]:
            self._reconsolidate(s, query_embedding)
            results.append(s)
        return results

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------
    def _apply_decay(self):
        now = time.time()
        dead_ids = []
        for s in self._slots.values():
            dt = (now - s.last_accessed) / 3600.0  # hours
            if dt <= 0:
                continue
            if s.layer == "short_term":
                s.strength *= math.exp(-self.LAMBDA_SHORT * dt)
            else:
                s.strength *= math.exp(-self.LAMBDA_LONG * (dt ** 0.8))
            if s.strength < self.STRENGTH_FLOOR:
                dead_ids.append(s.id)
        for sid in dead_ids:
            del self._slots[sid]

    # ------------------------------------------------------------------
    # Reconsolidation
    # ------------------------------------------------------------------
    def _reconsolidate(self, slot: MemorySlot, query_emb: np.ndarray):
        slot.access_count += 1
        slot.last_accessed = time.time()
        slot.strength = max(slot.strength, 0.8)

        alpha = (
            self.RECONSOLIDATION_ALPHA_SHORT
            if slot.layer == "short_term"
            else self.RECONSOLIDATION_ALPHA_LONG
        )
        slot.embedding = (1 - alpha) * slot.embedding + alpha * query_emb
        # re-normalise
        norm = np.linalg.norm(slot.embedding) + 1e-9
        slot.embedding = slot.embedding / norm

    # ------------------------------------------------------------------
    # Consolidation (short-term → long-term)
    # ------------------------------------------------------------------
    def consolidate(self):
        """Promote qualifying short-term memories to long-term."""
        for s in list(self._slots.values()):
            if s.layer != "short_term":
                continue
            if s.access_count >= 3 or s.strength > 0.7:
                s.layer = "long_term"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(dot / norm)

    def _enforce_capacity(self):
        short = [s for s in self._slots.values() if s.layer == "short_term"]
        if len(short) <= self.SHORT_TERM_CAPACITY:
            return
        short.sort(key=lambda s: s.strength)
        for s in short[: len(short) - self.SHORT_TERM_CAPACITY]:
            del self._slots[s.id]

    def __len__(self):
        return len(self._slots)

    @property
    def stats(self) -> dict:
        layers = {}
        for s in self._slots.values():
            layers[s.layer] = layers.get(s.layer, 0) + 1
        return {"total": len(self), "by_layer": layers}
