"""
Failure Memory — learns from past mistakes to prevent future ones.

Biological basis:
  Damasio's somatic marker hypothesis: past failures leave emotional
  traces that bias future decisions before conscious reasoning.
  The cerebellum contributes by encoding the PATTERN of the situation
  (state + action) that led to failure, so that similar future
  situations trigger a preemptive "bad feeling."

Digital implementation:
  Store (state_embedding, action_embedding) fingerprints of failed
  steps.  Before each new step, compare the current fingerprint
  against stored failures.  If similar → warn the agent.

This is distinct from SomaticMarker (which works on prediction head
divergence patterns).  FailureMemory works on the actual state/action
embeddings, making it directly interpretable: "the last time you
tried X in situation Y, it failed."
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from digital_cerebellum.monitor.types import FailureWarning


@dataclass
class _FailureRecord:
    """One stored failure pattern."""

    fingerprint: np.ndarray
    action_text: str
    state_text: str
    error_description: str
    severity: float
    step_number: int
    strength: float = 1.0


class FailureMemory:
    """
    Stores and retrieves failure patterns for preemptive warning.

    Framework-agnostic: only sees embedding vectors and text
    descriptions.  Doesn't know whether the agent is clicking
    buttons or playing chess.
    """

    def __init__(
        self,
        max_records: int = 200,
        similarity_threshold: float = 0.80,
        decay_rate: float = 0.995,
    ):
        self._records: deque[_FailureRecord] = deque(maxlen=max_records)
        self._similarity_threshold = similarity_threshold
        self._decay_rate = decay_rate
        self._total_recorded = 0
        self._total_warnings = 0

    def record(
        self,
        state_emb: np.ndarray,
        action_emb: np.ndarray,
        action_text: str = "",
        state_text: str = "",
        error_description: str = "",
        severity: float = 1.0,
        step_number: int = 0,
    ) -> None:
        """Record a failure pattern for future reference."""
        fp = self._make_fingerprint(state_emb, action_emb)
        self._records.append(_FailureRecord(
            fingerprint=fp,
            action_text=action_text,
            state_text=state_text,
            error_description=error_description,
            severity=np.clip(severity, 0.0, 1.0),
            step_number=step_number,
        ))
        self._total_recorded += 1

    def check(
        self,
        state_emb: np.ndarray,
        action_emb: np.ndarray,
    ) -> FailureWarning | None:
        """
        Check if the current (state, action) matches a known failure.

        Returns a FailureWarning if a similar pattern was seen before,
        None otherwise.
        """
        if not self._records:
            return None

        current_fp = self._make_fingerprint(state_emb, action_emb)
        fp_norm = np.linalg.norm(current_fp)
        if fp_norm < 1e-9:
            return None

        best_sim = -1.0
        best_record: _FailureRecord | None = None

        for record in self._records:
            rn = np.linalg.norm(record.fingerprint)
            if rn < 1e-9:
                continue
            sim = float(np.dot(current_fp, record.fingerprint) / (fp_norm * rn))

            if sim > best_sim:
                best_sim = sim
                best_record = record

        if best_record is None or best_sim < self._similarity_threshold:
            return None

        self._total_warnings += 1
        return FailureWarning(
            pattern_description=(
                best_record.error_description
                or f"Similar to failed step {best_record.step_number}: "
                   f"action='{best_record.action_text}'"
            ),
            similarity=best_sim,
            severity=best_record.severity * best_record.strength,
            suggested_alternative=f"Previous failure in similar context: "
                                  f"{best_record.error_description}",
        )

    def decay(self) -> None:
        """Apply time-based decay (call during sleep/maintenance)."""
        for record in self._records:
            record.strength *= self._decay_rate

    @staticmethod
    def _make_fingerprint(
        state_emb: np.ndarray,
        action_emb: np.ndarray,
    ) -> np.ndarray:
        """Combine state and action into a single fingerprint vector."""
        s = state_emb / max(np.linalg.norm(state_emb), 1e-9)
        a = action_emb / max(np.linalg.norm(action_emb), 1e-9)
        return np.concatenate([s, a]).astype(np.float32)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "stored_failures": len(self._records),
            "total_recorded": self._total_recorded,
            "total_warnings": self._total_warnings,
            "mean_severity": round(
                float(np.mean([r.severity for r in self._records])), 3
            ) if self._records else 0.0,
        }
