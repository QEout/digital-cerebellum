"""
State Estimator — parallel to the Prediction Engine.

Biology: Recent findings (2026 Nature Communications) show that certain
cerebellar regions encode ground-truth self-motion rather than predictions.
This means the cerebellum doesn't just predict—it also maintains an accurate
estimate of "what is actually happening right now."

Digital: While the Prediction Engine asks "what will happen next?", the
State Estimator asks "what is the current state of the system?"  It maintains
a compressed running estimate of the agent's operational context by tracking:

  1. Recent action history (what tools were called)
  2. Outcome statistics (success/failure rates)
  3. Temporal patterns (cadence of events)
  4. Risk profile (running estimate of environmental danger)

The state estimate is fed back into the Prediction Engine and Decision Router,
improving context-awareness for multi-step action sequences.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time

import torch
import torch.nn as nn


@dataclass
class StateSnapshot:
    """A point-in-time summary of the agent's operational state."""
    timestamp: float = 0.0
    action_count: int = 0
    fast_path_count: int = 0
    slow_path_count: int = 0
    success_rate: float = 0.5
    avg_confidence: float = 0.5
    risk_level: float = 0.5
    event_cadence: float = 0.0
    recent_tools: list[str] = field(default_factory=list)


class StateEstimator(nn.Module):
    """
    Maintains a continuous estimate of the agent's operational state.

    Parameters
    ----------
    state_dim : int
        Dimension of the state embedding vector.
    history_len : int
        Number of recent events to keep for context.
    """

    def __init__(self, state_dim: int = 64, history_len: int = 50):
        super().__init__()
        self.state_dim = state_dim
        self.history_len = history_len

        self._history: deque[dict] = deque(maxlen=history_len)
        self._timestamps: deque[float] = deque(maxlen=history_len)
        self._successes: deque[bool] = deque(maxlen=history_len)
        self._confidences: deque[float] = deque(maxlen=history_len)
        self._routes: deque[str] = deque(maxlen=history_len)
        self._risk_ema: float = 0.5

        raw_features = 8
        self.state_proj = nn.Sequential(
            nn.Linear(raw_features, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )

    def record_event(
        self,
        tool_name: str,
        route: str,
        confidence: float,
        success: bool | None = None,
        risk_score: float = 0.0,
    ):
        """Record an event for state tracking."""
        now = time.time()
        self._history.append({
            "tool": tool_name, "route": route,
            "confidence": confidence, "risk": risk_score,
        })
        self._timestamps.append(now)
        self._confidences.append(confidence)
        self._routes.append(route)
        if success is not None:
            self._successes.append(success)

        self._risk_ema = 0.9 * self._risk_ema + 0.1 * risk_score

    def get_snapshot(self) -> StateSnapshot:
        """Compute the current state summary."""
        n = len(self._history)
        if n == 0:
            return StateSnapshot()

        fast_count = sum(1 for r in self._routes if r == "fast")
        slow_count = sum(1 for r in self._routes if r in ("slow", "shadow"))
        success_rate = (
            sum(self._successes) / len(self._successes)
            if self._successes else 0.5
        )
        avg_conf = sum(self._confidences) / len(self._confidences)

        cadence = 0.0
        if len(self._timestamps) >= 2:
            intervals = [
                self._timestamps[i] - self._timestamps[i - 1]
                for i in range(1, len(self._timestamps))
            ]
            cadence = 1.0 / max(sum(intervals) / len(intervals), 0.001)

        recent_tools = [
            e["tool"] for e in list(self._history)[-5:]
        ]

        return StateSnapshot(
            timestamp=self._timestamps[-1] if self._timestamps else 0.0,
            action_count=n,
            fast_path_count=fast_count,
            slow_path_count=slow_count,
            success_rate=success_rate,
            avg_confidence=avg_conf,
            risk_level=self._risk_ema,
            event_cadence=cadence,
            recent_tools=recent_tools,
        )

    def forward(self, external_context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Produce a state embedding vector from current statistics.

        Parameters
        ----------
        external_context : optional (dim,) tensor to incorporate

        Returns
        -------
        state_vec : (state_dim,) tensor
        """
        snap = self.get_snapshot()

        raw = torch.tensor([
            min(snap.action_count / 100.0, 1.0),
            snap.fast_path_count / max(snap.action_count, 1),
            snap.success_rate,
            snap.avg_confidence,
            snap.risk_level,
            min(snap.event_cadence / 10.0, 1.0),
            math.tanh(snap.action_count / 50.0),
            1.0 - snap.risk_level,
        ], dtype=torch.float32)

        state_vec = self.state_proj(raw)
        return state_vec

    @property
    def stats(self) -> dict:
        snap = self.get_snapshot()
        return {
            "action_count": snap.action_count,
            "fast_ratio": snap.fast_path_count / max(snap.action_count, 1),
            "success_rate": snap.success_rate,
            "avg_confidence": snap.avg_confidence,
            "risk_level": snap.risk_level,
            "cadence_hz": snap.event_cadence,
        }
