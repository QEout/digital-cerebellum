"""Core data types for the Digital Cerebellum."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class EventContext:
    user_id: str | None = None
    session_id: str | None = None
    related_events: list[str] | None = None


@dataclass
class CerebellumEvent:
    type: str
    source: str
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    context: EventContext | None = None


# ---------------------------------------------------------------------------
# Tool-call specific (Phase 0 primary input)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallEvent(CerebellumEvent):
    """A specialised event wrapping an LLM agent's tool invocation."""

    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    conversation_context: str = ""

    def __post_init__(self):
        self.type = "tool_call"
        self.source = self.source or "agent"


# ---------------------------------------------------------------------------
# Prediction Engine I/O
# ---------------------------------------------------------------------------

@dataclass
class HeadPrediction:
    action_embedding: np.ndarray      # (action_dim,)
    outcome_embedding: np.ndarray     # (outcome_dim,)


@dataclass
class PredictionOutput:
    action_embedding: np.ndarray      # mean across K heads
    outcome_embedding: np.ndarray     # mean across K heads
    confidence: float                 # emergent from population agreement
    head_predictions: list[HeadPrediction]
    task_outputs: dict[str, float] = field(default_factory=dict)
    domain_logits: np.ndarray | None = None

    # raw tensors kept for backward pass
    _raw_tensor: torch.Tensor | None = field(default=None, repr=False)

    @property
    def safety_score(self) -> float:
        """Backward-compatible accessor for the tool_call microzone."""
        return self.task_outputs.get("safety", 0.5)


# ---------------------------------------------------------------------------
# Error signals (three channels)
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    SENSORY = "sensory"    # SPE: predicted vs actual outcome
    TEMPORAL = "temporal"  # TPE: predicted vs actual timing
    REWARD = "reward"      # RPE: expected value vs user feedback


@dataclass
class ErrorSignal:
    error_type: ErrorType
    value: float                 # scalar magnitude
    vector: np.ndarray | None    # detailed error vector (for SPE)
    source_event_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RouteDecision(Enum):
    FAST = "fast"
    SHADOW = "shadow"
    SLOW = "slow"


@dataclass
class RoutingResult:
    decision: RouteDecision
    confidence: float
    reason: str = ""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class MemorySlot:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    strength: float = 1.0
    layer: str = "short_term"   # "sensory" | "short_term" | "long_term"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool-call evaluation result (Phase 0 primary output)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallEvaluation:
    safe: bool
    risk_type: str          # "none" | "wrong_param" | "wrong_tool" | "hallucinated_data" | ...
    confidence: float
    predicted_outcome: np.ndarray | None = None
    details: str = ""
