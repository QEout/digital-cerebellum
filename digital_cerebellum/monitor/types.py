"""
Step Monitor data types — framework-agnostic protocol.

These types define the universal interface between any AI agent and
the Digital Cerebellum's step monitoring system.  An agent only needs
to speak in terms of actions, states, and outcomes — the cerebellum
handles prediction, error detection, and cascade prevention internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepPrediction:
    """
    Returned by StepMonitor.before_step().

    Tells the agent what the cerebellum expects and whether it's safe
    to proceed.
    """

    predicted_outcome: np.ndarray
    confidence: float
    risk_score: float
    should_proceed: bool = True
    failure_warning: FailureWarning | None = None
    cascade_risk: float = 0.0
    step_number: int = 0


@dataclass
class StepVerdict:
    """
    Returned by StepMonitor.after_step().

    The cerebellum's judgment on what just happened: was the outcome
    as predicted?  Is an error cascade building up?
    """

    spe: float
    should_pause: bool = False
    cascade_risk: float = 0.0
    consecutive_errors: int = 0
    suggestion: str = "continue"
    step_number: int = 0
    learned: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureWarning:
    """A preemptive warning from failure memory."""

    pattern_description: str
    similarity: float
    severity: float
    suggested_alternative: str = ""


@dataclass
class CascadeStatus:
    """Output of the ErrorCascadeDetector."""

    risk: float
    is_cascading: bool
    consecutive_high: int
    trend: str
    mean_recent_spe: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPlan:
    """
    Computed by StepMonitor when an error cascade is detected.

    Provides the agent with everything it needs to undo damage:
    which step to return to, what failed, and the state description
    at the last safe checkpoint.
    """

    rollback_to_step: int
    last_safe_state: str
    last_safe_outcome: str
    failed_steps: list[dict[str, Any]]
    total_steps: int
    steps_wasted: int
    cascade_risk: float
    recommendation: str


@dataclass
class StepRecord:
    """Internal record of one completed step."""

    step_number: int
    action_embedding: np.ndarray
    state_embedding: np.ndarray | None
    predicted_outcome: np.ndarray
    actual_outcome: np.ndarray | None = None
    spe: float = 0.0
    success: bool | None = None
    timestamp: float = 0.0
    action_text: str = ""
    state_text: str = ""
    outcome_text: str = ""
