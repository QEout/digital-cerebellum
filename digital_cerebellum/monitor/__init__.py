"""
Digital Cerebellum — Step Monitor (Phase 7).

Universal agent step monitoring protocol.  Any AI agent framework
can use these components to get predictive error interception:

    from digital_cerebellum.monitor import StepMonitor

    monitor = StepMonitor()

    prediction = monitor.before_step(action="...", state="...")
    # ... execute action ...
    verdict = monitor.after_step(outcome="...", success=True)

Components:
    StepMonitor          — main orchestrator (before_step / after_step)
    ErrorCascadeDetector — tracks SPE across steps, detects cascades
    FailureMemory        — learns from past failures, preemptive warning
    StepForwardModel     — predicts outcomes from (state, action) pairs
"""

from digital_cerebellum.monitor.step_monitor import StepMonitor
from digital_cerebellum.monitor.cascade_detector import ErrorCascadeDetector
from digital_cerebellum.monitor.failure_memory import FailureMemory
from digital_cerebellum.monitor.step_forward_model import StepForwardModel
from digital_cerebellum.monitor.types import (
    CascadeStatus,
    FailureWarning,
    RollbackPlan,
    StepPrediction,
    StepRecord,
    StepVerdict,
)

__all__ = [
    "StepMonitor",
    "ErrorCascadeDetector",
    "FailureMemory",
    "StepForwardModel",
    "CascadeStatus",
    "FailureWarning",
    "RollbackPlan",
    "StepPrediction",
    "StepRecord",
    "StepVerdict",
]
