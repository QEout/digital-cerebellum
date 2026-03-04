"""
Micro-Operation Engine — Phase 6: continuous real-time control.

Gives the cerebellum a "body" — the ability to execute continuous
actions in an environment at 60Hz+, learning from prediction errors.
"""

from digital_cerebellum.micro_ops.engine import (
    MicroOpEngine,
    MicroOpConfig,
    StepResult,
    Environment,
)

__all__ = [
    "MicroOpEngine",
    "MicroOpConfig",
    "StepResult",
    "Environment",
]
