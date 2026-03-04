"""
Phase 3 — Emergent cognitive properties built on cerebellar signals.

Three subsystems that transform low-level prediction/error signals into
higher-order cognitive phenomena:

  SomaticMarker  — "gut feeling" from population divergence + valence history
  CuriosityDrive — intrinsic motivation from learning progress
  SelfModel      — metacognitive competency awareness
"""

from digital_cerebellum.emergence.somatic_marker import SomaticMarker, GutFeeling
from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive, CuriositySignal
from digital_cerebellum.emergence.self_model import SelfModel, SelfReport

__all__ = [
    "SomaticMarker", "GutFeeling",
    "CuriosityDrive", "CuriositySignal",
    "SelfModel", "SelfReport",
]
