"""
Microzone — the universal cerebellar computation unit.

In biology, every cerebellar microzone uses the same circuit
(granule cells → Purkinje cells → deep nuclei) but processes
a different domain (motor, language, cognition, emotion).

Each digital Microzone defines:
    1. How to encode domain events into feature vectors
    2. What task-specific readout heads it needs
    3. How to use the LLM for slow-path evaluation
    4. How to interpret predictions on the fast path
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from digital_cerebellum.core.types import PredictionOutput


@dataclass
class TaskHeadConfig:
    """Specification for a task-specific readout head."""
    name: str                    # e.g. "safety", "risk_score", "action_type"
    output_dim: int = 1          # 1 for binary/scalar, N for multi-class
    activation: str = "sigmoid"  # "sigmoid" | "softmax" | "none"


@dataclass
class SlowPathRequest:
    """What to send to the LLM for slow-path evaluation."""
    system_prompt: str
    user_message: str


@dataclass
class LearningSignal:
    """Training labels extracted from an LLM response or external feedback."""
    task_labels: dict[str, float] = field(default_factory=dict)
    outcome_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Microzone(ABC):
    """
    Abstract base class for all cerebellar microzones.

    Subclass this to create a new domain (tool safety, payments,
    game actions, dialogue rhythm, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this microzone."""

    @abstractmethod
    def task_heads(self) -> list[TaskHeadConfig]:
        """Define the task-specific readout heads this microzone needs."""

    @abstractmethod
    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        """
        Convert a domain-specific payload into text for the shared encoder.

        The text will be passed through sentence-transformers → RFF → K heads.
        """

    @abstractmethod
    def build_slow_path_request(
        self,
        payload: dict[str, Any],
        context: str,
        prediction: PredictionOutput,
    ) -> SlowPathRequest:
        """Construct the LLM prompt for slow-path evaluation."""

    @abstractmethod
    def parse_slow_path_response(
        self,
        llm_response: dict[str, Any],
    ) -> LearningSignal:
        """Extract training labels and outcome from the LLM's response."""

    @abstractmethod
    def fast_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
    ) -> dict[str, Any]:
        """
        Produce a domain-specific evaluation from prediction alone.

        Called when the router selects the fast path.
        Returns a dict that the caller interprets per their domain.
        """

    @abstractmethod
    def slow_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
        llm_response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Produce a domain-specific evaluation using the LLM response.

        Called on slow/shadow path.
        """
