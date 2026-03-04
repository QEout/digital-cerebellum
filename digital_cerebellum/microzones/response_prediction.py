"""
Response Prediction Microzone — learn to predict LLM response patterns.

This is the most "cerebellar" of all microzones.  Rather than evaluating
safety, it predicts WHAT the LLM would respond — enabling the cerebellum
to short-circuit the LLM entirely for familiar patterns.

Biological analogy: when you catch a ball, the cerebellum doesn't wait
for visual feedback — it predicts the trajectory and moves the hand
proactively.  This microzone predicts the LLM's response trajectory
and provides the answer directly when confident.

This microzone works hand-in-hand with the SkillStore:
- ResponsePredictionMicrozone learns WHEN to predict (confidence routing)
- SkillStore stores WHAT to predict (the actual responses)
"""

from __future__ import annotations

import json
from typing import Any

from digital_cerebellum.core.microzone import (
    LearningSignal,
    Microzone,
    SlowPathRequest,
    TaskHeadConfig,
)
from digital_cerebellum.core.types import PredictionOutput


class ResponsePredictionMicrozone(Microzone):
    """Microzone for predicting LLM response patterns."""

    @property
    def name(self) -> str:
        return "response_prediction"

    def task_heads(self) -> list[TaskHeadConfig]:
        return [
            TaskHeadConfig(name="response_predictability", output_dim=1, activation="sigmoid"),
            TaskHeadConfig(name="response_complexity", output_dim=1, activation="sigmoid"),
        ]

    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        query = payload.get("query", "")
        query_type = payload.get("query_type", "general")

        text = f"[{query_type}] {query}"
        if context:
            text = f"{context} -> {text}"
        return text

    def build_slow_path_request(
        self,
        payload: dict[str, Any],
        context: str,
        prediction: PredictionOutput,
    ) -> SlowPathRequest:
        system_prompt = (
            "You are analyzing a query to determine its response characteristics. "
            "Assess: how predictable/templated is the expected response? "
            "How complex is the reasoning required?\n"
            "Respond ONLY with a JSON object: "
            '{"predictability": 0.0-1.0, '
            '"complexity": 0.0-1.0, '
            '"response_type": "factual"|"creative"|"analytical"|"procedural"|"conversational", '
            '"cacheable": bool, '
            '"reasoning": "..."}'
        )

        parts = [
            f"Query: {payload.get('query', '')}",
            f"Type: {payload.get('query_type', 'general')}",
        ]
        if context:
            parts.append(f"Context: {context}")

        predictability = prediction.task_outputs.get("response_predictability")
        if predictability is not None:
            parts.append(
                f"Cerebellar hint: predictability={predictability:.3f}, "
                f"complexity={prediction.task_outputs.get('response_complexity', 0):.3f}, "
                f"confidence={prediction.confidence:.3f}"
            )

        return SlowPathRequest(
            system_prompt=system_prompt,
            user_message="\n".join(parts),
        )

    def parse_slow_path_response(
        self,
        llm_response: dict[str, Any],
    ) -> LearningSignal:
        predictability = llm_response.get("predictability", 0.5)
        complexity = llm_response.get("complexity", 0.5)

        return LearningSignal(
            task_labels={
                "response_predictability": float(predictability),
                "response_complexity": float(complexity),
            },
            outcome_text=llm_response.get("reasoning", ""),
            metadata={
                "response_type": llm_response.get("response_type", "general"),
                "cacheable": llm_response.get("cacheable", False),
            },
        )

    def fast_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
    ) -> dict[str, Any]:
        predictability = prediction.task_outputs.get("response_predictability", 0.5)
        complexity = prediction.task_outputs.get("response_complexity", 0.5)

        can_skip_llm = predictability > 0.7 and complexity < 0.5

        return {
            "predictability": predictability,
            "complexity": complexity,
            "can_skip_llm": can_skip_llm,
            "confidence": prediction.confidence,
            "recommendation": (
                "use_skill" if can_skip_llm else
                "call_llm"
            ),
        }

    def slow_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
        llm_response: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "predictability": llm_response.get("predictability", 0.5),
            "complexity": llm_response.get("complexity", 0.5),
            "response_type": llm_response.get("response_type", "general"),
            "cacheable": llm_response.get("cacheable", False),
            "confidence": prediction.confidence,
            "reasoning": llm_response.get("reasoning", ""),
        }
