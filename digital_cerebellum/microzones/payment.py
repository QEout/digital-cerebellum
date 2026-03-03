"""
Payment Risk Microzone — cerebellar pre-evaluation for financial operations.

When an LLM agent is about to make a payment (HTTP 402, Stripe charge,
crypto transfer, etc.), the cerebellum predicts whether the amount,
recipient, and context are reasonable *before* the irreversible action.

This is the second microzone, demonstrating that the same cerebellar
engine generalises across domains with zero core changes.
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


class PaymentMicrozone(Microzone):
    """Microzone for evaluating payment / financial operation risk."""

    @property
    def name(self) -> str:
        return "payment"

    def task_heads(self) -> list[TaskHeadConfig]:
        return [
            TaskHeadConfig(name="payment_risk", output_dim=1, activation="sigmoid"),
        ]

    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        action = payload.get("action", "pay")
        amount = payload.get("amount", 0)
        currency = payload.get("currency", "USD")
        recipient = payload.get("recipient", "unknown")
        reason = payload.get("reason", "")

        text = f"{action} {amount} {currency} to {recipient}"
        if reason:
            text += f" for {reason}"
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
            "You are a payment risk evaluator for an AI agent. "
            "Given a financial operation, assess whether it is reasonable "
            "and safe to execute. Consider: amount vs typical range, "
            "recipient legitimacy, context appropriateness, irreversibility.\n"
            "Respond ONLY with a JSON object: "
            '{"approved": bool, '
            '"risk_level": "none"|"low"|"medium"|"high"|"critical", '
            '"risk_factors": ["..."], '
            '"reasoning": "...", '
            '"suggested_limit": null or number}'
        )

        parts = [
            f"Action: {payload.get('action', 'pay')}",
            f"Amount: {payload.get('amount', 0)} {payload.get('currency', 'USD')}",
            f"Recipient: {payload.get('recipient', 'unknown')}",
        ]
        if payload.get("reason"):
            parts.append(f"Reason: {payload['reason']}")
        if payload.get("metadata"):
            parts.append(f"Metadata: {json.dumps(payload['metadata'], ensure_ascii=False)}")
        if context:
            parts.append(f"Context: {context}")

        risk_score = prediction.task_outputs.get("payment_risk")
        if risk_score is not None:
            parts.append(
                f"Cerebellar hint (reference only): "
                f"risk={risk_score:.3f}, "
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
        approved = llm_response.get("approved", True)
        risk_level = llm_response.get("risk_level", "none")

        risk_score = {
            "none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0,
        }.get(risk_level, 0.5)

        return LearningSignal(
            task_labels={"payment_risk": risk_score},
            outcome_text=llm_response.get("reasoning", ""),
            metadata={
                "approved": approved,
                "risk_level": risk_level,
                "risk_factors": llm_response.get("risk_factors", []),
                "suggested_limit": llm_response.get("suggested_limit"),
            },
        )

    def fast_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
    ) -> dict[str, Any]:
        risk = prediction.task_outputs.get("payment_risk", 0.5)
        approved = risk < 0.5
        risk_level = (
            "none" if risk < 0.1 else
            "low" if risk < 0.3 else
            "medium" if risk < 0.6 else
            "high" if risk < 0.85 else
            "critical"
        )
        return {
            "approved": approved,
            "risk_level": risk_level,
            "risk_score": risk,
            "confidence": prediction.confidence,
        }

    def slow_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
        llm_response: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "approved": llm_response.get("approved", True),
            "risk_level": llm_response.get("risk_level", "none"),
            "risk_score": prediction.task_outputs.get("payment_risk", 0.5),
            "risk_factors": llm_response.get("risk_factors", []),
            "confidence": prediction.confidence,
            "reasoning": llm_response.get("reasoning", ""),
            "suggested_limit": llm_response.get("suggested_limit"),
        }
