"""
API Call Microzone — cerebellar pre-evaluation for external API requests.

When an LLM agent makes HTTP requests to external APIs, the cerebellum
evaluates: is this endpoint legitimate?  Are the parameters reasonable?
Is the request rate normal?  Could this leak sensitive data?

Biological analogy: the cerebellum's forward model predicts the sensory
consequences of actions before they happen.  This microzone predicts
the consequences of API calls — timeouts, errors, data leaks — before
the network request is sent.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from digital_cerebellum.core.microzone import (
    LearningSignal,
    Microzone,
    SlowPathRequest,
    TaskHeadConfig,
)
from digital_cerebellum.core.types import PredictionOutput

SENSITIVE_HEADERS = frozenset({
    "authorization", "x-api-key", "cookie",
    "x-auth-token", "x-csrf-token",
})

RISKY_ENDPOINTS = frozenset({
    "/admin", "/api/v1/users/delete", "/api/v1/billing",
    "/oauth/token", "/auth/reset-password",
})


class APICallMicrozone(Microzone):
    """Microzone for evaluating external API call safety."""

    @property
    def name(self) -> str:
        return "api_call"

    def task_heads(self) -> list[TaskHeadConfig]:
        return [
            TaskHeadConfig(name="api_safety", output_dim=1, activation="sigmoid"),
            TaskHeadConfig(name="api_data_leak_risk", output_dim=1, activation="sigmoid"),
        ]

    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        method = payload.get("method", "GET").upper()
        url = payload.get("url", "")
        body_preview = payload.get("body_preview", "")

        text = f"{method} {url}"
        if body_preview:
            text += f" body={body_preview[:100]}"
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
            "You are an API call safety evaluator for an AI agent. "
            "Given an HTTP request, assess whether it is safe to send.\n"
            "Consider: data leakage (PII in URL/body), authentication abuse, "
            "destructive methods (DELETE, PUT to critical endpoints), "
            "rate limiting risks, and unknown/suspicious domains.\n"
            "Respond ONLY with a JSON object: "
            '{"safe": bool, '
            '"risk_type": "none"|"data_leak"|"auth_abuse"|"destructive"|"rate_limit"|"suspicious_domain", '
            '"severity": "none"|"low"|"medium"|"high"|"critical", '
            '"reasoning": "...", "expected_outcome": "..."}'
        )

        parts = [
            f"Method: {payload.get('method', 'GET')}",
            f"URL: {payload.get('url', '')}",
        ]
        if payload.get("headers"):
            safe_headers = {
                k: ("***" if k.lower() in SENSITIVE_HEADERS else v)
                for k, v in payload["headers"].items()
            }
            parts.append(f"Headers: {json.dumps(safe_headers)}")
        if payload.get("body_preview"):
            parts.append(f"Body: {payload['body_preview'][:300]}")
        if context:
            parts.append(f"Context: {context}")

        safety = prediction.task_outputs.get("api_safety")
        if safety is not None:
            parts.append(
                f"Cerebellar hint: safety={safety:.3f}, "
                f"data_leak={prediction.task_outputs.get('api_data_leak_risk', 0):.3f}, "
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
        safe = llm_response.get("safe", True)
        severity = llm_response.get("severity", "none")
        risk_type = llm_response.get("risk_type", "none")

        severity_score = {
            "none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0,
        }.get(severity, 0.5)

        data_leak_score = 1.0 if risk_type == "data_leak" else severity_score * 0.5

        return LearningSignal(
            task_labels={
                "api_safety": 1.0 if safe else 0.0,
                "api_data_leak_risk": data_leak_score,
            },
            outcome_text=llm_response.get("expected_outcome", ""),
            metadata={
                "risk_type": risk_type,
                "severity": severity,
                "reasoning": llm_response.get("reasoning", ""),
            },
        )

    def fast_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
    ) -> dict[str, Any]:
        safety = prediction.task_outputs.get("api_safety", 0.5)
        data_leak = prediction.task_outputs.get("api_data_leak_risk", 0.5)

        url = payload.get("url", "")
        method = payload.get("method", "GET").upper()

        try:
            parsed = urlparse(url)
            path = parsed.path
        except Exception:
            path = ""

        endpoint_risky = any(r in path for r in RISKY_ENDPOINTS)
        destructive_method = method in ("DELETE", "PUT", "PATCH")

        safe = safety > 0.5 and not (endpoint_risky and destructive_method)

        return {
            "safe": safe,
            "risk_type": (
                "destructive" if endpoint_risky and destructive_method else
                "data_leak" if data_leak > 0.7 else
                "none" if safe else "predicted_unsafe"
            ),
            "safety_score": safety,
            "data_leak_score": data_leak,
            "confidence": prediction.confidence,
            "endpoint_risky": endpoint_risky,
        }

    def slow_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
        llm_response: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "safe": llm_response.get("safe", True),
            "risk_type": llm_response.get("risk_type", "none"),
            "severity": llm_response.get("severity", "none"),
            "safety_score": prediction.task_outputs.get("api_safety", 0.5),
            "data_leak_score": prediction.task_outputs.get("api_data_leak_risk", 0.5),
            "confidence": prediction.confidence,
            "reasoning": llm_response.get("reasoning", ""),
        }
