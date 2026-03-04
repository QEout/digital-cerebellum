"""
File Operation Microzone — cerebellar pre-evaluation for filesystem actions.

Evaluates file operations (read, write, delete, move, chmod) before
execution.  Learns patterns of safe vs dangerous file operations —
e.g. writing to /tmp is usually fine, but writing to /etc/passwd is not.

Biological analogy: the cerebellum learns spatial boundaries for limb
movements (don't reach into fire).  This microzone learns filesystem
boundaries (don't write to system directories).
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

SENSITIVE_PATHS = frozenset({
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/hosts", "/etc/ssh", "/root/.ssh",
    "~/.ssh/id_rsa", "~/.ssh/authorized_keys",
    ".env", "credentials.json", "secrets.yaml",
    "C:\\Windows\\System32", "C:\\Windows\\system.ini",
})

DESTRUCTIVE_OPERATIONS = frozenset({
    "delete", "remove", "rm", "rmdir", "truncate", "overwrite",
})


class FileOperationMicrozone(Microzone):
    """Microzone for evaluating filesystem operation safety."""

    @property
    def name(self) -> str:
        return "file_operation"

    def task_heads(self) -> list[TaskHeadConfig]:
        return [
            TaskHeadConfig(name="file_safety", output_dim=1, activation="sigmoid"),
            TaskHeadConfig(name="file_sensitivity", output_dim=1, activation="sigmoid"),
        ]

    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        operation = payload.get("operation", "read")
        path = payload.get("path", "")
        content_preview = payload.get("content_preview", "")

        text = f"{operation} {path}"
        if content_preview:
            text += f" [{content_preview[:100]}]"
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
            "You are a filesystem safety evaluator for an AI agent. "
            "Given a file operation, assess whether it is safe to execute.\n"
            "Consider: sensitive paths (credentials, system files), "
            "destructive operations (delete, overwrite), "
            "data exposure risks, and permission boundaries.\n"
            "Respond ONLY with a JSON object: "
            '{"safe": bool, '
            '"risk_type": "none"|"sensitive_path"|"destructive"|"data_exposure"|"permission_violation", '
            '"severity": "none"|"low"|"medium"|"high"|"critical", '
            '"reasoning": "...", "expected_outcome": "..."}'
        )

        parts = [
            f"Operation: {payload.get('operation', 'read')}",
            f"Path: {payload.get('path', '')}",
        ]
        if payload.get("content_preview"):
            parts.append(f"Content preview: {payload['content_preview'][:200]}")
        if payload.get("permissions"):
            parts.append(f"Permissions: {payload['permissions']}")
        if context:
            parts.append(f"Context: {context}")

        safety = prediction.task_outputs.get("file_safety")
        if safety is not None:
            parts.append(
                f"Cerebellar hint: safety={safety:.3f}, "
                f"sensitivity={prediction.task_outputs.get('file_sensitivity', 0):.3f}, "
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

        severity_score = {
            "none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0,
        }.get(severity, 0.5)

        return LearningSignal(
            task_labels={
                "file_safety": 1.0 if safe else 0.0,
                "file_sensitivity": severity_score,
            },
            outcome_text=llm_response.get("expected_outcome", ""),
            metadata={
                "risk_type": llm_response.get("risk_type", "none"),
                "severity": severity,
                "reasoning": llm_response.get("reasoning", ""),
            },
        )

    def fast_path_evaluate(
        self,
        payload: dict[str, Any],
        prediction: PredictionOutput,
    ) -> dict[str, Any]:
        safety = prediction.task_outputs.get("file_safety", 0.5)
        sensitivity = prediction.task_outputs.get("file_sensitivity", 0.5)

        path = payload.get("path", "")
        operation = payload.get("operation", "read")

        path_blocked = any(s in path for s in SENSITIVE_PATHS)
        destructive = operation.lower() in DESTRUCTIVE_OPERATIONS

        safe = safety > 0.5 and not (path_blocked and destructive)

        return {
            "safe": safe,
            "risk_type": (
                "sensitive_path" if path_blocked else
                "destructive" if destructive and safety < 0.5 else
                "none" if safe else "predicted_unsafe"
            ),
            "safety_score": safety,
            "sensitivity_score": sensitivity,
            "confidence": prediction.confidence,
            "path_blocked": path_blocked,
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
            "safety_score": prediction.task_outputs.get("file_safety", 0.5),
            "sensitivity_score": prediction.task_outputs.get("file_sensitivity", 0.5),
            "confidence": prediction.confidence,
            "reasoning": llm_response.get("reasoning", ""),
        }
