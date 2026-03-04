"""
Shell Command Microzone — cerebellar pre-evaluation for shell operations.

When an LLM agent executes shell commands (bash, PowerShell, etc.),
the cerebellum predicts whether the command is safe, identifies
destructive patterns, and learns from outcomes.

Biological analogy: just as the cerebellum learns to inhibit dangerous
motor programs (e.g., pulling hand from fire becomes automatic),
this microzone learns to recognise dangerous shell patterns
(rm -rf /, DROP TABLE, etc.) and block them before execution.
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

DANGEROUS_PATTERNS = frozenset({
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){:|:&};:",
    "chmod -R 777 /", "chown -R", "> /dev/sda",
    "DROP DATABASE", "DROP TABLE", "DELETE FROM",
    "shutdown", "reboot", "halt", "init 0",
    "wget|sh", "curl|bash", "eval(",
})


class ShellCommandMicrozone(Microzone):
    """Microzone for evaluating shell command safety."""

    @property
    def name(self) -> str:
        return "shell_command"

    def task_heads(self) -> list[TaskHeadConfig]:
        return [
            TaskHeadConfig(name="shell_safety", output_dim=1, activation="sigmoid"),
            TaskHeadConfig(name="shell_destructive", output_dim=1, activation="sigmoid"),
        ]

    def format_input(self, payload: dict[str, Any], context: str = "") -> str:
        command = payload.get("command", "")
        working_dir = payload.get("working_dir", "")
        shell = payload.get("shell", "bash")

        text = f"[{shell}] {command}"
        if working_dir:
            text = f"({working_dir}) {text}"
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
            "You are a shell command safety evaluator for an AI agent. "
            "Given a command, assess whether it is safe to execute.\n"
            "Consider: destructive operations (rm, format, drop), "
            "privilege escalation (sudo, chmod), data exfiltration (curl, wget piped), "
            "and irreversibility.\n"
            "Respond ONLY with a JSON object: "
            '{"safe": bool, '
            '"risk_type": "none"|"destructive"|"privilege_escalation"|"data_exfiltration"|"resource_abuse"|"unknown_binary", '
            '"severity": "none"|"low"|"medium"|"high"|"critical", '
            '"reasoning": "...", "expected_outcome": "..."}'
        )

        parts = [
            f"Command: {payload.get('command', '')}",
            f"Shell: {payload.get('shell', 'bash')}",
        ]
        if payload.get("working_dir"):
            parts.append(f"Working Directory: {payload['working_dir']}")
        if context:
            parts.append(f"Context: {context}")

        safety = prediction.task_outputs.get("shell_safety")
        destructive = prediction.task_outputs.get("shell_destructive")
        if safety is not None:
            parts.append(
                f"Cerebellar hint: safety={safety:.3f}, "
                f"destructive={destructive:.3f}, "
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
                "shell_safety": 1.0 if safe else 0.0,
                "shell_destructive": severity_score,
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
        safety = prediction.task_outputs.get("shell_safety", 0.5)
        destructive = prediction.task_outputs.get("shell_destructive", 0.5)

        command = payload.get("command", "")
        pattern_blocked = any(p in command for p in DANGEROUS_PATTERNS)

        safe = safety > 0.5 and not pattern_blocked

        return {
            "safe": safe,
            "risk_type": "destructive" if pattern_blocked else (
                "none" if safe else "predicted_unsafe"
            ),
            "destructive_score": destructive,
            "safety_score": safety,
            "confidence": prediction.confidence,
            "pattern_blocked": pattern_blocked,
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
            "safety_score": prediction.task_outputs.get("shell_safety", 0.5),
            "destructive_score": prediction.task_outputs.get("shell_destructive", 0.5),
            "confidence": prediction.confidence,
            "reasoning": llm_response.get("reasoning", ""),
        }
