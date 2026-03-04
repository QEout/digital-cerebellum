"""Tests for Phase 5 microzones: shell_command, file_operation, api_call, response_prediction."""

import numpy as np
import pytest

from digital_cerebellum.core.types import PredictionOutput, HeadPrediction
from digital_cerebellum.microzones.shell_command import ShellCommandMicrozone
from digital_cerebellum.microzones.file_operation import FileOperationMicrozone
from digital_cerebellum.microzones.api_call import APICallMicrozone
from digital_cerebellum.microzones.response_prediction import ResponsePredictionMicrozone


def _make_prediction(**task_outputs) -> PredictionOutput:
    """Create a PredictionOutput with given task outputs."""
    hp = HeadPrediction(
        action_embedding=np.zeros(128),
        outcome_embedding=np.zeros(128),
    )
    return PredictionOutput(
        action_embedding=np.zeros(128),
        outcome_embedding=np.zeros(128),
        confidence=0.8,
        head_predictions=[hp],
        task_outputs=task_outputs,
    )


# ======================================================================
# ShellCommandMicrozone
# ======================================================================

class TestShellCommandMicrozone:

    def setup_method(self):
        self.mz = ShellCommandMicrozone()

    def test_name(self):
        assert self.mz.name == "shell_command"

    def test_task_heads(self):
        heads = self.mz.task_heads()
        names = [h.name for h in heads]
        assert "shell_safety" in names
        assert "shell_destructive" in names

    def test_format_input_basic(self):
        text = self.mz.format_input({"command": "ls -la"})
        assert "ls -la" in text
        assert "bash" in text

    def test_format_input_with_context(self):
        text = self.mz.format_input(
            {"command": "git status", "shell": "bash", "working_dir": "/repo"},
            context="user asked for status",
        )
        assert "git status" in text
        assert "/repo" in text
        assert "user asked" in text

    def test_fast_path_safe_command(self):
        pred = _make_prediction(shell_safety=0.9, shell_destructive=0.1)
        result = self.mz.fast_path_evaluate(
            {"command": "ls -la"}, pred,
        )
        assert result["safe"] is True
        assert result["risk_type"] == "none"

    def test_fast_path_dangerous_pattern(self):
        pred = _make_prediction(shell_safety=0.9, shell_destructive=0.1)
        result = self.mz.fast_path_evaluate(
            {"command": "rm -rf /"}, pred,
        )
        assert result["safe"] is False
        assert result["pattern_blocked"] is True

    def test_fast_path_predicted_unsafe(self):
        pred = _make_prediction(shell_safety=0.2, shell_destructive=0.8)
        result = self.mz.fast_path_evaluate(
            {"command": "some_command"}, pred,
        )
        assert result["safe"] is False

    def test_parse_slow_path_response(self):
        signal = self.mz.parse_slow_path_response({
            "safe": False,
            "risk_type": "destructive",
            "severity": "critical",
            "reasoning": "rm -rf deletes everything",
            "expected_outcome": "data loss",
        })
        assert signal.task_labels["shell_safety"] == 0.0
        assert signal.task_labels["shell_destructive"] == 1.0
        assert signal.outcome_text == "data loss"

    def test_slow_path_evaluate(self):
        pred = _make_prediction(shell_safety=0.3, shell_destructive=0.8)
        result = self.mz.slow_path_evaluate(
            {"command": "rm -rf /"},
            pred,
            {"safe": False, "risk_type": "destructive", "severity": "critical",
             "reasoning": "deletes root"},
        )
        assert result["safe"] is False
        assert result["severity"] == "critical"


# ======================================================================
# FileOperationMicrozone
# ======================================================================

class TestFileOperationMicrozone:

    def setup_method(self):
        self.mz = FileOperationMicrozone()

    def test_name(self):
        assert self.mz.name == "file_operation"

    def test_task_heads(self):
        heads = self.mz.task_heads()
        names = [h.name for h in heads]
        assert "file_safety" in names
        assert "file_sensitivity" in names

    def test_format_input_basic(self):
        text = self.mz.format_input({"operation": "read", "path": "/tmp/data.txt"})
        assert "read" in text
        assert "/tmp/data.txt" in text

    def test_fast_path_safe_read(self):
        pred = _make_prediction(file_safety=0.9, file_sensitivity=0.1)
        result = self.mz.fast_path_evaluate(
            {"operation": "read", "path": "/tmp/data.txt"}, pred,
        )
        assert result["safe"] is True

    def test_fast_path_sensitive_path_delete(self):
        pred = _make_prediction(file_safety=0.9, file_sensitivity=0.1)
        result = self.mz.fast_path_evaluate(
            {"operation": "delete", "path": "/etc/passwd"}, pred,
        )
        assert result["safe"] is False
        assert result["path_blocked"] is True

    def test_fast_path_sensitive_path_read(self):
        pred = _make_prediction(file_safety=0.9, file_sensitivity=0.1)
        result = self.mz.fast_path_evaluate(
            {"operation": "read", "path": "/etc/passwd"}, pred,
        )
        assert result["safe"] is True  # reading sensitive path is allowed
        assert result["path_blocked"] is True

    def test_parse_slow_path_response(self):
        signal = self.mz.parse_slow_path_response({
            "safe": False,
            "risk_type": "sensitive_path",
            "severity": "high",
            "reasoning": "system file",
            "expected_outcome": "permission denied",
        })
        assert signal.task_labels["file_safety"] == 0.0
        assert signal.task_labels["file_sensitivity"] == 0.8


# ======================================================================
# APICallMicrozone
# ======================================================================

class TestAPICallMicrozone:

    def setup_method(self):
        self.mz = APICallMicrozone()

    def test_name(self):
        assert self.mz.name == "api_call"

    def test_task_heads(self):
        heads = self.mz.task_heads()
        names = [h.name for h in heads]
        assert "api_safety" in names
        assert "api_data_leak_risk" in names

    def test_format_input_basic(self):
        text = self.mz.format_input({
            "method": "GET",
            "url": "https://api.example.com/users",
        })
        assert "GET" in text
        assert "api.example.com" in text

    def test_fast_path_safe_get(self):
        pred = _make_prediction(api_safety=0.9, api_data_leak_risk=0.1)
        result = self.mz.fast_path_evaluate({
            "method": "GET",
            "url": "https://api.example.com/users",
        }, pred)
        assert result["safe"] is True

    def test_fast_path_risky_delete(self):
        pred = _make_prediction(api_safety=0.9, api_data_leak_risk=0.1)
        result = self.mz.fast_path_evaluate({
            "method": "DELETE",
            "url": "https://api.example.com/api/v1/users/delete",
        }, pred)
        assert result["safe"] is False
        assert result["endpoint_risky"] is True

    def test_fast_path_data_leak(self):
        pred = _make_prediction(api_safety=0.6, api_data_leak_risk=0.9)
        result = self.mz.fast_path_evaluate({
            "method": "POST",
            "url": "https://api.example.com/data",
        }, pred)
        assert result["risk_type"] == "data_leak"

    def test_parse_slow_path_data_leak(self):
        signal = self.mz.parse_slow_path_response({
            "safe": False,
            "risk_type": "data_leak",
            "severity": "high",
            "reasoning": "PII in request body",
        })
        assert signal.task_labels["api_safety"] == 0.0
        assert signal.task_labels["api_data_leak_risk"] == 1.0

    def test_build_slow_path_masks_auth_headers(self):
        pred = _make_prediction(api_safety=0.5, api_data_leak_risk=0.5)
        req = self.mz.build_slow_path_request(
            {"method": "GET", "url": "https://api.example.com",
             "headers": {"Authorization": "Bearer secret123", "Content-Type": "application/json"}},
            context="",
            prediction=pred,
        )
        assert "secret123" not in req.user_message
        assert "***" in req.user_message


# ======================================================================
# ResponsePredictionMicrozone
# ======================================================================

class TestResponsePredictionMicrozone:

    def setup_method(self):
        self.mz = ResponsePredictionMicrozone()

    def test_name(self):
        assert self.mz.name == "response_prediction"

    def test_task_heads(self):
        heads = self.mz.task_heads()
        names = [h.name for h in heads]
        assert "response_predictability" in names
        assert "response_complexity" in names

    def test_format_input(self):
        text = self.mz.format_input({
            "query": "What is 2+2?",
            "query_type": "factual",
        })
        assert "2+2" in text
        assert "factual" in text

    def test_fast_path_predictable(self):
        pred = _make_prediction(response_predictability=0.9, response_complexity=0.2)
        result = self.mz.fast_path_evaluate(
            {"query": "What is 2+2?"}, pred,
        )
        assert result["can_skip_llm"] is True
        assert result["recommendation"] == "use_skill"

    def test_fast_path_complex(self):
        pred = _make_prediction(response_predictability=0.3, response_complexity=0.9)
        result = self.mz.fast_path_evaluate(
            {"query": "Explain quantum entanglement"}, pred,
        )
        assert result["can_skip_llm"] is False
        assert result["recommendation"] == "call_llm"

    def test_parse_slow_path_response(self):
        signal = self.mz.parse_slow_path_response({
            "predictability": 0.8,
            "complexity": 0.3,
            "response_type": "factual",
            "cacheable": True,
            "reasoning": "simple factual lookup",
        })
        assert signal.task_labels["response_predictability"] == 0.8
        assert signal.task_labels["response_complexity"] == 0.3
        assert signal.metadata["cacheable"] is True


# ======================================================================
# Integration: Register all microzones on one engine
# ======================================================================

class TestAllMicrozonesIntegration:

    def test_register_all(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        from digital_cerebellum.microzones import ALL_MICROZONES

        cb = DigitalCerebellum(CerebellumConfig())
        for mz_cls in ALL_MICROZONES:
            cb.register_microzone(mz_cls())

        assert len(cb._microzones) == 6
        assert "tool_call" in cb._microzones
        assert "payment" in cb._microzones
        assert "shell_command" in cb._microzones
        assert "file_operation" in cb._microzones
        assert "api_call" in cb._microzones
        assert "response_prediction" in cb._microzones

    def test_evaluate_each_fast_path(self):
        """Evaluate each microzone — hits fast path (no LLM needed)."""
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        from digital_cerebellum.microzones import ALL_MICROZONES

        cfg = CerebellumConfig()
        cfg.threshold_high = 0.0  # force fast path for all
        cfg.threshold_low = 0.0
        cb = DigitalCerebellum(cfg)
        for mz_cls in ALL_MICROZONES:
            cb.register_microzone(mz_cls())

        payloads = {
            "tool_call": {"tool_name": "search", "tool_params": {"q": "test"}},
            "payment": {"amount": 10, "currency": "USD", "recipient": "alice"},
            "shell_command": {"command": "echo hello"},
            "file_operation": {"operation": "read", "path": "/tmp/test.txt"},
            "api_call": {"method": "GET", "url": "https://api.example.com"},
            "response_prediction": {"query": "What is 2+2?", "query_type": "factual"},
        }

        for zone_name, payload in payloads.items():
            result = cb.evaluate(zone_name, payload)
            assert "_route" in result, f"{zone_name} missing _route"
            assert "_latency_ms" in result, f"{zone_name} missing _latency_ms"
            assert "_event_id" in result, f"{zone_name} missing _event_id"
            assert result["_route"] == "fast", f"{zone_name} expected fast path"

    def test_all_task_heads_unique(self):
        from digital_cerebellum.microzones import ALL_MICROZONES

        all_names = []
        for mz_cls in ALL_MICROZONES:
            mz = mz_cls()
            for head in mz.task_heads():
                all_names.append(head.name)

        assert len(all_names) == len(set(all_names)), (
            f"Duplicate task head names: {all_names}"
        )
