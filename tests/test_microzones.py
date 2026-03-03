"""
Microzone integration tests — verify pluggable microzones work
with the core engine without any core modifications.

Run with:  pytest tests/test_microzones.py -v
"""

import numpy as np
import pytest

from digital_cerebellum.core.prediction_engine import EngineConfig, PredictionEngine
from digital_cerebellum.core.microzone import TaskHeadConfig
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.microzones.payment import PaymentMicrozone


class TestToolCallMicrozone:
    def test_name(self):
        mz = ToolCallMicrozone()
        assert mz.name == "tool_call"

    def test_task_heads(self):
        mz = ToolCallMicrozone()
        heads = mz.task_heads()
        assert len(heads) == 1
        assert heads[0].name == "safety"
        assert heads[0].activation == "sigmoid"

    def test_format_input(self):
        mz = ToolCallMicrozone()
        text = mz.format_input(
            {"tool_name": "send_email", "tool_params": {"to": "alice"}},
            context="user request",
        )
        assert "send_email" in text
        assert "alice" in text
        assert "user request" in text

    def test_format_input_no_context(self):
        mz = ToolCallMicrozone()
        text = mz.format_input({"tool_name": "read_file", "tool_params": {"path": "/tmp"}})
        assert "read_file" in text
        assert "->" not in text

    def test_fast_path_safe(self):
        mz = ToolCallMicrozone()
        engine = PredictionEngine(EngineConfig(rff_dim=256, action_dim=32, outcome_dim=32))
        engine.register_task_head("safety", 1, "sigmoid")
        z = np.random.randn(256).astype(np.float32)
        pred = engine.predict_numpy(z)
        result = mz.fast_path_evaluate({"tool_name": "test"}, pred)
        assert "safe" in result
        assert "confidence" in result
        assert isinstance(result["safe"], bool)


class TestPaymentMicrozone:
    def test_name(self):
        mz = PaymentMicrozone()
        assert mz.name == "payment"

    def test_task_heads(self):
        mz = PaymentMicrozone()
        heads = mz.task_heads()
        assert len(heads) == 1
        assert heads[0].name == "payment_risk"
        assert heads[0].activation == "sigmoid"

    def test_format_input_basic(self):
        mz = PaymentMicrozone()
        text = mz.format_input({
            "action": "charge",
            "amount": 49.99,
            "currency": "USD",
            "recipient": "Netflix",
            "reason": "monthly subscription",
        })
        assert "charge" in text
        assert "49.99" in text
        assert "Netflix" in text
        assert "monthly subscription" in text

    def test_format_input_with_context(self):
        mz = PaymentMicrozone()
        text = mz.format_input(
            {"amount": 100, "recipient": "Bob"},
            context="user asked to split dinner bill",
        )
        assert "user asked" in text
        assert "Bob" in text

    def test_fast_path_low_risk(self):
        mz = PaymentMicrozone()
        engine = PredictionEngine(EngineConfig(rff_dim=256, action_dim=32, outcome_dim=32))
        engine.register_task_head("payment_risk", 1, "sigmoid")
        z = np.random.randn(256).astype(np.float32)
        pred = engine.predict_numpy(z)
        # Force a low-risk prediction
        pred.task_outputs["payment_risk"] = 0.1
        result = mz.fast_path_evaluate({"amount": 5}, pred)
        assert result["approved"] is True
        assert result["risk_level"] == "low"

    def test_fast_path_high_risk(self):
        mz = PaymentMicrozone()
        engine = PredictionEngine(EngineConfig(rff_dim=256, action_dim=32, outcome_dim=32))
        engine.register_task_head("payment_risk", 1, "sigmoid")
        z = np.random.randn(256).astype(np.float32)
        pred = engine.predict_numpy(z)
        pred.task_outputs["payment_risk"] = 0.75
        result = mz.fast_path_evaluate({"amount": 50000}, pred)
        assert result["approved"] is False
        assert result["risk_level"] == "high"

    def test_parse_slow_path_response(self):
        mz = PaymentMicrozone()
        signal = mz.parse_slow_path_response({
            "approved": False,
            "risk_level": "high",
            "risk_factors": ["amount exceeds typical range"],
            "reasoning": "Large unexpected payment",
        })
        assert signal.task_labels["payment_risk"] == 0.8
        assert signal.metadata["risk_level"] == "high"
        assert len(signal.metadata["risk_factors"]) == 1

    def test_parse_slow_path_approved(self):
        mz = PaymentMicrozone()
        signal = mz.parse_slow_path_response({
            "approved": True,
            "risk_level": "none",
            "risk_factors": [],
            "reasoning": "Routine small payment",
        })
        assert signal.task_labels["payment_risk"] == 0.0


class TestMultipleMicrozonesOnSameEngine:
    """Verify two microzones can share a single PredictionEngine."""

    def test_register_both(self):
        engine = PredictionEngine(EngineConfig(rff_dim=256, action_dim=32, outcome_dim=32))
        tc = ToolCallMicrozone()
        pm = PaymentMicrozone()

        for head_cfg in tc.task_heads():
            engine.register_task_head(head_cfg.name, head_cfg.output_dim, head_cfg.activation)
        for head_cfg in pm.task_heads():
            engine.register_task_head(head_cfg.name, head_cfg.output_dim, head_cfg.activation)

        assert "safety" in engine.task_heads
        assert "payment_risk" in engine.task_heads

    def test_predict_produces_both_outputs(self):
        engine = PredictionEngine(EngineConfig(rff_dim=256, action_dim=32, outcome_dim=32))
        engine.register_task_head("safety", 1, "sigmoid")
        engine.register_task_head("payment_risk", 1, "sigmoid")

        z = np.random.randn(256).astype(np.float32)
        pred = engine.predict_numpy(z)
        assert "safety" in pred.task_outputs
        assert "payment_risk" in pred.task_outputs
        assert 0.0 <= pred.task_outputs["safety"] <= 1.0
        assert 0.0 <= pred.task_outputs["payment_risk"] <= 1.0
