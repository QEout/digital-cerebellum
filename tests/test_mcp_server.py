"""Tests for the MCP Server tool functions."""
import pytest

# Reset the global singleton between tests
import digital_cerebellum.mcp_server as mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    mod._cb = None
    yield
    mod._cb = None


class TestMCPTools:
    def test_evaluate_tool_call_returns_required_fields(self):
        result = mod.evaluate_tool_call(
            "send_email", {"to": "alice@example.com", "body": "hi"}
        )
        assert isinstance(result, dict)
        assert "safe" in result or "prediction" in result
        assert "confidence" in result
        assert "_route" in result
        assert "_event_id" in result

    def test_evaluate_payment_returns_required_fields(self):
        result = mod.evaluate_payment(100.0, "USD", "amazon.com", "credit_card")
        assert isinstance(result, dict)
        assert "confidence" in result
        assert "_route" in result
        assert "_event_id" in result

    def test_feedback_accepts_success(self):
        r = mod.evaluate_tool_call("ls", {"path": "."})
        resp = mod.feedback(r["_event_id"], True)
        assert resp["status"] == "ok"

    def test_feedback_accepts_failure(self):
        r = mod.evaluate_tool_call("rm", {"path": "/etc/passwd"})
        resp = mod.feedback(r["_event_id"], False)
        assert resp["status"] == "ok"

    def test_introspect_returns_self_report(self):
        r = mod.evaluate_tool_call("ls", {"path": "."})
        mod.feedback(r["_event_id"], True)

        report = mod.introspect()
        assert "competencies" in report
        assert "prompt" in report
        assert isinstance(report["prompt"], str)

    def test_introspect_domain_filter(self):
        mod.evaluate_tool_call("ls", {"path": "."})
        report = mod.introspect("tool_call")
        assert "competencies" in report

    def test_get_stats_returns_dict(self):
        stats = mod.get_stats()
        assert isinstance(stats, dict)
        assert "step" in stats
        assert "microzones" in stats

    def test_get_curiosity_ranking_returns_list(self):
        mod.evaluate_tool_call("x", {"a": 1})
        ranking = mod.get_curiosity_ranking()
        assert isinstance(ranking, list)
        if ranking:
            assert "domain" in ranking[0]
            assert "curiosity_score" in ranking[0]


class TestSanitize:
    def test_numpy_float(self):
        import numpy as np
        assert isinstance(mod._sanitize(np.float64(3.14)), float)

    def test_numpy_int(self):
        import numpy as np
        assert isinstance(mod._sanitize(np.int64(42)), int)

    def test_numpy_array(self):
        import numpy as np
        assert mod._sanitize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_nested_dict(self):
        import numpy as np
        data = {"a": np.float32(1.5), "b": [np.int64(2)]}
        result = mod._sanitize(data)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"][0], int)
