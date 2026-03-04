"""
Digital Cerebellum MCP Server.

Exposes the Digital Cerebellum as an MCP (Model Context Protocol) server,
so any MCP-compatible AI agent (Claude Desktop, Cursor, etc.) can use the
cerebellum for tool-call safety evaluation, payment risk assessment,
metacognitive self-reports, and more.

Usage (stdio transport, for Claude Desktop / Cursor):
    python -m digital_cerebellum.mcp_server

Usage (HTTP transport, for remote clients):
    python -m digital_cerebellum.mcp_server --http
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.microzones.payment import PaymentMicrozone

log = logging.getLogger(__name__)

mcp = FastMCP(
    "Digital Cerebellum",
    instructions=(
        "A neuroscience-inspired prediction-correction engine for AI agents. "
        "Evaluates tool calls for safety, assesses payment risk, provides "
        "metacognitive self-reports, and learns online from feedback."
    ),
)

_cb: DigitalCerebellum | None = None


def _get_cerebellum() -> DigitalCerebellum:
    """Lazy-init the cerebellum singleton."""
    global _cb
    if _cb is None:
        try:
            cfg = CerebellumConfig.from_yaml()
        except Exception:
            cfg = CerebellumConfig()

        cfg.enable_somatic_marker = True
        cfg.enable_curiosity_drive = True
        cfg.enable_self_model = True

        _cb = DigitalCerebellum(cfg)
        _cb.register_microzone(ToolCallMicrozone())
        _cb.register_microzone(PaymentMicrozone())
        log.info("Digital Cerebellum initialized with microzones: %s",
                 list(_cb._microzones.keys()))
    return _cb


# ======================================================================
# Tools
# ======================================================================

@mcp.tool()
def evaluate_tool_call(
    tool_name: str,
    tool_params: dict[str, Any],
    context: str = "",
) -> dict[str, Any]:
    """
    Evaluate a tool call for safety before execution.

    The cerebellum predicts whether this tool call is safe based on
    patterns learned from previous evaluations. High-confidence predictions
    bypass the LLM (fast path, <10ms); uncertain cases consult the LLM.

    Args:
        tool_name: Name of the tool being called (e.g., "send_email", "delete_file")
        tool_params: Parameters passed to the tool (e.g., {"to": "alice", "body": "hi"})
        context: Optional conversation context for better evaluation

    Returns:
        Dictionary with safety assessment, confidence, route taken, gut feeling,
        curiosity signal, and event_id for feedback.
    """
    cb = _get_cerebellum()
    result = cb.evaluate("tool_call", {
        "tool_name": tool_name,
        "tool_params": tool_params,
    }, context=context)
    return _sanitize(result)


@mcp.tool()
def evaluate_payment(
    amount: float,
    currency: str = "USD",
    recipient: str = "",
    method: str = "credit_card",
    context: str = "",
) -> dict[str, Any]:
    """
    Evaluate a payment transaction for risk.

    Assesses whether a payment is likely legitimate or fraudulent/risky
    based on amount, recipient, method, and learned patterns.

    Args:
        amount: Payment amount
        currency: Currency code (e.g., "USD", "EUR", "BTC")
        recipient: Payment recipient identifier
        method: Payment method (e.g., "credit_card", "bank_transfer", "crypto")
        context: Optional additional context

    Returns:
        Dictionary with risk assessment, confidence, and event_id for feedback.
    """
    cb = _get_cerebellum()
    result = cb.evaluate("payment", {
        "amount": amount,
        "currency": currency,
        "recipient": recipient,
        "method": method,
    }, context=context)
    return _sanitize(result)


@mcp.tool()
def feedback(event_id: str, success: bool) -> dict[str, str]:
    """
    Provide post-execution feedback for a previous evaluation.

    This is how the cerebellum learns: after a tool call or payment is
    executed, tell the cerebellum whether it succeeded. This drives
    online learning and improves future predictions.

    Args:
        event_id: The _event_id from the evaluation result
        success: Whether the execution was successful (True) or failed (False)

    Returns:
        Confirmation message.
    """
    cb = _get_cerebellum()
    cb.feedback(event_id, success)
    return {"status": "ok", "message": f"Feedback recorded: {'success' if success else 'failure'}"}


@mcp.tool()
def introspect(domain: str | None = None) -> dict[str, Any]:
    """
    Get a metacognitive self-report from the cerebellum.

    The cerebellum tracks its own competency across domains and can
    report: skill levels, accuracy, calibration error, strengths,
    weaknesses, and recommendations.

    Args:
        domain: Optional domain name to introspect (e.g., "tool_call", "payment").
                If not provided, reports on all tracked domains.

    Returns:
        Self-assessment with competency profiles, strengths, weaknesses,
        and adaptive threshold recommendations.
    """
    cb = _get_cerebellum()
    report = cb.introspect(domain)
    return {
        "competencies": {
            name: {
                "skill_level": cp.skill_level,
                "accuracy": round(cp.accuracy, 3),
                "calibration_error": round(cp.calibration_error, 3),
                "fast_path_ratio": round(cp.fast_path_ratio, 3),
                "learning_trend": cp.learning_trend,
                "total_observations": cp.total_observations,
            }
            for name, cp in report.competencies.items()
        },
        "strengths": report.strengths,
        "weaknesses": report.weaknesses,
        "overall_calibration": round(report.overall_calibration, 3),
        "recommendation": report.recommendation,
        "prompt": report.to_prompt(),
    }


@mcp.tool()
def get_stats() -> dict[str, Any]:
    """
    Get system statistics from the cerebellum.

    Returns step count, registered microzones, routing breakdown,
    memory stats, and Phase 3 emergence metrics (somatic markers,
    curiosity, self-model).
    """
    cb = _get_cerebellum()
    return _sanitize(cb.stats)


@mcp.tool()
def get_curiosity_ranking() -> list[dict[str, Any]]:
    """
    Get the curiosity-based exploration ranking.

    Returns domains ranked by learning potential — which areas
    the cerebellum is most actively learning about and would
    benefit from more practice.

    Returns:
        List of domains with their curiosity scores, sorted by
        learning potential (highest first).
    """
    cb = _get_cerebellum()
    ranking = cb.curiosity_drive.get_exploration_ranking()
    return [
        {"domain": domain, "curiosity_score": round(score, 4)}
        for domain, score in ranking
    ]


# ======================================================================
# Resources
# ======================================================================

@mcp.resource("cerebellum://status")
def cerebellum_status() -> str:
    """Current status of the Digital Cerebellum."""
    cb = _get_cerebellum()
    stats = cb.stats
    lines = [
        "Digital Cerebellum Status",
        f"  Step: {stats['step']}",
        f"  Microzones: {', '.join(stats['microzones'])}",
        f"  Routes: {stats['routes']}",
    ]
    if "somatic_marker" in stats:
        lines.append(f"  Somatic markers: {stats['somatic_marker']['count']}")
    return "\n".join(lines)


@mcp.resource("cerebellum://self-report")
def cerebellum_self_report() -> str:
    """Metacognitive self-report in natural language."""
    cb = _get_cerebellum()
    report = cb.introspect()
    return report.to_prompt()


# ======================================================================
# Prompts
# ======================================================================

@mcp.prompt()
def safety_check_prompt(tool_name: str, tool_params: str) -> str:
    """Generate a prompt for checking tool-call safety with the cerebellum."""
    return (
        f"Please evaluate the safety of this tool call using the Digital Cerebellum:\n\n"
        f"Tool: {tool_name}\n"
        f"Parameters: {tool_params}\n\n"
        f"Use the evaluate_tool_call tool to get a safety assessment, then report "
        f"the confidence level and whether the call should proceed."
    )


# ======================================================================
# Helpers
# ======================================================================

def _sanitize(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


# ======================================================================
# Entry point
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Digital Cerebellum MCP Server")
    parser.add_argument("--http", action="store_true",
                        help="Use HTTP transport (default: stdio)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port (default: 8000)")
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
