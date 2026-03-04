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
from digital_cerebellum.microzones import ALL_MICROZONES

log = logging.getLogger(__name__)

mcp = FastMCP(
    "Digital Cerebellum",
    instructions=(
        "A neuroscience-inspired prediction-correction engine for AI agents. "
        "Evaluates tool calls, shell commands, file operations, and API calls "
        "for safety. Learns skills from interactions and executes them directly. "
        "Provides metacognitive self-reports and curiosity-driven exploration."
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
        for mz_cls in ALL_MICROZONES:
            _cb.register_microzone(mz_cls())
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
def evaluate_shell_command(
    command: str,
    shell: str = "bash",
    working_dir: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Evaluate a shell command for safety before execution.

    Detects destructive operations (rm -rf, DROP TABLE), privilege
    escalation (sudo), data exfiltration, and other risky patterns.

    Args:
        command: The shell command to evaluate (e.g., "rm -rf /tmp/build")
        shell: Shell type (e.g., "bash", "powershell")
        working_dir: Current working directory
        context: Optional conversation context

    Returns:
        Safety assessment with risk type, severity, and confidence.
    """
    cb = _get_cerebellum()
    result = cb.evaluate("shell_command", {
        "command": command,
        "shell": shell,
        "working_dir": working_dir,
    }, context=context)
    return _sanitize(result)


@mcp.tool()
def evaluate_file_operation(
    operation: str,
    path: str,
    content_preview: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Evaluate a file operation for safety.

    Detects sensitive path access (credentials, system files),
    destructive operations (delete, overwrite), and data exposure risks.

    Args:
        operation: Type of operation (e.g., "read", "write", "delete", "chmod")
        path: File path being operated on
        content_preview: Optional preview of content being written
        context: Optional conversation context

    Returns:
        Safety assessment with sensitivity score and confidence.
    """
    cb = _get_cerebellum()
    result = cb.evaluate("file_operation", {
        "operation": operation,
        "path": path,
        "content_preview": content_preview,
    }, context=context)
    return _sanitize(result)


@mcp.tool()
def evaluate_api_call(
    method: str,
    url: str,
    body_preview: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Evaluate an external API call for safety.

    Detects data leakage, authentication abuse, destructive endpoints,
    and suspicious domains.

    Args:
        method: HTTP method (e.g., "GET", "POST", "DELETE")
        url: Target URL
        body_preview: Optional preview of request body
        context: Optional conversation context

    Returns:
        Safety assessment with data leak risk score and confidence.
    """
    cb = _get_cerebellum()
    result = cb.evaluate("api_call", {
        "method": method,
        "url": url,
        "body_preview": body_preview,
    }, context=context)
    return _sanitize(result)


@mcp.tool()
def learn_skill(
    input_text: str,
    response_text: str,
    domain: str = "response",
) -> dict[str, str]:
    """
    Teach the cerebellum a new skill from an interaction.

    After the LLM produces a response, store it as a learnable skill.
    Next time a similar query arrives, the cerebellum can respond directly
    without calling the LLM.

    Args:
        input_text: The query/input that triggered the response
        response_text: The response that was produced
        domain: Skill domain (default: "response")

    Returns:
        The skill_id for future reference.
    """
    cb = _get_cerebellum()
    skill_id = cb.learn_skill(input_text, response_text, domain=domain)
    return {"status": "ok", "skill_id": skill_id}


@mcp.tool()
def match_skill(query: str) -> dict[str, Any]:
    """
    Check if the cerebellum has learned a skill for this query.

    If a matching skill exists with high enough confidence, the
    cerebellum can respond directly without calling the LLM.

    Args:
        query: The input text to match against learned skills

    Returns:
        Skill match result with similarity, confidence, and the stored response.
        Returns {"matched": false} if no skill matches.
    """
    cb = _get_cerebellum()
    result = cb.match_skill(query)
    if result is None:
        return {"matched": False}
    return _sanitize({
        "matched": True,
        "similarity": result.similarity,
        "match_confidence": result.match_confidence,
        "should_execute": result.should_execute,
        "response_text": result.skill.response_text,
        "skill_id": result.skill.id,
        "domain": result.skill.domain,
    })


@mcp.tool()
def skill_feedback(skill_id: str, success: bool) -> dict[str, str]:
    """
    Provide feedback on a skill execution.

    Reinforces successful skills (making them more likely to be used)
    and weakens failed skills (reducing their confidence).

    Args:
        skill_id: The skill_id from a match_skill or learn_skill result
        success: Whether the skill execution was successful

    Returns:
        Confirmation message.
    """
    cb = _get_cerebellum()
    if success:
        cb.skill_store.reinforce(skill_id)
    else:
        cb.skill_store.weaken(skill_id)
    return {"status": "ok", "message": f"Skill {'reinforced' if success else 'weakened'}"}


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
    skill_stats = stats.get("skill_store", {})
    lines = [
        "Digital Cerebellum Status",
        f"  Step: {stats['step']}",
        f"  Microzones: {', '.join(stats['microzones'])}",
        f"  Routes: {stats['routes']}",
        f"  Skills learned: {skill_stats.get('total', 0)}",
        f"  Avg skill confidence: {skill_stats.get('avg_confidence', 0):.3f}",
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
