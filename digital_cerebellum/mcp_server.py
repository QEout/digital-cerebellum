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
from digital_cerebellum.monitor import StepMonitor

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
_monitor: StepMonitor | None = None


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
        _cb.skill_store.load()
        log.info("Digital Cerebellum initialized with microzones: %s, skills: %d",
                 list(_cb._microzones.keys()), len(_cb.skill_store))
    return _cb


def _get_monitor() -> StepMonitor:
    """Lazy-init the step monitor singleton.

    Can work standalone (no cerebellum needed) for faster startup,
    or shares the cerebellum encoder when available.
    """
    global _monitor
    if _monitor is None:
        if _cb is not None:
            _monitor = StepMonitor(cerebellum=_cb)
        else:
            _monitor = StepMonitor()
        log.info("StepMonitor initialized (standalone=%s)", _cb is None)
    return _monitor


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
    tool_calls: list[dict[str, Any]] | None = None,
    domain: str = "response",
) -> dict[str, str]:
    """
    Teach the cerebellum a new skill from an interaction.

    After the LLM produces a response, store it as a learnable skill.
    Next time a similar query arrives, the cerebellum can respond directly
    without calling the LLM — replaying the stored response or tool-call
    sequence.

    Args:
        input_text: The query/input that triggered the response
        response_text: The response that was produced
        tool_calls: Optional list of tool-call dicts for multi-step action
            sequences (e.g., [{"tool": "click", "params": {"x": 100}}]).
            When provided, the skill stores the full action sequence for replay.
        domain: Skill domain (default: "response")

    Returns:
        The skill_id for future reference.
    """
    cb = _get_cerebellum()
    skill_id = cb.learn_skill(input_text, response_text,
                              tool_calls=tool_calls, domain=domain)
    cb.skill_store.save()
    return {"status": "ok", "skill_id": skill_id, "is_sequence": bool(tool_calls)}


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
    out = {
        "matched": True,
        "similarity": result.similarity,
        "match_confidence": result.match_confidence,
        "should_execute": result.should_execute,
        "response_text": result.skill.response_text,
        "skill_id": result.skill.id,
        "domain": result.skill.domain,
        "is_sequence": result.skill.is_sequence,
    }
    if result.skill.tool_calls:
        out["tool_calls"] = result.skill.tool_calls
    return _sanitize(out)


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
# Step Monitor tools (Phase 7 — universal agent monitoring)
# ======================================================================

@mcp.tool()
def monitor_before_step(
    action: str,
    state: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Call BEFORE the agent executes an action.

    The cerebellum predicts the expected outcome and checks for known
    failure patterns.  Returns a risk assessment and recommendation.

    This is the universal monitoring protocol: any agent framework
    can call this before each step to get predictive error interception.

    Args:
        action: Description of what the agent intends to do
            (e.g., "click the save button", "run git push", "move north")
        state: Description of the current state (optional)
            (e.g., "file editor open with unsaved changes", "health=50 position=(3,4)")
        context: Additional context (optional)

    Returns:
        Prediction with should_proceed recommendation, risk score,
        confidence, and any failure warnings from past experience.
    """
    monitor = _get_monitor()
    pred = monitor.before_step(action=action, state=state or None, context=context)
    result = {
        "should_proceed": pred.should_proceed,
        "confidence": round(pred.confidence, 4),
        "risk_score": round(pred.risk_score, 4),
        "cascade_risk": round(pred.cascade_risk, 4),
        "step_number": pred.step_number,
    }
    if pred.failure_warning is not None:
        result["failure_warning"] = {
            "pattern": pred.failure_warning.pattern_description,
            "similarity": round(pred.failure_warning.similarity, 4),
            "severity": round(pred.failure_warning.severity, 4),
        }

    if _cb is not None:
        try:
            query_emb = _cb.encoder.encode_text(action)
            memories = _cb.memory.retrieve(query_emb, top_k=3, min_strength=0.2)
            if memories:
                result["relevant_memories"] = [
                    {"content": m.content, "strength": round(m.strength, 3)}
                    for m in memories
                ]
        except Exception:
            pass

    return result


@mcp.tool()
def monitor_after_step(
    outcome: str,
    success: bool | None = None,
) -> dict[str, Any]:
    """
    Call AFTER the agent executes an action.

    The cerebellum compares the actual outcome against its prediction,
    detects error cascades, and learns from the experience.

    Args:
        outcome: Description of what actually happened
            (e.g., "save dialog appeared", "error: permission denied")
        success: Whether the step succeeded (optional).
            If not provided, the cerebellum infers from prediction error.

    Returns:
        Verdict with should_pause recommendation, SPE magnitude,
        cascade risk, and suggested next action.
    """
    monitor = _get_monitor()
    verdict = monitor.after_step(outcome=outcome, success=success)

    if _cb is not None:
        try:
            import uuid
            from digital_cerebellum.core.types import MemorySlot
            content = f"Step {verdict.step_number}: {outcome[:200]}"
            if verdict.should_pause:
                content += " [CASCADE DETECTED]"
            emb = _cb.encoder.encode_text(outcome)
            _cb.memory.store(MemorySlot(
                id=uuid.uuid4().hex,
                content=content,
                embedding=emb,
                strength=1.0 if verdict.should_pause else 0.6,
                layer="short_term",
                metadata={"spe": verdict.spe, "step": verdict.step_number},
            ))
        except Exception:
            pass

    return _sanitize({
        "spe": round(verdict.spe, 4),
        "should_pause": verdict.should_pause,
        "cascade_risk": round(verdict.cascade_risk, 4),
        "consecutive_errors": verdict.consecutive_errors,
        "suggestion": verdict.suggestion,
        "step_number": verdict.step_number,
        "details": verdict.details,
    })


@mcp.tool()
def monitor_reset() -> dict[str, Any]:
    """
    Reset the step monitor for a new task/episode.

    Clears step history and cascade state, but keeps learned
    knowledge (forward model weights and failure memory persist).

    Returns:
        Summary of the completed episode.
    """
    monitor = _get_monitor()
    summary = monitor.reset()
    return _sanitize(summary)


@mcp.tool()
def monitor_rollback_plan() -> dict[str, Any]:
    """
    Get the auto-rollback plan after a cascade is detected.

    When monitor_after_step returns should_pause=true, a rollback
    plan is automatically computed.  This tool returns it with:
    - Which step to roll back to
    - The last safe state description
    - List of failed steps to undo
    - A human-readable recommendation

    Returns:
        The rollback plan, or {"available": false} if no cascade
        has been detected in this episode.
    """
    monitor = _get_monitor()
    plan = monitor.get_rollback_plan()
    if plan is None:
        return {"available": False}
    return _sanitize({
        "available": True,
        "rollback_to_step": plan.rollback_to_step,
        "last_safe_state": plan.last_safe_state,
        "last_safe_outcome": plan.last_safe_outcome,
        "steps_wasted": plan.steps_wasted,
        "total_steps": plan.total_steps,
        "cascade_risk": round(plan.cascade_risk, 4),
        "failed_steps": plan.failed_steps,
        "recommendation": plan.recommendation,
    })


@mcp.tool()
def monitor_status() -> dict[str, Any]:
    """
    Get current step monitor status.

    Returns forward model stats, cascade detector state,
    failure memory count, and current episode summary.
    """
    monitor = _get_monitor()
    return _sanitize(monitor.stats)


# ======================================================================
# Fluid Memory tools (hippocampal memory analogue)
# ======================================================================

@mcp.tool()
def store_memory(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store an experience in the cerebellum's fluid memory.

    Memories are stored with embeddings for semantic retrieval, decay over
    time (short-term → long-term promotion), and are reconsolidated on
    every recall — just like biological episodic memory.

    Args:
        content: Text description of the experience to remember
            (e.g., "User prefers dark mode", "API call to /users failed with 403")
        metadata: Optional metadata dict (e.g., {"domain": "preferences", "step": 5})

    Returns:
        Memory slot ID and layer (short_term initially).
    """
    import uuid
    from digital_cerebellum.core.types import MemorySlot

    cb = _get_cerebellum()
    emb = cb.encoder.encode_text(content)
    slot = MemorySlot(
        id=uuid.uuid4().hex,
        content=content,
        embedding=emb,
        strength=1.0,
        layer="short_term",
        metadata=metadata or {},
    )
    slot_id = cb.memory.store(slot)
    return {"status": "ok", "memory_id": slot_id, "layer": "short_term"}


@mcp.tool()
def retrieve_memories(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant memories by semantic similarity.

    The cerebellum's fluid memory works like human episodic recall:
    retrieved memories are reconsolidated (strengthened and slightly
    reshaped toward the query), making frequently accessed memories
    more durable — just like biological reconsolidation.

    Args:
        query: What to remember — natural language description
        top_k: Maximum number of memories to return (default: 5)

    Returns:
        List of matching memories with content, strength, layer, and metadata.
    """
    cb = _get_cerebellum()
    query_emb = cb.encoder.encode_text(query)
    slots = cb.memory.retrieve(query_emb, top_k=top_k)
    return [
        {
            "memory_id": s.id,
            "content": s.content,
            "strength": round(s.strength, 4),
            "layer": s.layer,
            "access_count": s.access_count,
            "metadata": s.metadata,
        }
        for s in slots
    ]


# ======================================================================
# Sleep Cycle (offline memory consolidation)
# ======================================================================

@mcp.tool()
def run_sleep_cycle() -> dict[str, Any]:
    """
    Run one sleep cycle — offline memory consolidation.

    Like biological sleep, this performs maintenance:
    - Decays weak memories (forgetting curve)
    - Promotes strong short-term memories to long-term
    - Merges similar skills (deduplication)
    - Prunes unreliable skills (extinction)

    Call this periodically (e.g., between task sessions) or when the
    agent is idle. Frequent sleep cycles keep memory lean and skills sharp.

    Returns:
        Consolidation report with counts of promoted, pruned, and merged items.
    """
    cb = _get_cerebellum()
    report = cb.sleep()
    cb.skill_store.save()
    return _sanitize({
        "status": "ok",
        "memory_consolidated": {
            "promoted": report.promoted,
            "decayed": report.decayed,
        },
        "skill_consolidation": report.skill_consolidation,
        "memory_stats": cb.memory.stats,
        "skill_count": len(cb.skill_store),
    })


# ======================================================================
# Gut Feeling (somatic marker — intuition)
# ======================================================================

@mcp.tool()
def get_gut_feeling(
    action: str,
    domain: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Ask the cerebellum for its gut feeling about an action.

    Based on Damasio's Somatic Marker Hypothesis: past experiences leave
    emotional traces (markers) that bias future decisions BEFORE conscious
    reasoning. The cerebellum's prediction heads produce divergence patterns;
    when a current pattern resembles a past bad outcome, the gut feeling
    turns negative — even if surface-level confidence looks fine.

    Use this to get a fast, pre-rational risk assessment:
    - "positive" = past similar situations went well
    - "uneasy" = mixed signals
    - "alarm" = past similar situations went badly — reconsider!

    Args:
        action: The action being considered (e.g., "delete the production database")
        domain: Optional domain for more specific gut feeling (e.g., "shell_command")
        context: Optional additional context

    Returns:
        Gut feeling with valence (-1 to +1), intensity (0 to 1), label,
        and whether the feeling is strong enough to override normal routing.
    """
    cb = _get_cerebellum()
    input_data = {"action": action, "context": context} if context else {"action": action}
    feature_vec = cb.encoder.encode_text(action)

    result = cb.evaluate(domain or "tool_call", input_data, context=context)
    gut = result.get("_gut_feeling", {})

    if not gut:
        return {
            "valence": 0.0,
            "intensity": 0.0,
            "label": "neutral",
            "should_override": False,
            "message": "No somatic markers accumulated yet — need more experience.",
        }

    return _sanitize({
        "valence": gut.get("valence", 0.0),
        "intensity": gut.get("intensity", 0.0),
        "label": gut.get("label", "neutral"),
        "should_override": gut.get("should_override", False),
        "trigger_pattern": gut.get("trigger_pattern", "none"),
        "message": _gut_message(gut),
    })


def _gut_message(gut: dict) -> str:
    label = gut.get("label", "neutral")
    intensity = gut.get("intensity", 0)
    if label == "alarm":
        return f"Strong negative gut feeling (intensity={intensity:.2f}). Past similar actions led to bad outcomes. Reconsider."
    if label == "uneasy":
        return f"Mild unease (intensity={intensity:.2f}). Proceed with caution."
    if label == "positive":
        return f"Positive gut feeling (intensity={intensity:.2f}). Past similar actions went well."
    return "No strong feeling either way."


# ======================================================================
# Exploration Suggestions (curiosity-driven)
# ======================================================================

@mcp.tool()
def get_exploration_suggestions() -> list[dict[str, Any]]:
    """
    Get actionable exploration suggestions from the curiosity drive.

    The cerebellum tracks where it's learning fastest and recommends
    concrete actions: "explore domain X" (actively learning),
    "investigate domain Y" (high error, needs data), or
    "abandon domain Z" (noise, not learnable).

    Like biological curiosity: dopaminergic signals don't just report
    novelty — they DRIVE exploratory behavior toward learnable frontiers.

    Returns:
        List of exploration suggestions with domain, action, urgency, and reason.
        Empty list if not enough data has been accumulated yet.
    """
    cb = _get_cerebellum()
    requests = cb.curiosity_drive.get_exploration_requests()
    return _sanitize(requests)


# ======================================================================
# Habit Learning & Rhythm (predictive awakening)
# ======================================================================

@mcp.tool()
def get_habit_patterns() -> dict[str, Any]:
    """
    Extract and return learned behaviour patterns from past interactions.

    The cerebellum observes every action the agent takes (timestamps,
    domains, success/failure) and extracts temporal patterns:

    - **Daily rhythms**: "user checks email at ~09:00 on weekdays"
    - **Sequential patterns**: "after email, user opens Slack within 10min"
    - **Frequency patterns**: "user deploys code ~2x per week"

    Call this during sleep cycles or when you want to understand user habits.

    Returns:
        Dict with 'patterns' (list of extracted patterns) and 'stats'.
    """
    monitor = _get_monitor()
    patterns = monitor.habit_observer.extract_patterns()
    return _sanitize({
        "patterns": [
            {
                "id": p.id,
                "action": p.action_pattern,
                "domain": p.domain,
                "type": p.pattern_type,
                "hour": round(p.hour_mean, 1) if p.pattern_type == "daily" else None,
                "hour_std": round(p.hour_std, 1) if p.pattern_type == "daily" else None,
                "weekdays": p.weekdays if p.pattern_type == "daily" else None,
                "predecessor": p.predecessor if p.pattern_type == "sequential" else None,
                "delay_seconds": round(p.delay_mean_seconds, 0) if p.pattern_type == "sequential" else None,
                "confidence": round(p.confidence, 3),
                "observations": p.sample_count,
            }
            for p in patterns
        ],
        "stats": monitor.habit_observer.stats,
    })


@mcp.tool()
def get_proactive_suggestions() -> list[dict[str, Any]]:
    """
    Get proactive suggestions based on learned user habits and current time.

    Instead of waiting for the user to ask, the cerebellum predicts what
    the user is likely to do next based on temporal patterns:

    - "You usually check email around now (09:00 ± 10min, 8 observations)"
    - "After checking email, you typically open Slack within 12 minutes"

    This is the RHYTHM system — event-driven + predictive, not polling.

    Returns:
        List of suggestions with action, confidence, reason, and hint.
        Empty if no strong predictions match the current time.
    """
    monitor = _get_monitor()
    from digital_cerebellum.rhythm.engine import RhythmEngine
    rhythm = RhythmEngine(monitor.habit_observer)
    return _sanitize(rhythm.get_proactive_suggestions())


@mcp.tool()
def get_rhythm_status() -> dict[str, Any]:
    """
    Get the current rhythm engine status.

    Shows activity level (active/normal/quiet), adaptive check interval,
    and number of learned patterns. Useful for understanding how the
    cerebellum is adapting its attention to the user's work patterns.

    Returns:
        Dict with mode, check_interval, activity_level, patterns_known.
    """
    monitor = _get_monitor()
    from digital_cerebellum.rhythm.engine import RhythmEngine
    rhythm = RhythmEngine(monitor.habit_observer)
    return _sanitize(rhythm.state)


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
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="HTTP bind address (default: 0.0.0.0)")
    parser.add_argument("--json-response", action="store_true",
                        help="Return JSON instead of SSE (stateless mode)")
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        if args.json_response:
            mcp.settings.json_response = True
            mcp.settings.stateless_http = True
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
