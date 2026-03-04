#!/usr/bin/env python3
"""
Fast AND Reliable Demo — SkillStore + StepMonitor working together.

This demo shows the cerebellum's two engines working in concert:

  SkillStore  → makes repeated tasks instant (<10ms, no LLM)
  StepMonitor → catches errors before they cascade

Scenario: a customer support agent handles tickets.
  Round 1: Agent sees tickets for the first time → slow (learns)
  Round 2: Same ticket types → fast (SkillStore replays)
  Round 3: Something goes wrong mid-task → StepMonitor catches it

This is the "既快又稳" story in code.

Usage:
    python examples/fast_and_reliable_demo.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from digital_cerebellum.main import DigitalCerebellum
from digital_cerebellum.monitor import StepMonitor


# ======================================================================
# Simulated support ticket domain
# ======================================================================

@dataclass
class Ticket:
    id: str
    category: str
    customer_query: str
    resolution: str
    steps: list[tuple[str, str, str]]  # (action, state, expected_outcome)


TICKET_TEMPLATES = [
    Ticket(
        id="T-001", category="password_reset",
        customer_query="I forgot my password and can't log in",
        resolution="Password reset email sent to registered address",
        steps=[
            ("look up customer account", "ticket open", "account found: user@example.com"),
            ("verify identity via security question", "account found", "identity confirmed"),
            ("send password reset email", "identity verified", "reset email sent successfully"),
            ("update ticket status to resolved", "email sent", "ticket marked resolved"),
        ],
    ),
    Ticket(
        id="T-002", category="billing_dispute",
        customer_query="I was charged twice for my subscription",
        resolution="Duplicate charge refunded, $9.99 credited",
        steps=[
            ("pull billing history for customer", "ticket open", "billing records loaded, 2 charges found"),
            ("identify duplicate charge", "records loaded", "duplicate charge $9.99 on March 1st confirmed"),
            ("initiate refund for duplicate", "duplicate confirmed", "refund of $9.99 initiated"),
            ("send confirmation email to customer", "refund initiated", "confirmation email sent"),
            ("update ticket status to resolved", "email sent", "ticket marked resolved"),
        ],
    ),
    Ticket(
        id="T-003", category="feature_request",
        customer_query="Can you add dark mode to the app?",
        resolution="Feature request logged, added to product roadmap",
        steps=[
            ("check existing feature requests", "ticket open", "no existing dark mode request found"),
            ("create feature request in backlog", "no duplicate found", "feature request FR-847 created"),
            ("notify product team", "request created", "product team notified via Slack"),
            ("send acknowledgment to customer", "team notified", "acknowledgment email sent"),
            ("update ticket status to resolved", "email sent", "ticket marked resolved"),
        ],
    ),
]


def simulate_llm_response(ticket: Ticket) -> str:
    """Simulate what an LLM would respond (expensive, 500ms+)."""
    time.sleep(0.02)  # simulate LLM latency (scaled down for demo)
    return ticket.resolution


# ======================================================================
# Demo
# ======================================================================


def main():
    print("=" * 72)
    print("  Digital Cerebellum — Fast AND Reliable Demo")
    print("  SkillStore (speed) + StepMonitor (reliability) in concert")
    print("=" * 72)

    cb = DigitalCerebellum()
    monitor = StepMonitor()

    # ── Round 1: First encounter — learn everything ──
    print("\n📚 ROUND 1: First encounter (cold start, everything goes through LLM)")
    print("─" * 72)

    round1_times = []
    for ticket in TICKET_TEMPLATES:
        t0 = time.perf_counter()

        match = cb.match_skill(ticket.customer_query)
        if match is not None and match.should_execute:
            response = match.skill.response_text
            path = "cerebellum"
        else:
            response = simulate_llm_response(ticket)
            path = "cortex"
            cb.learn_skill(
                input_text=ticket.customer_query,
                response_text=response,
                domain=ticket.category,
            )

        for action, state, expected_outcome in ticket.steps:
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=expected_outcome, success=True)

        elapsed = (time.perf_counter() - t0) * 1000
        round1_times.append(elapsed)
        monitor.reset()

        print(f"  [{ticket.id}] {ticket.category:<20s} | path={path:<11s} | {elapsed:>7.1f}ms | {response[:50]}")

    for skill in cb.skill_store._skills.values():
        for _ in range(4):
            cb.skill_store.reinforce(skill.id)

    print(f"\n  Skills learned: {len(cb.skill_store._skills)}")
    print(f"  Avg latency:    {sum(round1_times)/len(round1_times):.1f}ms")

    # ── Round 2: Same ticket types — SkillStore should fire ──
    print(f"\n⚡ ROUND 2: Repeat tickets (SkillStore kicks in)")
    print("─" * 72)

    similar_queries = [
        ("I forgot my password, I can't log in to my account", "password_reset"),
        ("I was charged twice for my subscription this month", "billing_dispute"),
        ("Can you add a dark mode feature to the app?", "feature_request"),
    ]

    round2_times = []
    skill_hits = 0

    for query, expected_cat in similar_queries:
        t0 = time.perf_counter()

        match = cb.match_skill(query)
        if match is not None and match.should_execute:
            response = match.skill.response_text
            path = "cerebellum"
            skill_hits += 1
        else:
            response = f"[LLM fallback for: {query[:30]}...]"
            path = "cortex"
            time.sleep(0.02)

        elapsed = (time.perf_counter() - t0) * 1000
        round2_times.append(elapsed)

        print(f"  [{expected_cat:<20s}] path={path:<11s} | {elapsed:>7.1f}ms | {response[:50]}")

    r1_avg = sum(round1_times) / len(round1_times)
    r2_avg = sum(round2_times) / len(round2_times)
    speedup = r1_avg / r2_avg if r2_avg > 0 else float('inf')

    print(f"\n  Skill hits:  {skill_hits}/{len(similar_queries)}")
    print(f"  Round 1 avg: {r1_avg:.1f}ms (with LLM)")
    print(f"  Round 2 avg: {r2_avg:.1f}ms (SkillStore)")
    print(f"  Speedup:     {speedup:.0f}x faster")

    # ── Round 3: Error injection — StepMonitor catches cascade ──
    print(f"\n🛡️  ROUND 3: Error injection (StepMonitor catches cascade)")
    print("─" * 72)

    # Train a fresh monitor on billing dispute (focused training)
    monitor = StepMonitor(cascade_consecutive_limit=2)
    billing_ticket = TICKET_TEMPLATES[1]
    for _ in range(5):
        for action, state, outcome in billing_ticket.steps:
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=outcome, success=True)
        monitor.reset()

    print(f"  Monitor trained on {monitor.forward_model.stats['step']} successful steps")

    # Now run billing dispute that goes wrong at step 2
    error_steps = [
        ("pull billing history for customer", "ticket open",
         "billing records loaded, 2 charges found", True),
        ("identify duplicate charge", "records loaded",
         "error: billing system API timeout, no data returned", False),
        ("initiate refund for duplicate", "duplicate confirmed",
         "error: cannot refund, no duplicate charge identified", False),
        ("send confirmation email to customer", "refund initiated",
         "error: sent email saying refund processed when it wasn't", False),
        ("update ticket status to resolved", "email sent",
         "ticket marked resolved but refund never happened", False),
    ]

    stopped_at = None
    for i, (action, state, outcome, is_ok) in enumerate(error_steps, 1):
        pred = monitor.before_step(action=action, state=state)

        status = "✅" if pred.should_proceed else "🛑"
        warning = f" ⚠️ {pred.failure_warning.pattern_description[:40]}" if pred.failure_warning else ""

        print(f"  Step {i}: {status} {action:<45s} risk={pred.risk_score:.2f}{warning}")

        if not pred.should_proceed:
            stopped_at = i
            monitor.after_step(outcome=outcome)
            print(f"\n  🛑 StepMonitor blocked step {i} — cascade prevented!")
            break

        verdict = monitor.after_step(outcome=outcome, success=is_ok)
        spe_str = f"  spe={verdict.spe:.2f}"

        if verdict.should_pause:
            stopped_at = i
            print(f"       └─{spe_str}")
            print(f"\n  🛑 StepMonitor detected cascade after step {i}!")
            print(f"     SPE: {verdict.spe:.3f} | Consecutive errors: {verdict.consecutive_errors}")
            break
        else:
            print(f"       └─{spe_str}")

    # ── AutoRollback plan ──
    plan = monitor.get_rollback_plan()
    if plan is not None:
        print(f"\n  📋 AutoRollback Plan:")
        print(f"     Roll back to: step {plan.rollback_to_step}")
        print(f"     Last safe state: {plan.last_safe_state[:60]}")
        print(f"     Steps wasted: {plan.steps_wasted}")
        print(f"     Recommendation: {plan.recommendation[:80]}")

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY: The cerebellum in action")
    print(f"{'=' * 72}")
    print()
    print(f"  SPEED (SkillStore):")
    print(f"    Round 1 (cold):  {r1_avg:.1f}ms avg — every query calls LLM")
    print(f"    Round 2 (warm):  {r2_avg:.1f}ms avg — {skill_hits}/{len(similar_queries)} handled by SkillStore")
    print(f"    Speedup:         {speedup:.0f}x")
    print()
    print(f"  RELIABILITY (StepMonitor + AutoRollback):")
    print(f"    Error injected at step 2 of billing dispute")
    if stopped_at:
        saved = len(error_steps) - stopped_at
        print(f"    Detected at step {stopped_at} — prevented {saved} cascading error(s)")
    else:
        print(f"    Not caught (all steps executed)")
    if plan:
        print(f"    AutoRollback: revert to step {plan.rollback_to_step}, save {plan.steps_wasted} wasted steps")
    print()
    print(f"  COMBINED:")
    print(f"    Known patterns  → instant execution (<10ms, no LLM)")
    print(f"    Unknown/error   → detect, stop, and auto-rollback")
    print(f"    Same cerebellum → both capabilities from one architecture")
    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
