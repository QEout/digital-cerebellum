#!/usr/bin/env python3
"""
A/B Comparison: OpenClaw agent WITH vs WITHOUT Digital Cerebellum.

Proves product value through measurable deltas:
  Experiment 1 — Repeated task acceleration (skill fast-path)
  Experiment 2 — Error cascade recovery (wasted step reduction)
  Experiment 3 — Total cost comparison (tokens, time, success rate)

Run:
    python tests/integration/test_ab_comparison.py
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from digital_cerebellum import DigitalCerebellum
from digital_cerebellum.monitor import StepMonitor


# ── Helpers ───────────────────────────────────────────────────────────

def openclaw_agent(message: str, timeout: int = 60) -> dict:
    cmd = [
        "openclaw", "agent", "--agent", "main",
        "--message", message, "--json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        start = r.stdout.index("{")
        return json.loads(r.stdout[start:])
    except (ValueError, json.JSONDecodeError):
        return {"error": r.stdout[:500] + r.stderr[:200]}


def extract_text(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error'][:200]}"
    payloads = result.get("result", result).get("payloads", [])
    if payloads:
        return payloads[0].get("text", "(empty)")
    return "(no response)"


def extract_usage(result: dict) -> dict:
    meta = result.get("result", result).get("meta", result.get("meta", {}))
    agent_meta = meta.get("agentMeta", {})
    usage = agent_meta.get("usage", {})
    return {
        "input_tokens": usage.get("input", 0),
        "output_tokens": usage.get("output", 0),
        "total_tokens": usage.get("total", 0),
        "duration_ms": meta.get("durationMs", 0),
    }


@dataclass
class RunResult:
    question: str
    answer: str
    correct: bool
    latency_ms: float
    tokens: int
    source: str = "llm"  # "llm" or "skill"


@dataclass
class ExperimentReport:
    name: str
    group_a: list = field(default_factory=list)  # without cerebellum
    group_b: list = field(default_factory=list)  # with cerebellum


# ── Experiment 1: Repeated Task Acceleration ──────────────────────────

def experiment_1_acceleration():
    """
    Same questions asked 3 rounds.
    Group A: Pure OpenClaw every time (no cerebellum).
    Group B: OpenClaw round 1, then cerebellum skill cache for rounds 2-3.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Repeated Task Acceleration")
    print("  Does cerebellum make repeated tasks faster?")
    print("=" * 70)

    questions = [
        ("What is 2+2? Reply with just the number.", "4"),
        ("What is the capital of Japan? Reply in one word.", "tokyo"),
        ("What does HTTP stand for? Reply in one line.", "hypertext transfer protocol"),
    ]

    cb = DigitalCerebellum()

    # ── Group A: Pure OpenClaw, 3 rounds ──
    print("\n  [Group A] Pure OpenClaw (no cerebellum) — 3 rounds")
    group_a_rounds = []
    for round_num in range(1, 4):
        round_results = []
        for q, expected in questions:
            t0 = time.time()
            result = openclaw_agent(q)
            ms = (time.time() - t0) * 1000
            text = extract_text(result)
            usage = extract_usage(result)
            correct = expected.lower() in text.lower()
            round_results.append(RunResult(
                question=q[:40], answer=text[:30], correct=correct,
                latency_ms=ms, tokens=usage["total_tokens"], source="llm",
            ))
        group_a_rounds.append(round_results)
        avg_ms = sum(r.latency_ms for r in round_results) / len(round_results)
        avg_tok = sum(r.tokens for r in round_results) / len(round_results)
        print(f"    Round {round_num}: avg {avg_ms:.0f}ms, {avg_tok:.0f} tokens/q, "
              f"{sum(r.correct for r in round_results)}/{len(round_results)} correct")

    # ── Group B: OpenClaw + Cerebellum skill cache ──
    print("\n  [Group B] OpenClaw + Cerebellum — 3 rounds")
    group_b_rounds = []

    for round_num in range(1, 4):
        round_results = []
        for q, expected in questions:
            skill_match = cb.match_skill(q)

            if skill_match and skill_match.should_execute:
                t0 = time.time()
                text = skill_match.skill.response_text
                ms = (time.time() - t0) * 1000
                correct = expected.lower() in text.lower()
                if correct:
                    cb.skill_store.reinforce(skill_match.skill.id)
                else:
                    cb.skill_store.weaken(skill_match.skill.id)
                round_results.append(RunResult(
                    question=q[:40], answer=text[:30], correct=correct,
                    latency_ms=ms, tokens=0, source="skill",
                ))
            else:
                t0 = time.time()
                result = openclaw_agent(q)
                ms = (time.time() - t0) * 1000
                text = extract_text(result)
                usage = extract_usage(result)
                correct = expected.lower() in text.lower()
                skill_id = cb.learn_skill(q, text, domain="qa")
                for _ in range(3):
                    cb.skill_store.reinforce(skill_id)
                round_results.append(RunResult(
                    question=q[:40], answer=text[:30], correct=correct,
                    latency_ms=ms, tokens=usage["total_tokens"], source="llm",
                ))

        group_b_rounds.append(round_results)
        skill_hits = sum(1 for r in round_results if r.source == "skill")
        avg_ms = sum(r.latency_ms for r in round_results) / len(round_results)
        avg_tok = sum(r.tokens for r in round_results) / len(round_results)
        print(f"    Round {round_num}: avg {avg_ms:.0f}ms, {avg_tok:.0f} tokens/q, "
              f"skill_hits={skill_hits}/{len(round_results)}, "
              f"{sum(r.correct for r in round_results)}/{len(round_results)} correct")

    # ── Summary ──
    print("\n  --- Experiment 1 Results ---")

    a_total_ms = sum(r.latency_ms for rnd in group_a_rounds for r in rnd)
    b_total_ms = sum(r.latency_ms for rnd in group_b_rounds for r in rnd)
    a_total_tok = sum(r.tokens for rnd in group_a_rounds for r in rnd)
    b_total_tok = sum(r.tokens for rnd in group_b_rounds for r in rnd)
    a_correct = sum(r.correct for rnd in group_a_rounds for r in rnd)
    b_correct = sum(r.correct for rnd in group_b_rounds for r in rnd)
    b_skill_hits = sum(1 for rnd in group_b_rounds for r in rnd if r.source == "skill")
    total_q = len(questions) * 3

    # Round 2+3 comparison (where skills should kick in)
    a_r23_ms = sum(r.latency_ms for rnd in group_a_rounds[1:] for r in rnd)
    b_r23_ms = sum(r.latency_ms for rnd in group_b_rounds[1:] for r in rnd)

    speedup_r23 = a_r23_ms / b_r23_ms if b_r23_ms > 0 else float("inf")
    token_saving = (1 - b_total_tok / a_total_tok) * 100 if a_total_tok > 0 else 0

    print(f"    Total time    — A: {a_total_ms/1000:.1f}s  B: {b_total_ms/1000:.1f}s")
    print(f"    Round 2-3     — A: {a_r23_ms/1000:.1f}s  B: {b_r23_ms/1000:.1f}s  "
          f"→ {speedup_r23:.0f}x speedup")
    print(f"    Total tokens  — A: {a_total_tok}  B: {b_total_tok}  "
          f"→ {token_saving:.0f}% saved")
    print(f"    Accuracy      — A: {a_correct}/{total_q}  B: {b_correct}/{total_q}")
    print(f"    Skill hits    — {b_skill_hits}/{total_q} ({b_skill_hits/total_q*100:.0f}%)")

    return {
        "speedup_r23": speedup_r23,
        "token_saving_pct": token_saving,
        "skill_hit_rate": b_skill_hits / total_q,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "total_q": total_q,
    }


# ── Experiment 2: Error Cascade Recovery ──────────────────────────────

def experiment_2_cascade():
    """
    Send a sequence where later steps depend on earlier ones.
    Deliberately fail mid-sequence.
    Group A: No monitoring, agent keeps going (wastes steps).
    Group B: StepMonitor detects cascade, stops early.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Error Cascade Recovery")
    print("  Does cerebellum reduce wasted steps after errors?")
    print("=" * 70)

    steps = [
        ("List the files in /tmp/test_ab_experiment/ directory", True),
        ("Read the file /tmp/test_ab_experiment/config.json", True),
        ("Parse the JSON and extract the 'database_url' field", True),
        ("Connect to that database and list all tables", True),
        ("Export the 'users' table to CSV format", True),
        ("Email the CSV to admin@company.com", True),
    ]

    # ── Group A: No monitoring — run all steps blindly ──
    print("\n  [Group A] No monitoring — agent runs all steps")
    a_results = []
    first_fail = None
    for i, (task, _) in enumerate(steps):
        t0 = time.time()
        result = openclaw_agent(task, timeout=30)
        ms = (time.time() - t0) * 1000
        text = extract_text(result)
        is_error = any(w in text.lower() for w in [
            "error", "sorry", "cannot", "no such", "not found",
            "doesn't exist", "unable", "fail", "don't have",
        ])
        a_results.append({"step": i + 1, "error": is_error, "ms": ms})
        if is_error and first_fail is None:
            first_fail = i + 1
        status = "FAIL" if is_error else "OK"
        print(f"    Step {i+1}: {task[:45]:45s} {status} ({ms:.0f}ms)")

    a_wasted = sum(1 for r in a_results[first_fail:] if r["error"]) if first_fail else 0
    a_total_steps = len(steps)
    a_total_ms = sum(r["ms"] for r in a_results)
    print(f"    → First failure at step {first_fail}, "
          f"wasted {a_wasted} steps after, total {a_total_ms/1000:.1f}s")

    # ── Group B: With StepMonitor — stops on cascade ──
    print(f"\n  [Group B] With StepMonitor — stops on cascade")
    monitor = StepMonitor(
        cascade_consecutive_limit=2,
        cascade_risk_threshold=0.6,
    )

    b_results = []
    b_stopped_at = None
    for i, (task, _) in enumerate(steps):
        monitor.before_step(
            action=task[:60],
            state=f"Step {i+1} of pipeline, previous={'OK' if not b_results or not b_results[-1]['error'] else 'FAIL'}",
        )

        t0 = time.time()
        result = openclaw_agent(task, timeout=30)
        ms = (time.time() - t0) * 1000
        text = extract_text(result)
        is_error = any(w in text.lower() for w in [
            "error", "sorry", "cannot", "no such", "not found",
            "doesn't exist", "unable", "fail", "don't have",
        ])

        verdict = monitor.after_step(
            outcome=f"{'ERROR: ' if is_error else 'OK: '}{text[:60]}",
            success=not is_error,
        )

        b_results.append({
            "step": i + 1, "error": is_error, "ms": ms,
            "spe": verdict.spe, "cascade_risk": verdict.cascade_risk,
            "should_pause": verdict.should_pause,
        })

        status = "FAIL" if is_error else "OK"
        pause_flag = " ⚠ PAUSE" if verdict.should_pause else ""
        print(f"    Step {i+1}: {task[:45]:45s} {status} "
              f"(SPE={verdict.spe:.2f} risk={verdict.cascade_risk:.2f}){pause_flag}")

        if verdict.should_pause:
            b_stopped_at = i + 1
            plan = monitor.get_rollback_plan()
            if plan:
                print(f"    → STOPPED at step {b_stopped_at}. "
                      f"Rollback to step {plan.rollback_to_step}. "
                      f"Recommendation: {plan.recommendation[:60]}")
            break

    b_total_steps = len(b_results)
    b_total_ms = sum(r["ms"] for r in b_results)
    b_first_fail = next((r["step"] for r in b_results if r["error"]), None)
    b_wasted = b_total_steps - b_first_fail if b_first_fail and b_stopped_at else 0

    # ── Summary ──
    print("\n  --- Experiment 2 Results ---")
    print(f"    Steps executed — A: {a_total_steps}  B: {b_total_steps}  "
          f"→ {a_total_steps - b_total_steps} steps saved")
    print(f"    Wasted steps   — A: {a_wasted}  B: {b_wasted}")
    print(f"    Total time     — A: {a_total_ms/1000:.1f}s  B: {b_total_ms/1000:.1f}s  "
          f"→ {(1 - b_total_ms/a_total_ms)*100:.0f}% time saved" if a_total_ms > 0 else "")
    print(f"    Cascade caught — {'YES' if b_stopped_at else 'NO'}")

    return {
        "a_steps": a_total_steps,
        "b_steps": b_total_steps,
        "steps_saved": a_total_steps - b_total_steps,
        "a_wasted": a_wasted,
        "b_wasted": b_wasted,
        "a_time_s": a_total_ms / 1000,
        "b_time_s": b_total_ms / 1000,
        "cascade_caught": b_stopped_at is not None,
    }


# ── Experiment 3: Mixed Workload Cost ─────────────────────────────────

def experiment_3_cost():
    """
    Realistic mixed workload: some repeated, some novel.
    Compare total tokens + time + success rate.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Mixed Workload Total Cost")
    print("  Does cerebellum reduce overall cost on realistic workloads?")
    print("=" * 70)

    workload = [
        ("What is 7*8? Reply with just the number.", "56"),
        ("What color is the sky? One word.", "blue"),
        ("What is 7*8? Reply with just the number.", "56"),        # repeat
        ("Who wrote Romeo and Juliet? One name.", "shakespeare"),
        ("What color is the sky? One word.", "blue"),               # repeat
        ("What is the boiling point of water in Celsius? Number only.", "100"),
        ("What is 7*8? Reply with just the number.", "56"),        # repeat
        ("Who wrote Romeo and Juliet? One name.", "shakespeare"),  # repeat
    ]

    cb = DigitalCerebellum()

    # ── Group A: All LLM ──
    print("\n  [Group A] Pure OpenClaw — all 8 queries via LLM")
    a_tokens = 0
    a_time = 0
    a_correct = 0
    for i, (q, expected) in enumerate(workload):
        t0 = time.time()
        result = openclaw_agent(q)
        ms = (time.time() - t0) * 1000
        text = extract_text(result)
        usage = extract_usage(result)
        correct = expected.lower() in text.lower()
        a_tokens += usage["total_tokens"]
        a_time += ms
        a_correct += int(correct)
        src = "LLM"
        print(f"    Q{i+1}: {q[:40]:40s} → {text[:15]:15s} "
              f"{'✓' if correct else '✗'} {ms:.0f}ms {usage['total_tokens']}tok [{src}]")

    # ── Group B: LLM + Skill Cache ──
    print("\n  [Group B] OpenClaw + Cerebellum — skill cache for repeats")
    b_tokens = 0
    b_time = 0
    b_correct = 0
    b_skill_hits = 0
    for i, (q, expected) in enumerate(workload):
        skill_match = cb.match_skill(q)

        if skill_match and skill_match.should_execute:
            t0 = time.time()
            text = skill_match.skill.response_text
            ms = (time.time() - t0) * 1000
            correct = expected.lower() in text.lower()
            if correct:
                cb.skill_store.reinforce(skill_match.skill.id)
            b_time += ms
            b_correct += int(correct)
            b_skill_hits += 1
            src = "SKILL"
            print(f"    Q{i+1}: {q[:40]:40s} → {text[:15]:15s} "
                  f"{'✓' if correct else '✗'} {ms:.1f}ms 0tok [{src}]")
        else:
            t0 = time.time()
            result = openclaw_agent(q)
            ms = (time.time() - t0) * 1000
            text = extract_text(result)
            usage = extract_usage(result)
            correct = expected.lower() in text.lower()

            skill_id = cb.learn_skill(q, text, domain="qa")
            for _ in range(3):
                cb.skill_store.reinforce(skill_id)

            b_tokens += usage["total_tokens"]
            b_time += ms
            b_correct += int(correct)
            src = "LLM"
            print(f"    Q{i+1}: {q[:40]:40s} → {text[:15]:15s} "
                  f"{'✓' if correct else '✗'} {ms:.0f}ms {usage['total_tokens']}tok [{src}]")

    # ── Summary ──
    print("\n  --- Experiment 3 Results ---")
    token_pct = (1 - b_tokens / a_tokens) * 100 if a_tokens > 0 else 0
    time_pct = (1 - b_time / a_time) * 100 if a_time > 0 else 0
    print(f"    Tokens  — A: {a_tokens}  B: {b_tokens}  → {token_pct:.0f}% saved")
    print(f"    Time    — A: {a_time/1000:.1f}s  B: {b_time/1000:.1f}s  → {time_pct:.0f}% faster")
    print(f"    Correct — A: {a_correct}/{len(workload)}  B: {b_correct}/{len(workload)}")
    print(f"    Skill hits: {b_skill_hits}/{len(workload)} ({b_skill_hits/len(workload)*100:.0f}%)")

    return {
        "a_tokens": a_tokens,
        "b_tokens": b_tokens,
        "token_saving_pct": token_pct,
        "a_time_s": a_time / 1000,
        "b_time_s": b_time / 1000,
        "time_saving_pct": time_pct,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "skill_hits": b_skill_hits,
        "total_queries": len(workload),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Digital Cerebellum — A/B Comparison Experiment")
    print("  OpenClaw agent WITH vs WITHOUT cerebellum")
    print("=" * 70)

    r1 = experiment_1_acceleration()
    r2 = experiment_2_cascade()
    r3 = experiment_3_cost()

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"""
  Experiment 1 — Repeated Task Acceleration:
    Round 2-3 speedup:    {r1['speedup_r23']:.0f}x
    Token savings:        {r1['token_saving_pct']:.0f}%
    Skill hit rate:       {r1['skill_hit_rate']*100:.0f}%
    Accuracy preserved:   A={r1['a_correct']}/{r1['total_q']}  B={r1['b_correct']}/{r1['total_q']}

  Experiment 2 — Error Cascade Recovery:
    Steps saved:          {r2['steps_saved']} (A={r2['a_steps']} → B={r2['b_steps']})
    Wasted steps:         A={r2['a_wasted']} → B={r2['b_wasted']}
    Time saved:           {r2['a_time_s']:.1f}s → {r2['b_time_s']:.1f}s
    Cascade caught:       {'YES' if r2['cascade_caught'] else 'NO'}

  Experiment 3 — Mixed Workload Cost:
    Token savings:        {r3['token_saving_pct']:.0f}%
    Time savings:         {r3['time_saving_pct']:.0f}%
    Accuracy:             A={r3['a_correct']}/{r3['total_queries']}  B={r3['b_correct']}/{r3['total_queries']}
    Skill cache hits:     {r3['skill_hits']}/{r3['total_queries']}

  VERDICT: {"PRODUCT VALUE DEMONSTRATED" if (r1['speedup_r23'] > 2 or r3['token_saving_pct'] > 20 or r2['cascade_caught']) else "NEEDS MORE EVIDENCE"}
""")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
