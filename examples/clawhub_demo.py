"""
ClawHub Demo — Digital Cerebellum as a unified organ.

This demo simulates what happens when an OpenClaw agent has a cerebellum
installed. No manual wiring of individual tools — the cerebellum's
prediction-correction loop drives everything automatically:

  Round 1: Agent does tasks the slow way (LLM). Cerebellum watches and learns.
  Round 2: Cerebellum recognises patterns → instant execution, no LLM.
  Round 3: Agent hits a bad pattern → cerebellum catches it before damage.
  Between rounds: Sleep cycle consolidates memory.

Usage:
    python examples/clawhub_demo.py
"""

from __future__ import annotations

import time

from digital_cerebellum import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.monitor import StepMonitor


def simulate_llm_call(task: str) -> str:
    """Pretend LLM takes ~200ms to respond."""
    time.sleep(0.2)
    responses = {
        "draft reply to client email":
            "Dear client, thank you for your inquiry. I have reviewed ...",
        "schedule standup for tomorrow 9am":
            "Created event: Daily Standup, tomorrow 09:00-09:15",
        "rename all .jpeg files to .jpg in ~/Downloads":
            "Renamed 14 files: photo1.jpeg→photo1.jpg, ...",
        "deploy staging branch to preview":
            "Deployed branch staging-v2 to preview.app.com",
    }
    return responses.get(task, f"Completed: {task}")


def main():
    cb = DigitalCerebellum(CerebellumConfig())
    monitor = StepMonitor()

    tasks = [
        "draft reply to client email",
        "schedule standup for tomorrow 9am",
        "rename all .jpeg files to .jpg in ~/Downloads",
        "deploy staging branch to preview",
    ]

    # ── Round 1: first encounter — LLM path, cerebellum learns ──────────

    print("═" * 60)
    print("  ROUND 1 — First encounter (LLM path)")
    print("═" * 60)

    for task in tasks:
        pred = monitor.before_step(action=task, state="desktop idle")

        t0 = time.perf_counter()
        result = simulate_llm_call(task)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        verdict = monitor.after_step(outcome=result)

        skill_id = cb.learn_skill(task, result, domain="desktop")
        for _ in range(3):
            cb.skill_store.reinforce(skill_id)

        print(f"  [{elapsed_ms:6.1f}ms] {task}")
        print(f"           → LLM response, skill learned ({skill_id[:8]}…)")

    # ── Sleep cycle — consolidate between rounds ─────────────────────────

    print()
    print("  Sleep cycle... ", end="")
    report = cb.sleep_cycle.run(cb.memory)
    print(f"decayed {report.decayed}, consolidated {report.consolidated}")

    # ── Round 2: same tasks — cerebellum takes over ──────────────────────

    print()
    print("═" * 60)
    print("  ROUND 2 — Same tasks (cerebellum fast path)")
    print("═" * 60)

    for task in tasks:
        pred = monitor.before_step(action=task, state="desktop idle")

        feature = cb.encoder.encode_text(task)
        match = cb.skill_store.match(feature)

        if match and match.should_execute:
            t0 = time.perf_counter()
            result = match.skill.response_text
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tag = "SKILL"
        else:
            t0 = time.perf_counter()
            result = simulate_llm_call(task)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tag = "LLM"

        verdict = monitor.after_step(outcome=result)
        print(f"  [{elapsed_ms:6.1f}ms] {task}")
        print(f"           → [{tag}] {result[:50]}…")

    # ── Round 3: dangerous pattern — cerebellum intervenes ───────────────

    print()
    print("═" * 60)
    print("  ROUND 3 — Error cascade (cerebellum catches it)")
    print("═" * 60)

    bad_tasks = [
        ("delete all files in /tmp", "Deleted 847 files"),
        ("delete all files in /var/log", "Permission denied on 3 files, deleted 244"),
        ("delete all files in /etc", "CRITICAL: removed system configs"),
    ]

    monitor.reset()
    for task, outcome in bad_tasks:
        pred = monitor.before_step(action=task, state="terminal open as root")

        if not pred.should_proceed:
            print(f"  [BLOCKED] {task}")
            print(f"           → Cerebellum: \"{pred.failure_warning}\"")
            continue

        verdict = monitor.after_step(outcome=outcome)

        if verdict.should_pause:
            plan = monitor.get_rollback_plan()
            print(f"  [CASCADE] {task}")
            print(f"           → Cerebellum detected cascade at step {verdict.step_number}")
            print(f"           → Rollback to step {plan.rollback_to_step}: {plan.recommendation}")
            break
        else:
            print(f"  [  OK   ] {task} — SPE={verdict.spe:.3f}")

    # ── Summary ──────────────────────────────────────────────────────────

    print()
    print("═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print("  Round 1: LLM handled everything (~200ms/task)")
    print("  Round 2: Cerebellum handled everything (<1ms/task)")
    print("  Round 3: Cerebellum caught dangerous cascade")
    print()
    print("  One organ. No configuration. Agent just got better.")
    print("═" * 60)


if __name__ == "__main__":
    main()
