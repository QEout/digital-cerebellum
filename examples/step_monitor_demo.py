#!/usr/bin/env python3
"""
Step Monitor Demo — shows predictive error interception in action.

Simulates a 10-step desktop automation task where Step 4 goes wrong.
Without a cerebellum: the agent blindly continues, all subsequent steps fail.
With a cerebellum: the error is caught immediately, cascade prevented.

Usage:
    python examples/step_monitor_demo.py

No external dependencies needed (no LLM, no game, no desktop).
"""

from digital_cerebellum.monitor import StepMonitor


def main():
    print("=" * 70)
    print("  Digital Cerebellum — Step Monitor Demo")
    print("  Predictive Error Interception for AI Agents")
    print("=" * 70)

    # ── Phase 1: Train the cerebellum on a successful run ──
    print("\n📚 Phase 1: Learning from a successful task execution...\n")

    monitor = StepMonitor()

    successful_task = [
        ("open file manager",        "desktop with icons",         "file manager window appeared"),
        ("navigate to documents",    "file manager showing root",  "documents folder contents shown"),
        ("select report.pdf",        "documents folder open",      "report.pdf is highlighted"),
        ("click open button",        "report.pdf selected",        "PDF viewer opened with report"),
        ("scroll to page 3",         "PDF on page 1",              "PDF showing page 3"),
        ("highlight key paragraph",  "PDF on page 3",              "text is highlighted in yellow"),
        ("copy selected text",       "text highlighted",           "text copied to clipboard"),
        ("switch to email app",      "PDF viewer active",          "email compose window open"),
        ("paste into email body",    "empty email draft",          "email body contains pasted text"),
        ("click send",               "email with content",         "email sent confirmation shown"),
    ]

    for round_num in range(3):
        for action, state, outcome in successful_task:
            monitor.before_step(action=action, state=state)
            monitor.after_step(outcome=outcome, success=True)
        monitor.reset()

    fm_stats = monitor.forward_model.stats
    print(f"  Trained on 3 successful runs ({fm_stats['step']} steps)")
    print(f"  Forward model error: {fm_stats['mean_recent_error']:.6f}")
    print(f"  Forward model improving: {fm_stats['is_improving']}")

    # ── Phase 2: Run with an error at step 4 ──
    print("\n" + "=" * 70)
    print("\n🚨 Phase 2: Running task with error at step 4...\n")

    task_with_error = [
        ("open file manager",        "desktop with icons",         "file manager window appeared",     True),
        ("navigate to documents",    "file manager showing root",  "documents folder contents shown",  True),
        ("select report.pdf",        "documents folder open",      "report.pdf is highlighted",        True),
        ("click open button",        "report.pdf selected",        "error: file not found, PDF viewer shows blank screen", False),
        ("scroll to page 3",         "PDF on page 1",              "error: no document loaded",        False),
        ("highlight key paragraph",  "PDF on page 3",              "error: no text to select",         False),
        ("copy selected text",       "text highlighted",           "error: clipboard empty",           False),
        ("switch to email app",      "PDF viewer active",          "email compose window open",        True),
        ("paste into email body",    "empty email draft",          "email body is empty, nothing pasted", False),
        ("click send",               "email with content",         "sent empty email to recipient",    False),
    ]

    paused_at = None
    for i, (action, state, outcome, expected_success) in enumerate(task_with_error, 1):
        pred = monitor.before_step(action=action, state=state)

        status = "✅" if pred.should_proceed else "🛑"
        warning = ""
        if pred.failure_warning:
            warning = f" ⚠️  FAILURE WARNING: {pred.failure_warning.pattern_description[:60]}"

        print(f"  Step {i:2d}: {status} {action:<30s} | risk={pred.risk_score:.2f} conf={pred.confidence:.2f}{warning}")

        if not pred.should_proceed:
            print(f"\n  🛑 CEREBELLUM SAYS STOP before step {i}!")
            print(f"     Risk score: {pred.risk_score:.3f}")
            print(f"     Cascade risk: {pred.cascade_risk:.3f}")
            if pred.failure_warning:
                print(f"     Warning: {pred.failure_warning.pattern_description}")
            paused_at = i
            # Still execute to show the verdict
            verdict = monitor.after_step(outcome=outcome, success=expected_success)
            break

        verdict = monitor.after_step(outcome=outcome, success=expected_success)

        if verdict.should_pause:
            print(f"\n  🛑 CEREBELLUM DETECTED CASCADE after step {i}!")
            print(f"     SPE: {verdict.spe:.3f}")
            print(f"     Consecutive errors: {verdict.consecutive_errors}")
            print(f"     Suggestion: {verdict.suggestion}")
            paused_at = i
            break

    # ── Phase 3: Summary ──
    print("\n" + "=" * 70)
    print("\n📊 Results:\n")

    summary = monitor.episode_summary

    if paused_at:
        remaining = len(task_with_error) - paused_at
        wasted_without = len(task_with_error) - 3  # steps 4-10 are all wrong
        print(f"  Without cerebellum: Agent runs all 10 steps, {wasted_without} wasted")
        print(f"  With cerebellum:    Stopped at step {paused_at}, saved {remaining} wasted steps")
        print(f"  Damage prevented:   Avoided sending empty email to recipient")
    else:
        print(f"  Task completed all {len(task_with_error)} steps (no cascade detected)")

    print(f"\n  Episode stats:")
    print(f"    Steps executed: {summary['steps']}")
    print(f"    Mean SPE:       {summary.get('mean_spe', 0):.4f}")
    print(f"    Max SPE:        {summary.get('max_spe', 0):.4f}")
    print(f"    Success rate:   {summary.get('success_rate', 'N/A')}")

    monitor_stats = monitor.stats
    print(f"\n  Cerebellum state:")
    print(f"    Forward model steps:  {monitor_stats['forward_model']['step']}")
    print(f"    Failure patterns:     {monitor_stats['failure_memory']['stored_failures']}")
    print(f"    Total pauses:         {monitor_stats['total_pauses']}")

    print("\n" + "=" * 70)
    print("  This is what Digital Cerebellum does:")
    print("  Faster for what it knows, careful for what it doesn't.")
    print("=" * 70)


if __name__ == "__main__":
    main()
