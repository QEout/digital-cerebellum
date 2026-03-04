#!/usr/bin/env python3
"""
OpenClaw Desktop Automation Benchmark.

Simulates 5 real-world OpenClaw-style desktop automation tasks and measures
the Digital Cerebellum's impact on speed (SkillStore) and reliability
(StepMonitor + AutoRollback).

Four rounds:
  Round 1 (Cold):     First encounter, every task goes through "LLM"
  Round 2 (Warm):     Same tasks repeated, SkillStore kicks in
  Round 3 (Failure):  Each task runs with a realistic fault injection
  Round 4 (Curve):    10 rounds measuring learning convergence

Usage:
    python benchmarks/openclaw_benchmark.py
    python benchmarks/openclaw_benchmark.py --verbose
    python benchmarks/openclaw_benchmark.py --json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

from digital_cerebellum.main import DigitalCerebellum
from digital_cerebellum.monitor import StepMonitor


# ======================================================================
# Data model
# ======================================================================

@dataclass
class Step:
    action: str
    state: str
    outcome_ok: str
    outcome_fail: str


@dataclass
class Scenario:
    name: str
    task_query: str
    resolution: str
    steps: list[Step]
    failure_step: int
    cascade_consequence: str


@dataclass
class SpeedResult:
    name: str
    cold_ms: float
    warm_ms: float
    speedup: float
    skill_hit: bool


@dataclass
class ReliabilityResult:
    name: str
    total_steps: int
    failure_step: int
    detected_at: int | None
    steps_wasted_no_cb: int
    steps_wasted_cb: int
    steps_saved: int
    rollback_to: int | None
    rollback_correct: bool


@dataclass
class CurvePoint:
    round_num: int
    hit_rate: float
    avg_latency_ms: float


@dataclass
class BenchmarkResults:
    speed: list[SpeedResult]
    reliability: list[ReliabilityResult]
    curve: list[CurvePoint]


# ======================================================================
# 5 OpenClaw scenarios — realistic desktop automation tasks
# ======================================================================

SCENARIOS: list[Scenario] = [
    # ── 1. Email Processing ──
    Scenario(
        name="Email Processing",
        task_query="Download the invoice attachment from the latest email and reply with confirmation",
        resolution="Invoice downloaded, confirmation reply sent, email archived",
        steps=[
            Step("open email client", "desktop idle",
                 "email client opened, inbox visible with 23 unread",
                 "email client opened, inbox visible with 23 unread"),
            Step("search inbox for 'invoice'", "email client open",
                 "found 3 emails matching 'invoice'",
                 "found 3 emails matching 'invoice'"),
            Step("open most recent invoice email", "search results shown",
                 "email opened: from vendor@acme.com, subject 'March Invoice'",
                 "email opened: from vendor@acme.com, subject 'March Invoice'"),
            Step("download PDF attachment", "email content visible",
                 "invoice_march_2026.pdf downloaded to ~/Downloads (2.3MB)",
                 "error: attachment download timed out after 30s, file is 0 bytes"),
            Step("open PDF to verify contents", "PDF downloaded",
                 "PDF viewer shows invoice: Acme Corp, total $4,500.00",
                 "error: cannot open file, PDF is corrupted/empty"),
            Step("compose reply with confirmation", "invoice verified",
                 "reply drafted: 'Invoice received, processing payment'",
                 "reply drafted referencing empty/corrupt attachment"),
            Step("send reply and archive email", "reply composed",
                 "reply sent, email moved to 'Processed' folder",
                 "sent confirmation for invoice never actually received"),
        ],
        failure_step=3,
        cascade_consequence="Confirmed receipt of invoice that was never downloaded — vendor thinks payment is coming",
    ),

    # ── 2. Calendar Management ──
    Scenario(
        name="Calendar Management",
        task_query="Schedule a team standup meeting for tomorrow at 9am with the engineering team",
        resolution="Meeting created at 9am local time, all engineers invited, reminder set",
        steps=[
            Step("open calendar application", "desktop idle",
                 "calendar app opened, showing today's schedule",
                 "calendar app opened, showing today's schedule"),
            Step("navigate to tomorrow's date", "calendar on today",
                 "showing March 5, 2026 — 2 existing meetings",
                 "showing March 5, 2026 — 2 existing meetings"),
            Step("create new event at 9:00 AM", "tomorrow's schedule visible",
                 "new event created: 9:00 AM local time (PST)",
                 "error: calendar API returned UTC instead of local — event created at 9:00 UTC (1:00 AM PST)"),
            Step("set title to 'Engineering Standup'", "event created",
                 "title set: 'Engineering Standup'",
                 "title set on wrong-timezone event"),
            Step("add engineering team members", "title set",
                 "invited: alice@, bob@, carol@, dave@ — 4 attendees",
                 "invited team to 1:00 AM meeting"),
            Step("set 15-minute reminder", "attendees added",
                 "reminder set: 15 minutes before (8:45 AM)",
                 "reminder set at 12:45 AM — everyone gets woken up"),
            Step("save and send invitations", "reminder configured",
                 "event saved, invitations sent to all attendees",
                 "sent invitations for 1:00 AM meeting to entire team"),
        ],
        failure_step=2,
        cascade_consequence="Team gets invited to a 1:00 AM meeting instead of 9:00 AM — everyone's alarm goes off at midnight",
    ),

    # ── 3. Browser Form Submission ──
    Scenario(
        name="Browser Form Submission",
        task_query="Fill out the expense report form on the company portal with this month's receipts",
        resolution="Expense report submitted with all receipts attached",
        steps=[
            Step("open browser and navigate to company portal", "desktop idle",
                 "company portal login page loaded",
                 "company portal login page loaded"),
            Step("log in with SSO credentials", "login page shown",
                 "authenticated, dashboard visible",
                 "authenticated, dashboard visible"),
            Step("navigate to expense report form", "dashboard visible",
                 "expense form loaded, new report started",
                 "expense form loaded, new report started"),
            Step("fill page 1: trip details and dates", "form page 1",
                 "page 1 complete: NYC trip, March 1-3, business purpose filled",
                 "page 1 complete: NYC trip, March 1-3, business purpose filled"),
            Step("click next to page 2", "page 1 filled",
                 "page 2 loaded: itemized expenses",
                 "error: session expired, redirected to login page — form data lost"),
            Step("fill page 2: itemized expenses with receipts", "page 2 loaded",
                 "5 expenses entered: hotel $890, flights $450, meals $230, taxi $95, conference $300",
                 "filling form on login page — data going nowhere"),
            Step("click next to page 3: approval", "page 2 filled",
                 "page 3 loaded: supervisor approval selection",
                 "login page reloads, no form visible"),
            Step("select supervisor and submit", "page 3 loaded",
                 "expense report #ER-2026-0847 submitted for approval",
                 "submitted empty form or triggered error"),
        ],
        failure_step=4,
        cascade_consequence="Session expired mid-form; agent keeps filling fields on the login page — report never submitted",
    ),

    # ── 4. File Organization ──
    Scenario(
        name="File Organization",
        task_query="Organize the Downloads folder: sort files by type into subfolders",
        resolution="All files sorted into Documents, Images, Videos, Archives, Other",
        steps=[
            Step("scan Downloads folder for all files", "Downloads folder exists",
                 "found 127 files: 45 PDFs, 32 images, 18 videos, 15 archives, 17 other",
                 "found 127 files: 45 PDFs, 32 images, 18 videos, 15 archives, 17 other"),
            Step("create category subfolders", "file listing complete",
                 "created: Documents/, Images/, Videos/, Archives/, Other/",
                 "created: Documents/, Images/, Videos/, Archives/, Other/"),
            Step("move PDF and document files to Documents/", "subfolders created",
                 "moved 45 files to Documents/ (PDFs, DOCX, TXT)",
                 "error: glob pattern *.{pdf,doc}* also matched .docker/ config — moved system files"),
            Step("move image files to Images/", "documents moved",
                 "moved 32 files to Images/ (PNG, JPG, SVG)",
                 "glob matched .icons/ directory — moved desktop icons"),
            Step("move video files to Videos/", "images moved",
                 "moved 18 files to Videos/ (MP4, MOV, AVI)",
                 "moved screen recording cache files"),
            Step("move archives to Archives/", "videos moved",
                 "moved 15 files to Archives/ (ZIP, TAR, GZ)",
                 "moved .git pack files from a project directory"),
            Step("move remaining to Other/", "archives moved",
                 "moved 17 remaining files to Other/",
                 "moved more system files"),
            Step("verify file counts match original scan", "all moved",
                 "verification passed: 127 files accounted for in 5 folders",
                 "verification fails: 340 files found (included hidden/system files)"),
        ],
        failure_step=2,
        cascade_consequence="Glob pattern too aggressive — system config files moved, desktop icons gone, git repos broken",
    ),

    # ── 5. Shell Deploy Sequence ──
    Scenario(
        name="Shell Deploy Sequence",
        task_query="Deploy the latest version of the web app to production",
        resolution="Latest code pulled, built, tested, deployed, and verified",
        steps=[
            Step("git pull origin main", "project directory, on main branch",
                 "pulled 3 commits: 'fix auth bug', 'update deps', 'add caching'",
                 "pulled 3 commits: 'fix auth bug', 'update deps', 'add caching'"),
            Step("npm install to update dependencies", "code pulled",
                 "installed 847 packages, 0 vulnerabilities",
                 "installed 847 packages, 0 vulnerabilities"),
            Step("npm run build", "dependencies installed",
                 "build successful: dist/ created (2.4MB, 12 chunks)",
                 "error: build failed — TypeScript error in auth.ts line 47, but exit code 0 (swallowed by CI script)"),
            Step("npm run test", "build complete",
                 "143 tests passed, 0 failed, coverage 87%",
                 "tests ran against stale dist/ from previous build — all pass (false green)"),
            Step("docker build -t webapp:latest .", "tests pass",
                 "Docker image built: webapp:latest (234MB)",
                 "Docker image contains broken build artifacts"),
            Step("docker push to registry", "image built",
                 "pushed webapp:latest to registry.company.com",
                 "pushed broken image to production registry"),
            Step("kubectl rollout restart deployment/webapp", "image pushed",
                 "rolling restart: 3/3 pods updated, all healthy",
                 "pods crash-loop: new image has broken auth module"),
            Step("verify health endpoint returns 200", "rollout complete",
                 "GET /health → 200 OK, response time 45ms",
                 "GET /health → 500 Internal Server Error"),
        ],
        failure_step=2,
        cascade_consequence="Build failure silently swallowed — stale artifacts deployed to production, auth module broken for all users",
    ),
]

# Paraphrased queries for Round 2 (SkillStore warm path)
WARM_QUERIES = [
    "Download the invoice attachment from my latest email and reply to confirm",
    "Schedule an engineering standup meeting for tomorrow at 9am with the team",
    "Fill out the expense report form on the company portal for this month",
    "Organize the Downloads folder by sorting files into subfolders by type",
    "Deploy the latest version of the web application to production",
]


# ======================================================================
# Simulation helpers
# ======================================================================

def simulate_llm_latency():
    """Simulate LLM response time (scaled down for benchmark)."""
    time.sleep(0.02)


# ======================================================================
# Round 1: Cold start — learn skills
# ======================================================================

def run_round1(cb: DigitalCerebellum, monitor: StepMonitor, verbose: bool = False) -> list[dict]:
    """Cold start: every task goes through LLM, cerebellum learns."""
    results = []
    for scenario in SCENARIOS:
        t0 = time.perf_counter()

        simulate_llm_latency()
        cb.learn_skill(
            input_text=scenario.task_query,
            response_text=scenario.resolution,
            domain="desktop_automation",
        )

        for step in scenario.steps:
            monitor.before_step(action=step.action, state=step.state)
            monitor.after_step(outcome=step.outcome_ok, success=True)

        elapsed = (time.perf_counter() - t0) * 1000
        monitor.reset()

        results.append({
            "name": scenario.name,
            "path": "cortex",
            "latency_ms": elapsed,
        })

        if verbose:
            print(f"  [{scenario.name:<25s}] path=cortex   | {elapsed:>7.1f}ms")

    # Reinforce all learned skills
    for skill in cb.skill_store._skills.values():
        for _ in range(4):
            cb.skill_store.reinforce(skill.id)

    return results


# ======================================================================
# Round 2: Warm path — SkillStore fires
# ======================================================================

def run_round2(cb: DigitalCerebellum, verbose: bool = False) -> list[SpeedResult]:
    """Warm path: similar queries, SkillStore should match."""
    results = []
    for scenario, warm_query in zip(SCENARIOS, WARM_QUERIES):
        t0 = time.perf_counter()
        match = cb.match_skill(warm_query)
        elapsed = (time.perf_counter() - t0) * 1000

        hit = match is not None and match.should_execute
        if not hit:
            simulate_llm_latency()
            elapsed = (time.perf_counter() - t0) * 1000

        results.append(SpeedResult(
            name=scenario.name,
            cold_ms=0.0,
            warm_ms=elapsed,
            speedup=0.0,
            skill_hit=hit,
        ))

        if verbose:
            path = "cerebellum" if hit else "cortex"
            print(f"  [{scenario.name:<25s}] path={path:<11s} | {elapsed:>7.1f}ms")

    return results


# ======================================================================
# Round 3: Failure injection — StepMonitor + AutoRollback
# ======================================================================

def run_round3(verbose: bool = False) -> list[ReliabilityResult]:
    """Each scenario runs with failure injection, measure detection + rollback."""
    results = []

    for scenario in SCENARIOS:
        monitor = StepMonitor(cascade_consecutive_limit=2)

        # Train on successful runs
        for _ in range(5):
            for step in scenario.steps:
                monitor.before_step(action=step.action, state=step.state)
                monitor.after_step(outcome=step.outcome_ok, success=True)
            monitor.reset()

        # Run with failure
        detected_at = None
        for i, step in enumerate(scenario.steps):
            pred = monitor.before_step(action=step.action, state=step.state)

            if not pred.should_proceed:
                detected_at = i + 1
                monitor.after_step(
                    outcome=step.outcome_fail if i >= scenario.failure_step else step.outcome_ok
                )
                break

            is_failed = i >= scenario.failure_step
            outcome = step.outcome_fail if is_failed else step.outcome_ok

            verdict = monitor.after_step(
                outcome=outcome,
                success=not is_failed,
            )

            if verdict.should_pause:
                detected_at = i + 1
                break

        plan = monitor.get_rollback_plan()

        wasted_no_cb = len(scenario.steps) - scenario.failure_step - 1
        wasted_cb = max(0, (detected_at or len(scenario.steps)) - scenario.failure_step - 1)
        saved = wasted_no_cb - wasted_cb

        rollback_to = plan.rollback_to_step if plan else None
        rollback_correct = (
            plan is not None and plan.rollback_to_step <= scenario.failure_step
        )

        results.append(ReliabilityResult(
            name=scenario.name,
            total_steps=len(scenario.steps),
            failure_step=scenario.failure_step + 1,
            detected_at=detected_at,
            steps_wasted_no_cb=wasted_no_cb,
            steps_wasted_cb=wasted_cb,
            steps_saved=saved,
            rollback_to=rollback_to,
            rollback_correct=rollback_correct,
        ))

        if verbose:
            det = f"step {detected_at}" if detected_at else "NOT CAUGHT"
            rb = f"step {rollback_to}" if rollback_to is not None else "N/A"
            print(f"  [{scenario.name:<25s}] fail@{scenario.failure_step+1} detect@{det:<8s} saved={saved} rollback={rb}")

    return results


# ======================================================================
# Round 4: Learning curve — 10 rounds
# ======================================================================

def run_round4(verbose: bool = False) -> list[CurvePoint]:
    """10 rounds measuring SkillStore convergence with paraphrased queries."""
    cb = DigitalCerebellum()
    curve = []

    for round_num in range(1, 11):
        hits = 0
        total_ms = 0.0

        for scenario, warm_query in zip(SCENARIOS, WARM_QUERIES):
            query = warm_query if round_num > 1 else scenario.task_query

            t0 = time.perf_counter()
            match = cb.match_skill(query)
            elapsed = (time.perf_counter() - t0) * 1000

            if match is not None and match.should_execute:
                hits += 1
            else:
                simulate_llm_latency()
                elapsed = (time.perf_counter() - t0) * 1000
                cb.learn_skill(
                    input_text=scenario.task_query,
                    response_text=scenario.resolution,
                    domain="desktop_automation",
                )

            total_ms += elapsed

        for skill in cb.skill_store._skills.values():
            cb.skill_store.reinforce(skill.id)

        hit_rate = hits / len(SCENARIOS)
        avg_ms = total_ms / len(SCENARIOS)
        curve.append(CurvePoint(round_num, hit_rate, avg_ms))

        if verbose:
            print(f"  Round {round_num:>2d} | hit_rate={hit_rate:>5.0%} | avg={avg_ms:>8.1f}ms")

    return curve


# ======================================================================
# Main benchmark
# ======================================================================

def run_benchmark(verbose: bool = False) -> BenchmarkResults:
    """Run all 4 rounds and return structured results."""
    cb = DigitalCerebellum()
    monitor = StepMonitor()

    # Round 1
    if verbose:
        print("\n  ROUND 1: Cold start (LLM path, learning)")
        print("  " + "─" * 60)
    r1 = run_round1(cb, monitor, verbose)

    # Round 2
    if verbose:
        print(f"\n  ROUND 2: Warm path (SkillStore)")
        print("  " + "─" * 60)
    speed_results = run_round2(cb, verbose)
    for sr, r1_data in zip(speed_results, r1):
        sr.cold_ms = r1_data["latency_ms"]
        sr.speedup = sr.cold_ms / sr.warm_ms if sr.warm_ms > 0 else float('inf')

    # Round 3
    if verbose:
        print(f"\n  ROUND 3: Failure injection (StepMonitor + AutoRollback)")
        print("  " + "─" * 60)
    reliability_results = run_round3(verbose)

    # Round 4
    if verbose:
        print(f"\n  ROUND 4: Learning curve (10 rounds)")
        print("  " + "─" * 60)
    curve = run_round4(verbose)

    return BenchmarkResults(
        speed=speed_results,
        reliability=reliability_results,
        curve=curve,
    )


def print_report(results: BenchmarkResults):
    """Print formatted benchmark report."""
    W = 80

    print("\n" + "=" * W)
    print("  OpenClaw Desktop Automation Benchmark")
    print("  Digital Cerebellum — Speed + Reliability")
    print("=" * W)

    # ── Speed ──
    print(f"\n  SPEED (SkillStore):")
    print(f"  {'Task':<25s} {'Cold':>8s} {'Warm':>8s} {'Speedup':>8s} {'Hit':>5s}")
    print("  " + "─" * 58)
    for sr in results.speed:
        hit_str = "YES" if sr.skill_hit else "no"
        print(f"  {sr.name:<25s} {sr.cold_ms:>7.0f}ms {sr.warm_ms:>7.1f}ms {sr.speedup:>7.0f}x {hit_str:>5s}")

    hits = sum(1 for sr in results.speed if sr.skill_hit)
    avg_cold = sum(sr.cold_ms for sr in results.speed) / len(results.speed)
    avg_warm = sum(sr.warm_ms for sr in results.speed) / len(results.speed)
    avg_speedup = avg_cold / avg_warm if avg_warm > 0 else 0

    print("  " + "─" * 58)
    print(f"  {'AVERAGE':<25s} {avg_cold:>7.0f}ms {avg_warm:>7.1f}ms {avg_speedup:>7.0f}x {hits}/{len(results.speed)}")

    # ── Reliability ──
    print(f"\n  RELIABILITY (StepMonitor + AutoRollback):")
    print(f"  {'Task':<25s} {'Fail@':>5s} {'Det@':>5s} {'NoMonitor':>9s} {'Monitor':>7s} {'Saved':>5s} {'Rollback':>8s}")
    print("  " + "─" * 68)
    for rr in results.reliability:
        det = str(rr.detected_at) if rr.detected_at else "MISS"
        rb = f"step {rr.rollback_to}" if rr.rollback_to is not None else "N/A"
        rb_mark = " OK" if rr.rollback_correct else ""
        print(f"  {rr.name:<25s} {rr.failure_step:>5d} {det:>5s} "
              f"{rr.steps_wasted_no_cb:>5d}w    {rr.steps_wasted_cb:>4d}w  {rr.steps_saved:>5d} {rb:>8s}{rb_mark}")

    caught = sum(1 for rr in results.reliability if rr.detected_at is not None)
    total_wasted_no = sum(rr.steps_wasted_no_cb for rr in results.reliability)
    total_wasted_cb = sum(rr.steps_wasted_cb for rr in results.reliability)
    total_saved = sum(rr.steps_saved for rr in results.reliability)
    correct_rb = sum(1 for rr in results.reliability if rr.rollback_correct)
    prevention_rate = (total_saved / total_wasted_no * 100) if total_wasted_no > 0 else 0

    print("  " + "─" * 68)
    print(f"  Cascade detection:  {caught}/{len(results.reliability)}")
    print(f"  Waste prevention:   {total_wasted_no} → {total_wasted_cb} ({prevention_rate:.0f}% reduced)")
    print(f"  Correct rollbacks:  {correct_rb}/{caught}")

    # ── Learning Curve ──
    print(f"\n  LEARNING CURVE (SkillStore convergence):")
    print(f"  {'Round':>5s} {'Hit Rate':>8s} {'Avg Latency':>12s}")
    print("  " + "─" * 30)
    for cp in results.curve:
        bar = "#" * int(cp.hit_rate * 10)
        print(f"  {cp.round_num:>5d} {cp.hit_rate:>7.0%}  {cp.avg_latency_ms:>10.1f}ms  {bar}")

    # ── Aggregate ──
    print(f"\n  " + "─" * 40)
    print(f"  AGGREGATE:")
    print(f"    Speed:       {avg_speedup:.0f}x avg speedup ({hits}/{len(results.speed)} skills matched)")
    print(f"    Reliability: {caught}/{len(results.reliability)} cascades caught, {prevention_rate:.0f}% waste prevented")
    print(f"    Rollback:    {correct_rb}/{caught} correct rollback plans")
    if results.curve:
        final = results.curve[-1]
        print(f"    Convergence: {final.hit_rate:.0%} hit rate by round {final.round_num} ({final.avg_latency_ms:.0f}ms)")
    print(f"\n" + "=" * W)


def to_json(results: BenchmarkResults) -> dict:
    """Convert results to JSON-serializable dict."""
    return {
        "speed": [
            {
                "name": sr.name,
                "cold_ms": round(sr.cold_ms, 1),
                "warm_ms": round(sr.warm_ms, 1),
                "speedup": round(sr.speedup, 1),
                "skill_hit": sr.skill_hit,
            }
            for sr in results.speed
        ],
        "reliability": [
            {
                "name": rr.name,
                "total_steps": rr.total_steps,
                "failure_step": rr.failure_step,
                "detected_at": rr.detected_at,
                "steps_wasted_no_monitor": rr.steps_wasted_no_cb,
                "steps_wasted_with_monitor": rr.steps_wasted_cb,
                "steps_saved": rr.steps_saved,
                "rollback_to": rr.rollback_to,
                "rollback_correct": rr.rollback_correct,
            }
            for rr in results.reliability
        ],
        "learning_curve": [
            {
                "round": cp.round_num,
                "hit_rate": round(cp.hit_rate, 2),
                "avg_latency_ms": round(cp.avg_latency_ms, 1),
            }
            for cp in results.curve
        ],
        "aggregate": {
            "avg_speedup": round(
                sum(sr.cold_ms for sr in results.speed)
                / max(sum(sr.warm_ms for sr in results.speed), 1e-9), 1
            ),
            "skills_matched": f"{sum(1 for sr in results.speed if sr.skill_hit)}/{len(results.speed)}",
            "cascades_caught": f"{sum(1 for rr in results.reliability if rr.detected_at)}/{len(results.reliability)}",
            "correct_rollbacks": sum(1 for rr in results.reliability if rr.rollback_correct),
        },
    }


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="OpenClaw Desktop Automation Benchmark")
    parser.add_argument("--verbose", action="store_true", help="Show per-round details")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    results = run_benchmark(verbose=args.verbose)

    if args.json:
        print(json.dumps(to_json(results), indent=2))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
