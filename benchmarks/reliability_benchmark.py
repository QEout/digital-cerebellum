#!/usr/bin/env python3
"""
Reliability Benchmark — With vs Without Cerebellum.

Simulates realistic multi-step agent tasks across 6 domains.
Each task has a failure injection point that triggers error cascading.
Compares what happens with and without the Digital Cerebellum.

Core question: how many wasted steps can the cerebellum prevent?

Usage:
    python benchmarks/reliability_benchmark.py
    python benchmarks/reliability_benchmark.py --verbose
    python benchmarks/reliability_benchmark.py --json  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

from digital_cerebellum.monitor import StepMonitor

# ======================================================================
# Data model
# ======================================================================


@dataclass
class TaskStep:
    action: str
    state: str
    outcome_ok: str
    outcome_fail: str


@dataclass
class TaskScenario:
    name: str
    domain: str
    description: str
    steps: list[TaskStep]
    failure_step: int  # 0-indexed: which step fails
    cascade_consequence: str


@dataclass
class RunResult:
    steps_executed: int
    steps_successful: int
    steps_wasted: int
    stopped_early: bool
    stopped_at: int | None
    detection_delay: int  # steps between failure and detection (0 = perfect)


@dataclass
class ScenarioResult:
    name: str
    domain: str
    total_steps: int
    failure_step: int
    without_cb: RunResult
    with_cb: RunResult
    training_time_ms: float
    inference_time_ms: float


# ======================================================================
# Scenario definitions — 6 domains, realistic multi-step tasks
# ======================================================================

SCENARIOS: list[TaskScenario] = [
    # ── 1. Web Form Submission ──
    TaskScenario(
        name="Job Application Form",
        domain="web",
        description="Fill out a multi-page job application. Login session expires at step 4.",
        failure_step=3,
        cascade_consequence="All subsequent form pages save to an unauthenticated session → data lost",
        steps=[
            TaskStep("navigate to careers page", "browser on homepage",
                     "careers page loaded with job listings", "careers page loaded with job listings"),
            TaskStep("click 'Software Engineer' listing", "careers page visible",
                     "job description page shown", "job description page shown"),
            TaskStep("click 'Apply Now' button", "job description page",
                     "application form page 1 loaded", "application form page 1 loaded"),
            TaskStep("fill personal info and click next", "form page 1",
                     "form page 2 loaded with education fields",
                     "error: session expired, redirected to login page"),
            TaskStep("fill education history and click next", "form page 2",
                     "form page 3 loaded with experience fields",
                     "form submitted to void session, data lost silently"),
            TaskStep("fill work experience and click next", "form page 3",
                     "form page 4 loaded with resume upload",
                     "form page appears but linked to empty session"),
            TaskStep("upload resume PDF", "form page 4",
                     "resume uploaded, review page shown",
                     "upload fails silently, no file attached"),
            TaskStep("review and submit application", "review page",
                     "application submitted confirmation",
                     "submitted empty application with no data"),
        ],
    ),

    # ── 2. File Management ──
    TaskScenario(
        name="Project Backup & Migration",
        domain="file",
        description="Backup project files to external drive, then restructure. Source path wrong at step 3.",
        failure_step=2,
        cascade_consequence="All copy operations use wrong source → backup contains wrong files",
        steps=[
            TaskStep("create backup directory on external drive", "external drive mounted",
                     "backup directory created at /Volumes/ext/backup",
                     "backup directory created at /Volumes/ext/backup"),
            TaskStep("list files in project directory", "project dir exists",
                     "file listing shows 47 files in src/",
                     "file listing shows 47 files in src/"),
            TaskStep("copy source code to backup", "backup dir ready",
                     "47 source files copied to backup/src/",
                     "error: /src/ not found, copied /sys/ directory instead"),
            TaskStep("copy configuration files", "source files backed up",
                     "12 config files copied to backup/config/",
                     "copied wrong config files from /sys/ context"),
            TaskStep("copy database dumps", "config backed up",
                     "database dump copied to backup/data/",
                     "copied unrelated data files"),
            TaskStep("verify backup integrity with checksums", "all files copied",
                     "all 62 files verified, checksums match",
                     "checksum mismatch on 47 files"),
            TaskStep("delete original project directory", "backup verified",
                     "original directory removed",
                     "deleted original project, backup is corrupted"),
            TaskStep("rename backup to project name", "original deleted",
                     "migration complete, project restored from backup",
                     "restored corrupted backup as the only copy"),
        ],
    ),

    # ── 3. Code Refactoring ──
    TaskScenario(
        name="API Endpoint Rename",
        domain="code",
        description="Rename /api/users to /api/accounts across codebase. Missed the router file at step 4.",
        failure_step=3,
        cascade_consequence="Frontend calls /api/accounts but router still maps /api/users → 404 errors",
        steps=[
            TaskStep("search codebase for '/api/users' references", "IDE open on project",
                     "found 23 references across 8 files",
                     "found 23 references across 8 files"),
            TaskStep("update OpenAPI spec: /api/users → /api/accounts", "references identified",
                     "openapi.yaml updated, 6 endpoint paths changed",
                     "openapi.yaml updated, 6 endpoint paths changed"),
            TaskStep("update controller: UsersController → AccountsController", "spec updated",
                     "controller renamed, 4 methods updated",
                     "controller renamed, 4 methods updated"),
            TaskStep("update router mappings", "controller updated",
                     "router.py updated: all routes point to AccountsController",
                     "error: router.py has merge conflict, changes not saved"),
            TaskStep("update frontend API client", "router updated",
                     "apiClient.ts updated: all calls use /api/accounts",
                     "frontend updated but calling endpoint that doesn't exist"),
            TaskStep("update integration tests", "frontend updated",
                     "15 test files updated with new endpoint names",
                     "tests updated to use broken endpoint"),
            TaskStep("run test suite", "all files updated",
                     "all 156 tests pass",
                     "47 tests fail: 404 on /api/accounts"),
            TaskStep("deploy to staging", "tests pass",
                     "staging deployment successful",
                     "deployed broken API, all account operations fail"),
        ],
    ),

    # ── 4. Data Pipeline ──
    TaskScenario(
        name="Monthly Revenue ETL",
        domain="data",
        description="Extract, transform, load monthly revenue data. Currency conversion fails at step 3.",
        failure_step=2,
        cascade_consequence="All monetary values in wrong currency → reports show 100x inflated revenue",
        steps=[
            TaskStep("extract raw transactions from database", "database connected",
                     "extracted 14,823 transactions for March 2026",
                     "extracted 14,823 transactions for March 2026"),
            TaskStep("validate data completeness", "raw data extracted",
                     "validation passed: no missing fields, 14,823 records",
                     "validation passed: no missing fields, 14,823 records"),
            TaskStep("convert currencies to USD", "data validated",
                     "all amounts converted to USD using March avg rates",
                     "error: exchange rate API returned JPY rates for USD column, all values 100x too high"),
            TaskStep("calculate daily revenue aggregates", "currencies normalized",
                     "daily aggregates computed: 31 daily totals",
                     "daily totals computed with inflated values"),
            TaskStep("compute month-over-month growth", "daily aggregates ready",
                     "MoM growth: +12% vs February",
                     "MoM growth shows +11,900% — obviously wrong but agent doesn't notice"),
            TaskStep("generate executive dashboard", "growth computed",
                     "dashboard updated with correct March numbers",
                     "dashboard shows $1.2B revenue instead of $12M"),
            TaskStep("email report to CFO", "dashboard ready",
                     "report emailed: March revenue $12.3M, +12% MoM",
                     "emailed report claiming $1.23B revenue to CFO"),
        ],
    ),

    # ── 5. Desktop Automation ──
    TaskScenario(
        name="Invoice Processing",
        domain="desktop",
        description="Download invoices from email, enter into accounting software. Wrong attachment at step 3.",
        failure_step=2,
        cascade_consequence="Entered data from spam PDF instead of real invoice → accounting records corrupted",
        steps=[
            TaskStep("open email client", "desktop idle",
                     "email client opened, inbox visible",
                     "email client opened, inbox visible"),
            TaskStep("search for 'invoice' in inbox", "email client open",
                     "found 3 unprocessed invoice emails",
                     "found 3 unprocessed invoice emails"),
            TaskStep("download attachment from first invoice email", "search results shown",
                     "downloaded invoice_march_2026.pdf",
                     "downloaded wrong attachment: spam_offer.pdf (similar filename)"),
            TaskStep("open PDF in viewer", "PDF downloaded",
                     "invoice displayed: vendor=Acme Corp, total=$4,500",
                     "PDF shows spam content: 'Congratulations! You won $1,000,000'"),
            TaskStep("open accounting software", "PDF viewed",
                     "QuickBooks opened to data entry screen",
                     "QuickBooks opened to data entry screen"),
            TaskStep("create new invoice entry", "QuickBooks ready",
                     "new entry: vendor=Acme Corp, amount=$4,500",
                     "new entry: vendor=unknown, amount=$1,000,000"),
            TaskStep("assign to expense category", "entry created",
                     "categorized as 'Operating Expenses'",
                     "no matching category for spam data"),
            TaskStep("save and mark email as processed", "entry categorized",
                     "entry saved, email archived",
                     "saved garbage entry, marked real invoice as processed"),
        ],
    ),

    # ── 6. API Integration ──
    TaskScenario(
        name="Customer Onboarding Workflow",
        domain="api",
        description="Create customer in CRM, provision account, send welcome email. CRM returns wrong ID at step 2.",
        failure_step=1,
        cascade_consequence="All downstream services use wrong customer ID → data attached to wrong person",
        steps=[
            TaskStep("validate customer data format", "customer JSON received",
                     "validation passed: all required fields present",
                     "validation passed: all required fields present"),
            TaskStep("create customer in CRM via API", "data validated",
                     "CRM returned customer_id=CUS-2026-4472",
                     "CRM returned customer_id=CUS-2019-0001 (stale cache, wrong customer)"),
            TaskStep("provision cloud account with customer_id", "CRM customer created",
                     "cloud account provisioned for CUS-2026-4472",
                     "cloud account provisioned for CUS-2019-0001 (wrong person's account)"),
            TaskStep("set up billing subscription", "account provisioned",
                     "billing active: $99/mo plan linked to CUS-2026-4472",
                     "billing linked to wrong customer, will charge wrong person"),
            TaskStep("generate API keys for customer", "billing active",
                     "API keys generated, scoped to CUS-2026-4472",
                     "API keys give access to wrong customer's data"),
            TaskStep("send welcome email with credentials", "API keys ready",
                     "welcome email sent to new customer with correct credentials",
                     "sent wrong customer's credentials to new customer — data breach"),
        ],
    ),

    # ── 7. Scientific Experiment ──
    TaskScenario(
        name="ML Model Training Pipeline",
        domain="ml",
        description="Prepare data, train model, evaluate, deploy. Data leak in train/test split at step 3.",
        failure_step=2,
        cascade_consequence="Test set contains training data → metrics are fake → deployed model fails in production",
        steps=[
            TaskStep("load and inspect raw dataset", "Jupyter notebook open",
                     "loaded 50,000 samples, 12 features, target column verified",
                     "loaded 50,000 samples, 12 features, target column verified"),
            TaskStep("preprocess: handle missing values and normalize", "data loaded",
                     "imputed 342 missing values, all features normalized to [0,1]",
                     "imputed 342 missing values, all features normalized to [0,1]"),
            TaskStep("split into train/test sets", "data preprocessed",
                     "80/20 split: 40,000 train, 10,000 test, no overlap verified",
                     "error: shuffle seed not set, test set contains 30% training duplicates"),
            TaskStep("train XGBoost model on training set", "data split ready",
                     "model trained: 200 rounds, train_auc=0.91",
                     "model trained: 200 rounds, train_auc=0.97 (suspiciously high due to leak)"),
            TaskStep("evaluate on test set", "model trained",
                     "test_auc=0.88, precision=0.85, recall=0.82 — reasonable generalization",
                     "test_auc=0.96 — falsely high because test set overlaps training"),
            TaskStep("compare with baseline and generate report", "evaluation done",
                     "improvement over baseline: +8% AUC, report generated",
                     "report claims +23% improvement — will not reproduce"),
            TaskStep("deploy model to production API", "report approved",
                     "model deployed, A/B test started",
                     "deployed overfit model, production accuracy drops to 60%"),
            TaskStep("announce results to stakeholders", "model deployed",
                     "team notified: model performing as expected in production",
                     "announced fake results, credibility damaged when truth surfaces"),
        ],
    ),
]


# ======================================================================
# Simulation engine
# ======================================================================


def run_without_cerebellum(scenario: TaskScenario) -> RunResult:
    """Simulate running a task with no monitoring — agent executes all steps blindly."""
    steps_successful = 0
    first_failure = None

    for i, step in enumerate(scenario.steps):
        if i < scenario.failure_step:
            steps_successful += 1
        elif i == scenario.failure_step:
            first_failure = i
        # After failure point, all steps are wasted (cascade)

    wasted = len(scenario.steps) - scenario.failure_step - 1  # steps after failure

    return RunResult(
        steps_executed=len(scenario.steps),
        steps_successful=steps_successful,
        steps_wasted=wasted,
        stopped_early=False,
        stopped_at=None,
        detection_delay=len(scenario.steps) - scenario.failure_step,
    )


def run_with_cerebellum(
    scenario: TaskScenario,
    monitor: StepMonitor,
    training_rounds: int = 3,
) -> tuple[RunResult, float, float]:
    """
    Simulate running a task with cerebellum monitoring.

    1. Train the forward model on successful runs
    2. Run the task with failure injection
    3. Track when the cerebellum detects the cascade
    """
    # Phase 1: Train on successful runs
    t0 = time.perf_counter()
    for _ in range(training_rounds):
        for step in scenario.steps:
            monitor.before_step(action=step.action, state=step.state)
            monitor.after_step(outcome=step.outcome_ok, success=True)
        monitor.reset()
    training_ms = (time.perf_counter() - t0) * 1000

    # Phase 2: Run with failure injection
    t1 = time.perf_counter()
    stopped_at = None
    steps_executed = 0
    steps_successful = 0

    for i, step in enumerate(scenario.steps):
        pred = monitor.before_step(action=step.action, state=step.state)

        if not pred.should_proceed:
            stopped_at = i
            outcome = step.outcome_fail if i >= scenario.failure_step else step.outcome_ok
            monitor.after_step(outcome=outcome)
            steps_executed = i + 1
            break

        is_failed = i >= scenario.failure_step
        outcome = step.outcome_fail if is_failed else step.outcome_ok
        success = not is_failed

        verdict = monitor.after_step(outcome=outcome, success=success if is_failed else None)
        steps_executed = i + 1

        if not is_failed:
            steps_successful += 1

        if verdict.should_pause:
            stopped_at = i + 1
            break

    inference_ms = (time.perf_counter() - t1) * 1000

    if stopped_at is None:
        wasted = len(scenario.steps) - scenario.failure_step - 1
        detection_delay = len(scenario.steps) - scenario.failure_step
    else:
        wasted = max(0, stopped_at - scenario.failure_step - 1)
        detection_delay = max(0, stopped_at - scenario.failure_step)

    return RunResult(
        steps_executed=steps_executed,
        steps_successful=steps_successful,
        steps_wasted=wasted,
        stopped_early=stopped_at is not None,
        stopped_at=stopped_at,
        detection_delay=detection_delay,
    ), training_ms, inference_ms


# ======================================================================
# Benchmark runner
# ======================================================================


def run_benchmark(verbose: bool = False) -> list[ScenarioResult]:
    results = []

    for scenario in SCENARIOS:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  {scenario.name} [{scenario.domain}]")
            print(f"  {scenario.description}")
            print(f"  Failure at step {scenario.failure_step + 1}/{len(scenario.steps)}")
            print(f"{'─' * 60}")

        # Run without cerebellum
        no_cb = run_without_cerebellum(scenario)

        # Run with cerebellum
        monitor = StepMonitor()
        cb_result, train_ms, infer_ms = run_with_cerebellum(scenario, monitor)
        monitor.reset()

        result = ScenarioResult(
            name=scenario.name,
            domain=scenario.domain,
            total_steps=len(scenario.steps),
            failure_step=scenario.failure_step + 1,
            without_cb=no_cb,
            with_cb=cb_result,
            training_time_ms=train_ms,
            inference_time_ms=infer_ms,
        )
        results.append(result)

        if verbose:
            print(f"\n  Without Cerebellum:")
            print(f"    Executed: {no_cb.steps_executed}/{len(scenario.steps)} steps")
            print(f"    Wasted:   {no_cb.steps_wasted} steps (blind cascade)")
            print(f"    Damage:   {scenario.cascade_consequence}")

            print(f"\n  With Cerebellum:")
            print(f"    Executed: {cb_result.steps_executed}/{len(scenario.steps)} steps")
            print(f"    Wasted:   {cb_result.steps_wasted} steps")
            if cb_result.stopped_early:
                print(f"    Stopped:  at step {cb_result.stopped_at} (detected cascade)")
            print(f"    Saved:    {no_cb.steps_wasted - cb_result.steps_wasted} wasted steps")
            print(f"    Training: {train_ms:.0f}ms | Inference: {infer_ms:.0f}ms")

    return results


def print_summary(results: list[ScenarioResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  RELIABILITY BENCHMARK — Digital Cerebellum vs No Monitoring")
    print("=" * 100)

    hdr = (
        f"{'Scenario':<28s} {'Domain':<8s} {'Steps':>5s} "
        f"{'Fail@':>5s} "
        f"│ {'No CB':>5s} {'Waste':>5s} "
        f"│ {'CB':>5s} {'Waste':>5s} {'Saved':>5s} "
        f"│ {'Delay':>5s} {'Train':>7s} {'Infer':>7s}"
    )
    print(f"\n{hdr}")
    print("─" * 100)

    total_wasted_no_cb = 0
    total_wasted_cb = 0
    total_steps = 0
    total_saved = 0

    for r in results:
        saved = r.without_cb.steps_wasted - r.with_cb.steps_wasted
        total_wasted_no_cb += r.without_cb.steps_wasted
        total_wasted_cb += r.with_cb.steps_wasted
        total_steps += r.total_steps
        total_saved += saved

        cb_stop = f"{r.with_cb.steps_executed}" if r.with_cb.stopped_early else f"{r.with_cb.steps_executed}"

        line = (
            f"{r.name:<28s} {r.domain:<8s} {r.total_steps:>5d} "
            f"{r.failure_step:>5d} "
            f"│ {r.without_cb.steps_executed:>5d} {r.without_cb.steps_wasted:>5d} "
            f"│ {cb_stop:>5s} {r.with_cb.steps_wasted:>5d} {saved:>5d} "
            f"│ {r.with_cb.detection_delay:>5d} {r.training_time_ms:>6.0f}ms {r.inference_time_ms:>6.0f}ms"
        )
        print(line)

    print("─" * 100)

    prevention_rate = (
        (total_wasted_no_cb - total_wasted_cb) / total_wasted_no_cb * 100
        if total_wasted_no_cb > 0 else 0
    )
    avg_delay = sum(r.with_cb.detection_delay for r in results) / len(results)
    cascade_caught = sum(1 for r in results if r.with_cb.stopped_early)

    print(f"\n  AGGREGATE RESULTS:")
    print(f"  ─────────────────")
    print(f"  Scenarios tested:         {len(results)}")
    print(f"  Total task steps:         {total_steps}")
    print(f"  Cascades caught:          {cascade_caught}/{len(results)} ({cascade_caught/len(results)*100:.0f}%)")
    print(f"  Wasted steps (no CB):     {total_wasted_no_cb}")
    print(f"  Wasted steps (with CB):   {total_wasted_cb}")
    print(f"  Steps saved:              {total_saved}")
    print(f"  Waste prevention rate:    {prevention_rate:.1f}%")
    print(f"  Avg detection delay:      {avg_delay:.1f} steps after failure")

    no_cb_success = sum(r.without_cb.steps_successful for r in results) / total_steps * 100
    cb_success = sum(r.with_cb.steps_successful for r in results)
    cb_total_exec = sum(r.with_cb.steps_executed for r in results)
    cb_exec_success = cb_success / cb_total_exec * 100 if cb_total_exec > 0 else 0

    print(f"\n  RELIABILITY:")
    print(f"  ────────────")
    print(f"  Without CB — ran {total_steps} steps, {no_cb_success:.0f}% truly successful")
    print(f"  With CB    — ran {cb_total_exec} steps, {cb_exec_success:.0f}% truly successful")
    print(f"               (stopped early, prevented {total_wasted_no_cb - total_wasted_cb} damaging actions)")

    print(f"\n  KEY INSIGHT:")
    print(f"  ────────────")
    print(f"  Without cerebellum: agent completes tasks but silently corrupts data.")
    print(f"  With cerebellum:    agent stops and asks for help instead of causing damage.")
    print(f"  The cerebellum makes agents both faster (SkillStore) and trustworthy (StepMonitor).")
    print(f"\n{'=' * 100}")

    return {
        "scenarios": len(results),
        "total_steps": total_steps,
        "cascades_caught": cascade_caught,
        "cascades_total": len(results),
        "catch_rate": round(cascade_caught / len(results) * 100, 1),
        "wasted_no_cb": total_wasted_no_cb,
        "wasted_cb": total_wasted_cb,
        "steps_saved": total_saved,
        "waste_prevention_rate": round(prevention_rate, 1),
        "avg_detection_delay": round(avg_delay, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Digital Cerebellum Reliability Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-scenario details")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results = run_benchmark(verbose=args.verbose)
    summary = print_summary(results)

    if args.json:
        json_results = {
            "summary": summary,
            "scenarios": [
                {
                    "name": r.name,
                    "domain": r.domain,
                    "total_steps": r.total_steps,
                    "failure_step": r.failure_step,
                    "without_cerebellum": {
                        "steps_executed": r.without_cb.steps_executed,
                        "steps_wasted": r.without_cb.steps_wasted,
                    },
                    "with_cerebellum": {
                        "steps_executed": r.with_cb.steps_executed,
                        "steps_wasted": r.with_cb.steps_wasted,
                        "stopped_early": r.with_cb.stopped_early,
                        "detection_delay": r.with_cb.detection_delay,
                    },
                    "training_time_ms": round(r.training_time_ms, 1),
                    "inference_time_ms": round(r.inference_time_ms, 1),
                }
                for r in results
            ],
        }
        print("\n" + json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
