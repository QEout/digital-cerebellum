#!/usr/bin/env python3
"""
WebArena Benchmark Analysis.

Loads benchmark results and produces comparison tables, ablation analysis,
and paper-ready data exports.

Usage:
    python -m benchmarks.webarena.analysis --results-dir benchmarks/results
    python -m benchmarks.webarena.analysis --results-dir benchmarks/results --latex
    python -m benchmarks.webarena.analysis --results-dir benchmarks/results --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ======================================================================
# Data loading
# ======================================================================

def load_run(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_runs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all webarena_*.json files from the results directory."""
    runs = {}
    for p in sorted(results_dir.glob("webarena_*.json")):
        data = load_run(p)
        if "summary" not in data or "tasks" not in data:
            continue
        label = data["summary"].get("config", p.stem)
        runs[label] = data
    return runs


# ======================================================================
# Analysis functions
# ======================================================================

def compute_ablation_table(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute ablation table showing each component's contribution."""
    rows = []
    for label, data in runs.items():
        s = data.get("summary", {})
        tasks = data.get("tasks", [])

        n_errors = sum(1 for t in tasks if t.get("error"))
        n_passed = sum(1 for t in tasks if t.get("score", 0) > 0)
        total = len(tasks) or 1

        rows.append({
            "config": label,
            "tasks": total,
            "passed": n_passed,
            "pass_rate": round(n_passed / total, 4),
            "avg_steps": s.get("avg_steps", 0),
            "avg_llm_calls": s.get("avg_llm_calls", 0),
            "total_skill_hits": s.get("total_skill_hits", 0),
            "total_cascades": s.get("total_cascades_caught", 0),
            "total_time_s": s.get("total_time_s", 0),
            "error_rate": round(n_errors / total, 4),
        })
    return rows


def compute_per_task_comparison(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare baseline vs full cerebellum task-by-task."""
    baseline_tasks = {t["task_id"]: t for t in runs.get("baseline", {}).get("tasks", [])}
    full_tasks = {t["task_id"]: t for t in runs.get("full", {}).get("tasks", [])}

    comparisons = []
    for tid in sorted(set(baseline_tasks) | set(full_tasks)):
        bt = baseline_tasks.get(tid, {})
        ft = full_tasks.get(tid, {})
        comparisons.append({
            "task_id": tid,
            "intent": (bt or ft).get("intent", "")[:80],
            "baseline_score": bt.get("score", -1),
            "cerebellum_score": ft.get("score", -1),
            "baseline_steps": bt.get("steps", -1),
            "cerebellum_steps": ft.get("steps", -1),
            "skill_hits": ft.get("skill_hits", 0),
            "cascade_caught": ft.get("cascades_caught", 0),
            "improved": ft.get("score", 0) > bt.get("score", 0),
            "regressed": ft.get("score", 0) < bt.get("score", 0),
        })
    return comparisons


def compute_cascade_analysis(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Analyze cascade detection effectiveness."""
    full_tasks = runs.get("full", {}).get("tasks", [])
    baseline_tasks = runs.get("baseline", {}).get("tasks", [])

    baseline_errors = {t["task_id"] for t in baseline_tasks if t.get("error")}
    full_caught = [t for t in full_tasks if t.get("cascades_caught", 0) > 0]

    prevented = [t for t in full_caught if t["task_id"] in baseline_errors]

    return {
        "baseline_errors": len(baseline_errors),
        "cascades_caught": len(full_caught),
        "cascades_that_prevented_errors": len(prevented),
        "catch_rate": round(len(prevented) / max(len(baseline_errors), 1), 4),
    }


def compute_skill_analysis(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Analyze SkillStore effectiveness."""
    full_tasks = runs.get("full", {}).get("tasks", [])
    total_steps = sum(t.get("steps", 0) for t in full_tasks)
    total_skill_hits = sum(t.get("skill_hits", 0) for t in full_tasks)
    total_llm = sum(t.get("llm_calls", 0) for t in full_tasks)

    baseline_tasks = runs.get("baseline", {}).get("tasks", [])
    baseline_llm = sum(t.get("llm_calls", 0) for t in baseline_tasks)

    return {
        "total_steps": total_steps,
        "skill_hits": total_skill_hits,
        "skill_hit_rate": round(total_skill_hits / max(total_steps, 1), 4),
        "llm_calls_cerebellum": total_llm,
        "llm_calls_baseline": baseline_llm,
        "llm_savings": round(1 - total_llm / max(baseline_llm, 1), 4),
    }


# ======================================================================
# Output formatters
# ======================================================================

def format_ablation_table(rows: list[dict[str, Any]], fmt: str = "text") -> str:
    if fmt == "latex":
        lines = [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Config & Tasks & Pass\% & Avg Steps & Avg LLM & Skills & Cascades \\",
            r"\midrule",
        ]
        for r in rows:
            lines.append(
                f"  {r['config']} & {r['tasks']} & {r['pass_rate']*100:.1f}\\% & "
                f"{r['avg_steps']:.1f} & {r['avg_llm_calls']:.1f} & "
                f"{r['total_skill_hits']} & {r['total_cascades']} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    elif fmt == "csv":
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    else:
        header = (
            f"{'Config':<15} {'Tasks':>6} {'Pass%':>7} {'AvgSteps':>9} "
            f"{'AvgLLM':>7} {'Skills':>7} {'Cascade':>8} {'Time':>8}"
        )
        lines = [header, "-" * len(header)]
        for r in rows:
            lines.append(
                f"{r['config']:<15} {r['tasks']:>6} {r['pass_rate']*100:>6.1f}% "
                f"{r['avg_steps']:>9.1f} {r['avg_llm_calls']:>7.1f} "
                f"{r['total_skill_hits']:>7} {r['total_cascades']:>8} "
                f"{r['total_time_s']:>7.1f}s"
            )
        return "\n".join(lines)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="WebArena Benchmark Analysis")
    parser.add_argument("--results-dir", type=str, default="benchmarks/results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--export", type=str, help="Export analysis to JSON file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    runs = load_all_runs(results_dir)
    if not runs:
        print(f"No webarena_*.json files found in {results_dir}")
        sys.exit(1)

    print(f"Loaded {len(runs)} run(s): {', '.join(runs.keys())}\n")

    # Ablation table
    ablation = compute_ablation_table(runs)
    fmt = "latex" if args.latex else ("csv" if args.csv else "text")
    print("=== Ablation Table ===")
    print(format_ablation_table(ablation, fmt))

    # Per-task comparison (if baseline and full both exist)
    if "baseline" in runs and "full" in runs:
        comparison = compute_per_task_comparison(runs)
        improved = sum(1 for c in comparison if c["improved"])
        regressed = sum(1 for c in comparison if c["regressed"])
        print(f"\n=== Per-Task Comparison ===")
        print(f"  Improved: {improved}/{len(comparison)}")
        print(f"  Regressed: {regressed}/{len(comparison)}")
        print(f"  Unchanged: {len(comparison) - improved - regressed}/{len(comparison)}")

        cascade = compute_cascade_analysis(runs)
        print(f"\n=== Cascade Detection ===")
        for k, v in cascade.items():
            print(f"  {k}: {v}")

        skill = compute_skill_analysis(runs)
        print(f"\n=== SkillStore Effectiveness ===")
        for k, v in skill.items():
            print(f"  {k}: {v}")

    if args.export:
        export_data = {
            "ablation": ablation,
            "cascade": compute_cascade_analysis(runs) if "baseline" in runs and "full" in runs else {},
            "skill": compute_skill_analysis(runs) if "full" in runs else {},
        }
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\nExported to {export_path}")


if __name__ == "__main__":
    import sys
    main()
