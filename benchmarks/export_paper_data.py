"""
Export benchmark results into paper-ready formats:
  - CSV files for plotting (matplotlib / pgfplots)
  - LaTeX tables for direct inclusion
  - Summary statistics as JSON
"""
import csv
import json
import os
from pathlib import Path

OUT_DIR = Path("benchmarks/results/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = json.loads(
    Path("benchmarks/results/full_results.json").read_text(encoding="utf-8")
)

# ── 1. Step-level CSV (for learning curves, scatter plots) ──────────────
full_run = data[0]  # full config, 500 samples
csv_path = OUT_DIR / "step_level.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "step", "sample_id", "tool_name", "ground_truth", "predicted_safe",
        "safety_score", "confidence", "route", "latency_ms", "correct", "difficulty",
    ])
    for i, s in enumerate(full_run["steps"]):
        w.writerow([
            i, s["sample_id"], s["tool_name"], s["ground_truth"],
            s["predicted_safe"], round(s["safety_score"], 4),
            round(s["confidence"], 4), s["route"],
            round(s["latency_ms"], 2), s["correct"], s["difficulty"],
        ])
print(f"[1/5] Step-level CSV: {csv_path}  ({len(full_run['steps'])} rows)")

# ── 2. Learning curve CSV (rolling accuracy every 10 steps) ─────────────
lc_path = OUT_DIR / "learning_curve.csv"
steps = full_run["steps"]
window = 20
with open(lc_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step", "rolling_accuracy", "rolling_fast_ratio", "rolling_confidence"])
    for i in range(window, len(steps) + 1):
        chunk = steps[i - window:i]
        acc = sum(1 for s in chunk if s["correct"]) / len(chunk)
        fast = sum(1 for s in chunk if s["route"] == "fast") / len(chunk)
        conf = sum(s["confidence"] for s in chunk) / len(chunk)
        w.writerow([i, round(acc, 4), round(fast, 4), round(conf, 4)])
print(f"[2/5] Learning curve CSV: {lc_path}")

# ── 3. Ablation comparison CSV ──────────────────────────────────────────
ablation_runs = [r for r in data if r["total"] == 300]
abl_path = OUT_DIR / "ablation.csv"
with open(abl_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "config", "accuracy", "precision", "recall", "f1",
        "fast_path_ratio", "fast_path_accuracy", "speedup",
        "avg_fast_ms", "avg_slow_ms",
    ])
    for r in ablation_runs:
        w.writerow([
            r["config_label"],
            round(r["accuracy"], 4), round(r["precision"], 4),
            round(r["recall"], 4), round(r["f1"], 4),
            round(r["fast_path_ratio"], 4),
            round(r.get("fast_path_accuracy", 0), 4),
            round(r.get("speedup", 0), 1),
            round(r.get("avg_fast_latency_ms", 0), 1),
            round(r.get("avg_slow_latency_ms", 0), 1),
        ])
print(f"[3/5] Ablation CSV: {abl_path}  ({len(ablation_runs)} configs)")

# ── 4. Baseline comparison CSV ──────────────────────────────────────────
baseline_runs = [r for r in data if r["total"] == 200]
base_path = OUT_DIR / "baseline.csv"
with open(base_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "config", "accuracy", "f1", "fast_path_ratio",
        "fast_path_accuracy", "speedup", "avg_slow_ms",
    ])
    for r in baseline_runs:
        w.writerow([
            r["config_label"],
            round(r["accuracy"], 4), round(r["f1"], 4),
            round(r["fast_path_ratio"], 4),
            round(r.get("fast_path_accuracy", 0), 4),
            round(r.get("speedup", 0), 1),
            round(r.get("avg_slow_latency_ms", 0), 1),
        ])
print(f"[4/5] Baseline CSV: {base_path}  ({len(baseline_runs)} configs)")

# ── 5. LaTeX tables ─────────────────────────────────────────────────────
tex_path = OUT_DIR / "tables.tex"
lines = []

# Table 1: Main results
lines.append(r"% Table 1: Main System Results (500 samples)")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Main evaluation results on ToolCallBench-500.}")
lines.append(r"\label{tab:main_results}")
lines.append(r"\begin{tabular}{lc}")
lines.append(r"\toprule")
lines.append(r"Metric & Value \\")
lines.append(r"\midrule")
r0 = full_run
lines.append(f"Accuracy & {r0['accuracy']:.1%} \\\\")
lines.append(f"Precision & {r0['precision']:.1%} \\\\")
lines.append(f"Recall & {r0['recall']:.1%} \\\\")
lines.append(f"F1 Score & {r0['f1']:.3f} \\\\")
lines.append(r"\midrule")
lines.append(f"Fast-path ratio & {r0['fast_path_ratio']:.1%} \\\\")
lines.append(f"Fast-path accuracy & {r0.get('fast_path_accuracy', 0):.1%} \\\\")
lines.append(f"Avg fast latency & {r0.get('avg_fast_latency_ms', 0):.1f}ms \\\\")
lines.append(f"Avg slow latency & {r0.get('avg_slow_latency_ms', 0):.1f}ms \\\\")
lines.append(f"Speedup & {r0.get('speedup', 0):.1f}$\\times$ \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
lines.append("")

# Table 2: Ablation
lines.append(r"% Table 2: Ablation Study")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Ablation study: contribution of each biologically-inspired component.}")
lines.append(r"\label{tab:ablation}")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(r"Configuration & Acc. & F1 & Fast\% & FastAcc & Speedup \\")
lines.append(r"\midrule")

config_labels = {
    "full": "Full system",
    "no_mask": "w/o Dendritic Mask",
    "no_replay": "w/o Experience Replay",
    "no_ewc": "w/o EWC",
    "single_head": "Single Head (K=1)",
    "no_task_heads": "w/o Task Heads",
}
for r in ablation_runs:
    label = config_labels.get(r["config_label"], r["config_label"])
    lines.append(
        f"{label} & {r['accuracy']:.1%} & {r['f1']:.3f} "
        f"& {r['fast_path_ratio']:.1%} & {r.get('fast_path_accuracy', 0):.1%} "
        f"& {r.get('speedup', 0):.1f}$\\times$ \\\\"
    )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
lines.append("")

# Table 3: Baseline
lines.append(r"% Table 3: Baseline Comparison")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Comparison with baseline approaches.}")
lines.append(r"\label{tab:baseline}")
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r"Approach & Accuracy & F1 & Speedup & Avg Latency \\")
lines.append(r"\midrule")
baseline_labels = {
    "pure_llm": "Pure LLM",
    "no_learning": "LLM (no learning)",
    "full": "Digital Cerebellum",
}
for r in baseline_runs:
    label = baseline_labels.get(r["config_label"], r["config_label"])
    avg_lat = (
        r.get("avg_fast_latency_ms", 0) * r["fast_path_ratio"]
        + r.get("avg_slow_latency_ms", 0) * (1 - r["fast_path_ratio"])
    )
    lines.append(
        f"{label} & {r['accuracy']:.1%} & {r['f1']:.3f} "
        f"& {r.get('speedup', 0):.1f}$\\times$ & {avg_lat:.0f}ms \\\\"
    )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

Path(tex_path).write_text("\n".join(lines), encoding="utf-8")
print(f"[5/5] LaTeX tables: {tex_path}")

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\nAll paper data exported to {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name:25s}  {size:>8,} bytes")
