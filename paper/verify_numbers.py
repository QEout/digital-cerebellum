"""Verify that all numbers cited in the paper match the actual data."""
import json

full = json.loads(open("benchmarks/results/full_results.json", encoding="utf-8").read())
cl = json.loads(open("experiments/results/closed_loop_results.json", encoding="utf-8").read())

r0 = full[0]
s = cl["summary"]

def check(label, ok):
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: {label}")

print("=== Paper claims vs actual data ===\n")
print("--- Table 1: Main Results (500 samples) ---")
check("Accuracy 98.2%", abs(r0["accuracy"] - 0.982) < 0.001)
check("Precision 97.4%", abs(r0["precision"] - 0.974) < 0.002)
check("Recall 100%", abs(r0["recall"] - 1.0) < 0.001)
check("F1 0.987", abs(r0["f1"] - 0.987) < 0.001)
check("Fast-path 56.8%", abs(r0["fast_path_ratio"] - 0.568) < 0.001)
check("Fast acc 98.2%", abs(r0.get("fast_path_accuracy", 0) - 0.982) < 0.002)
check("Fast lat 10.3ms", abs(r0.get("avg_fast_latency_ms", 0) - 10.3) < 0.5)
check("Slow lat 1501ms", abs(r0.get("avg_slow_latency_ms", 0) - 1501) < 5)
check("Speedup 146x", abs(r0.get("speedup", 0) - 146.2) < 1)
check("TP=341", r0["confusion_matrix"]["tp"] == 341)
check("TN=150", r0["confusion_matrix"]["tn"] == 150)
check("FP=9", r0["confusion_matrix"]["fp"] == 9)
check("FN=0", r0["confusion_matrix"]["fn"] == 0)

print("\n--- Table 2: Ablation (300 samples each) ---")
ablation_runs = [r for r in full if r["total"] == 300]
for r in ablation_runs:
    label = r["config_label"]
    print(f"  {label}: acc={r['accuracy']:.1%} f1={r['f1']:.3f} fast={r['fast_path_ratio']:.1%}")

print("\n--- Table 4: Closed-loop ---")
check("M2 fast-path 97%", abs(s["m2_fast_path_ratio"] - 0.97) < 0.01)
check("M3 CB-GT 100%", abs(s["m3_cb_gt_accuracy"] - 1.0) < 0.01)
check("M3 CB-LLM 100%", abs(s["m3_cb_llm_agreement"] - 1.0) < 0.01)
check("M4 fast 12.8ms", abs(s["m4_avg_fast_ms"] - 12.8) < 0.5)
check("M4 speedup 114.9x", abs(s["m4_speedup"] - 114.9) < 0.5)
check("M5 savings 46%", abs(s["m5_savings_rate"] - 0.46) < 0.02)
check("Overall 4/4", s["overall_pass"] == 4)

print("\nAll checks complete.")
