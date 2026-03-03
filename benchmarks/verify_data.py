"""Quick verification that step-level data was saved correctly."""
import json

data = json.loads(open("benchmarks/results/full_results.json", encoding="utf-8").read())
print(f"Total experiment runs: {len(data)}\n")

for r in data:
    label = r["config_label"]
    name = r["name"]
    acc = r["accuracy"]
    n_steps = len(r.get("steps", []))
    print(f"  {label:20s} | {name:25s} | accuracy={acc:.1%} | steps={n_steps}")

print("\n--- Sample step (run 0, step 10) ---")
s = data[0]["steps"][10]
for k, v in s.items():
    print(f"  {k:18s}: {v}")

# Verify dataset
ds = json.loads(open("benchmarks/results/dataset_500.json", encoding="utf-8").read())
print(f"\n--- Dataset ---")
print(f"  Total samples: {len(ds['samples'])}")
print(f"  Fields per sample: {list(ds['samples'][0].keys())}")
safe_count = sum(1 for s in ds["samples"] if s["ground_truth_safe"])
print(f"  Safe/Unsafe: {safe_count}/{len(ds['samples'])-safe_count}")
