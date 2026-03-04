"""
Multi-model comparison: same benchmark, different LLM cortex.

Proves that Digital Cerebellum is model-agnostic — it can distill
safety knowledge from any LLM and still achieve strong performance.

Usage:
    python -m benchmarks.run_multi_model
"""

from __future__ import annotations

import json
from pathlib import Path

from digital_cerebellum.main import CerebellumConfig
from benchmarks.dataset import generate_comprehensive_benchmark
from benchmarks.runner import AblationConfig, BenchmarkRunner, compare_results

MODELS = [
    {
        "label": "Qwen-3.5-Flash",
        "model": "qwen3.5-flash",
        "api_key": None,  # loaded from config.yaml
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    {
        "label": "DeepSeek-V3",
        "model": "deepseek-chat",
        "api_key": "sk-81b6b2e8fbca45e2aaf067f80258de2d",
        "base_url": "https://api.deepseek.com/v1",
    },
]


def main():
    base_cfg = CerebellumConfig.from_yaml()
    dataset = generate_comprehensive_benchmark(n=200, seed=999)
    results = []

    print("=" * 70)
    print("  MULTI-MODEL COMPARISON")
    print("  Same 200-sample benchmark, different LLM cortex")
    print("=" * 70)

    for m in MODELS:
        print(f"\n--- {m['label']} ---")
        cfg = CerebellumConfig()
        cfg.llm_model = m["model"]
        cfg.llm_api_key = m["api_key"] or base_cfg.llm_api_key
        cfg.llm_base_url = m["base_url"]
        cfg.threshold_high = 0.85
        cfg.threshold_low = 0.4

        ablation = AblationConfig(label=m["label"])
        runner = BenchmarkRunner(cfg=cfg, ablation=ablation)
        result = runner.run(dataset, warmup_ratio=0.4, verbose=True)
        results.append(result)

        print(f"\n  {m['label']}: acc={result.accuracy:.1%} "
              f"f1={result.f1:.3f} fast={result.fast_path_ratio:.1%} "
              f"speedup={result.speedup:.1f}x")

    print("\n" + "=" * 70)
    print("  MULTI-MODEL COMPARISON")
    print("=" * 70)
    print(compare_results(results))

    out = [r.to_dict(include_steps=True) for r in results]
    out_path = Path("benchmarks/results/multi_model_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
