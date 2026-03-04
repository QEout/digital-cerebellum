"""
Run all benchmarks: full system + ablation study + baseline comparison.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --quick       # 100 samples, fast
    python -m benchmarks.run_all --ablation    # full ablation study
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from digital_cerebellum.main import CerebellumConfig
from benchmarks.dataset import generate_comprehensive_benchmark
from benchmarks.sequential_dataset import generate_sequential_benchmark
from benchmarks.runner import (
    AblationConfig,
    BenchmarkResult,
    BenchmarkRunner,
    compare_results,
)


def run_full_benchmark(cfg: CerebellumConfig, n: int = 500) -> BenchmarkResult:
    """Run the full system benchmark."""
    print("\n" + "=" * 70)
    print("  FULL SYSTEM BENCHMARK")
    print("=" * 70)

    dataset = generate_comprehensive_benchmark(n=n)
    dataset.save(f"benchmarks/results/dataset_{n}.json")
    print(f"  Dataset saved: benchmarks/results/dataset_{n}.json")

    runner = BenchmarkRunner(cfg=cfg, ablation=AblationConfig.full())
    return runner.run(dataset, warmup_ratio=0.4)


def run_ablation_study(cfg: CerebellumConfig, n: int = 300) -> list[BenchmarkResult]:
    """Run ablation study — toggle each component."""
    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)

    dataset = generate_comprehensive_benchmark(n=n, seed=123)
    results = []

    for ablation in AblationConfig.all_ablations():
        print(f"\n--- Running ablation: {ablation.label} ---")
        runner = BenchmarkRunner(cfg=cfg, ablation=ablation)
        result = runner.run(dataset, warmup_ratio=0.4, verbose=False)
        results.append(result)
        print(f"  {ablation.label}: acc={result.accuracy:.1%} "
              f"f1={result.f1:.3f} fast={result.fast_path_ratio:.1%}")

    print("\n" + "=" * 70)
    print("  ABLATION COMPARISON")
    print("=" * 70)
    print(compare_results(results))

    return results


def run_baseline_comparison(cfg: CerebellumConfig, n: int = 200) -> list[BenchmarkResult]:
    """
    Compare against baselines:
      1. Pure LLM (always slow path)
      2. Random routing (50/50)
      3. Our full system
    """
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON")
    print("=" * 70)

    dataset = generate_comprehensive_benchmark(n=n, seed=456)
    results = []

    # Baseline 1: Always LLM (threshold_high = 0, everything goes slow)
    print("\n--- Baseline: Pure LLM (always slow path) ---")
    cfg_llm = CerebellumConfig()
    cfg_llm.llm_model = cfg.llm_model
    cfg_llm.llm_api_key = cfg.llm_api_key
    cfg_llm.llm_base_url = cfg.llm_base_url
    cfg_llm.threshold_high = 1.0
    cfg_llm.threshold_low = 1.0
    ablation_llm = AblationConfig(label="pure_llm")
    runner_llm = BenchmarkRunner(cfg=cfg_llm, ablation=ablation_llm)
    results.append(runner_llm.run(dataset, warmup_ratio=0.0, verbose=False))

    # Baseline 2: No learning (just random predictions)
    print("--- Baseline: No learning (random) ---")
    ablation_rand = AblationConfig(
        label="no_learning",
        replay_per_step=0, ewc_lambda=0.0,
        num_heads=1, mask_ratio=0.0,
    )
    runner_rand = BenchmarkRunner(cfg=cfg, ablation=ablation_rand)
    results.append(runner_rand.run(dataset, warmup_ratio=0.0, verbose=False))

    # Our system
    print("--- Our system (full) ---")
    runner_full = BenchmarkRunner(cfg=cfg, ablation=AblationConfig.full())
    results.append(runner_full.run(dataset, warmup_ratio=0.4, verbose=False))

    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON")
    print("=" * 70)
    print(compare_results(results))

    return results


def run_phase2_comparison(cfg: CerebellumConfig, n: int = 300) -> list[BenchmarkResult]:
    """Compare Phase 1 (baseline) vs Phase 2 components."""
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPARISON (static benchmark)")
    print("=" * 70)

    dataset = generate_comprehensive_benchmark(n=n, seed=789)
    results = []

    for ablation in AblationConfig.phase2_ablations():
        print(f"\n--- Running: {ablation.label} ---")
        runner = BenchmarkRunner(cfg=cfg, ablation=ablation)
        result = runner.run(dataset, warmup_ratio=0.4, verbose=False)
        results.append(result)
        print(f"  {ablation.label}: acc={result.accuracy:.1%} "
              f"f1={result.f1:.3f} fast={result.fast_path_ratio:.1%} "
              f"speedup={result.speedup:.1f}x")

    print("\n" + "=" * 70)
    print("  PHASE 2 COMPARISON (static)")
    print("=" * 70)
    print(compare_results(results))

    return results


def run_sequential_benchmark(cfg: CerebellumConfig) -> list[BenchmarkResult]:
    """Run sequential (temporal) benchmark — Phase 2 components shine here."""
    print("\n" + "=" * 70)
    print("  SEQUENTIAL BENCHMARK (temporal patterns)")
    print("=" * 70)

    dataset = generate_sequential_benchmark(n_sessions=20, seed=42)
    print(f"  Dataset: {len(dataset)} samples in temporal sessions")
    results = []

    configs = [
        AblationConfig.full(),
        AblationConfig.phase2_full(),
        AblationConfig.phase2_state_only(),
    ]

    for ablation in configs:
        print(f"\n--- Running: {ablation.label} ---")
        runner = BenchmarkRunner(cfg=cfg, ablation=ablation)
        result = runner.run(dataset, warmup_ratio=0.3, verbose=False)
        results.append(result)
        print(f"  {ablation.label}: acc={result.accuracy:.1%} "
              f"f1={result.f1:.3f} fast={result.fast_path_ratio:.1%} "
              f"speedup={result.speedup:.1f}x")

    print("\n" + "=" * 70)
    print("  SEQUENTIAL BENCHMARK COMPARISON")
    print("=" * 70)
    print(compare_results(results))

    return results


def run_phase3_benchmark(cfg: CerebellumConfig, n: int = 300) -> list[BenchmarkResult]:
    """
    Phase 3 benchmark — evaluate emergent cognitive properties.

    Tests whether somatic marker, curiosity drive, and self-model
    improve accuracy and learning efficiency over the baseline.
    """
    print("\n" + "=" * 70)
    print("  PHASE 3 BENCHMARK (emergent cognition)")
    print("=" * 70)

    dataset = generate_comprehensive_benchmark(n=n, seed=303)
    results = []

    for ablation in AblationConfig.phase3_ablations():
        print(f"\n--- Running: {ablation.label} ---")
        runner = BenchmarkRunner(cfg=cfg, ablation=ablation)
        result = runner.run(dataset, warmup_ratio=0.4, verbose=False)
        results.append(result)
        print(f"  {ablation.label}: acc={result.accuracy:.1%} "
              f"f1={result.f1:.3f} fast={result.fast_path_ratio:.1%} "
              f"speedup={result.speedup:.1f}x")

    print("\n" + "=" * 70)
    print("  PHASE 3 COMPARISON")
    print("=" * 70)
    print(compare_results(results))

    return results


def main():
    parser = argparse.ArgumentParser(description="Digital Cerebellum Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run (100 samples)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--baseline", action="store_true", help="Run baseline comparison")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 comparison")
    parser.add_argument("--phase3", action="store_true", help="Run Phase 3 comparison")
    parser.add_argument("--sequential", action="store_true", help="Run sequential benchmark")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    cfg = CerebellumConfig.from_yaml()
    n = 100 if args.quick else 500
    all_results = []

    if args.all or (not any([args.ablation, args.baseline, args.phase2,
                             args.phase3, args.sequential])):
        result = run_full_benchmark(cfg, n=n)
        all_results.append(result)

    if args.ablation or args.all:
        ablation_results = run_ablation_study(cfg, n=min(n, 300))
        all_results.extend(ablation_results)

    if args.baseline or args.all:
        baseline_results = run_baseline_comparison(cfg, n=min(n, 200))
        all_results.extend(baseline_results)

    if args.phase2 or args.all:
        phase2_results = run_phase2_comparison(cfg, n=min(n, 300))
        all_results.extend(phase2_results)

    if args.phase3 or args.all:
        phase3_results = run_phase3_benchmark(cfg, n=min(n, 300))
        all_results.extend(phase3_results)

    if args.sequential or args.all:
        seq_results = run_sequential_benchmark(cfg)
        all_results.extend(seq_results)

    if args.save:
        out = [r.to_dict(include_steps=True) for r in all_results]
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.save} (with step-level data)")


if __name__ == "__main__":
    main()
