"""
Benchmark Runner — systematic evaluation framework.

Runs a dataset through the Digital Cerebellum and collects metrics:
  - Accuracy (safety classification)
  - Fast-path ratio (LLM call savings)
  - Latency (fast vs slow path)
  - Learning curve (accuracy over time)
  - Per-difficulty breakdown
  - Confusion matrix

Supports ablation by toggling engine components.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from benchmarks.dataset import BenchmarkDataset, ToolCallSample

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    sample_id: str
    tool_name: str
    ground_truth: bool | None
    predicted_safe: bool
    safety_score: float
    confidence: float
    route: str
    latency_ms: float
    correct: bool | None
    difficulty: str


@dataclass
class BenchmarkResult:
    """Full results of a benchmark run."""
    name: str
    config_label: str
    steps: list[StepResult] = field(default_factory=list)
    total_time_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    # -- Computed metrics --
    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def labelled(self) -> list[StepResult]:
        return [s for s in self.steps if s.ground_truth is not None]

    @property
    def accuracy(self) -> float:
        lab = self.labelled
        if not lab:
            return 0.0
        return sum(1 for s in lab if s.correct) / len(lab)

    @property
    def fast_path_ratio(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.route == "fast") / len(self.steps)

    @property
    def fast_path_accuracy(self) -> float:
        fast = [s for s in self.labelled if s.route == "fast"]
        if not fast:
            return 0.0
        return sum(1 for s in fast if s.correct) / len(fast)

    @property
    def slow_path_accuracy(self) -> float:
        slow = [s for s in self.labelled if s.route != "fast"]
        if not slow:
            return 0.0
        return sum(1 for s in slow if s.correct) / len(slow)

    @property
    def avg_fast_latency_ms(self) -> float:
        fast = [s.latency_ms for s in self.steps if s.route == "fast"]
        return float(np.mean(fast)) if fast else 0.0

    @property
    def avg_slow_latency_ms(self) -> float:
        slow = [s.latency_ms for s in self.steps if s.route != "fast"]
        return float(np.mean(slow)) if slow else 0.0

    @property
    def speedup(self) -> float:
        if self.avg_fast_latency_ms == 0:
            return 0.0
        return self.avg_slow_latency_ms / self.avg_fast_latency_ms

    @property
    def confusion_matrix(self) -> dict[str, int]:
        tp = sum(1 for s in self.labelled if s.ground_truth and s.predicted_safe)
        tn = sum(1 for s in self.labelled if not s.ground_truth and not s.predicted_safe)
        fp = sum(1 for s in self.labelled if not s.ground_truth and s.predicted_safe)
        fn = sum(1 for s in self.labelled if s.ground_truth and not s.predicted_safe)
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @property
    def precision(self) -> float:
        cm = self.confusion_matrix
        denom = cm["tp"] + cm["fp"]
        return cm["tp"] / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        cm = self.confusion_matrix
        denom = cm["tp"] + cm["fn"]
        return cm["tp"] / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def accuracy_by_difficulty(self) -> dict[str, float]:
        by_diff: dict[str, list[bool]] = {}
        for s in self.labelled:
            by_diff.setdefault(s.difficulty, []).append(s.correct)
        return {k: sum(v) / len(v) for k, v in by_diff.items()}

    def learning_curve(self, window: int = 20) -> list[float]:
        """Sliding-window accuracy over time."""
        if len(self.labelled) < window:
            return [self.accuracy]
        correct = [s.correct for s in self.labelled]
        curve = []
        for i in range(window, len(correct) + 1):
            w = correct[i - window:i]
            curve.append(sum(w) / len(w))
        return curve

    def summary(self) -> str:
        cm = self.confusion_matrix
        lines = [
            f"{'='*60}",
            f"  Benchmark: {self.name}  |  Config: {self.config_label}",
            f"{'='*60}",
            f"  Total samples:      {self.total}",
            f"  Labelled:           {len(self.labelled)}",
            f"",
            f"  Accuracy:           {self.accuracy:.1%}",
            f"  Precision:          {self.precision:.1%}",
            f"  Recall:             {self.recall:.1%}",
            f"  F1:                 {self.f1:.1%}",
            f"",
            f"  Fast-path ratio:    {self.fast_path_ratio:.1%}",
            f"  Fast-path accuracy: {self.fast_path_accuracy:.1%}",
            f"  Slow-path accuracy: {self.slow_path_accuracy:.1%}",
            f"",
            f"  Avg fast latency:   {self.avg_fast_latency_ms:.1f}ms",
            f"  Avg slow latency:   {self.avg_slow_latency_ms:.1f}ms",
            f"  Speedup:            {self.speedup:.1f}x",
            f"",
            f"  Confusion matrix:   TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}",
            f"  By difficulty:      {self.accuracy_by_difficulty()}",
            f"  Total time:         {self.total_time_s:.1f}s",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def to_dict(self, include_steps: bool = True) -> dict[str, Any]:
        d = {
            "name": self.name, "config_label": self.config_label,
            "total": self.total, "accuracy": self.accuracy,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "fast_path_ratio": self.fast_path_ratio,
            "fast_path_accuracy": self.fast_path_accuracy,
            "avg_fast_latency_ms": self.avg_fast_latency_ms,
            "avg_slow_latency_ms": self.avg_slow_latency_ms,
            "speedup": self.speedup,
            "confusion_matrix": self.confusion_matrix,
            "by_difficulty": self.accuracy_by_difficulty(),
            "learning_curve": self.learning_curve(),
            "total_time_s": self.total_time_s,
            "config": self.config,
        }
        if include_steps:
            d["steps"] = [
                {
                    "sample_id": s.sample_id, "tool_name": s.tool_name,
                    "ground_truth": s.ground_truth, "predicted_safe": s.predicted_safe,
                    "safety_score": s.safety_score, "confidence": s.confidence,
                    "route": s.route, "latency_ms": s.latency_ms,
                    "correct": s.correct, "difficulty": s.difficulty,
                }
                for s in self.steps
            ]
        return d


# ======================================================================
# Ablation configs
# ======================================================================

@dataclass
class AblationConfig:
    """Toggle components for ablation study."""
    label: str = "full"
    num_heads: int = 4
    mask_ratio: float = 0.5
    replay_per_step: int = 4
    replay_size: int = 64
    ewc_lambda: float = 400.0
    use_task_heads: bool = True

    @staticmethod
    def full() -> "AblationConfig":
        return AblationConfig(label="full")

    @staticmethod
    def no_dendritic_mask() -> "AblationConfig":
        return AblationConfig(label="no_mask", mask_ratio=0.0)

    @staticmethod
    def no_replay() -> "AblationConfig":
        return AblationConfig(label="no_replay", replay_per_step=0)

    @staticmethod
    def no_ewc() -> "AblationConfig":
        return AblationConfig(label="no_ewc", ewc_lambda=0.0)

    @staticmethod
    def single_head() -> "AblationConfig":
        return AblationConfig(label="single_head", num_heads=1)

    @staticmethod
    def no_task_heads() -> "AblationConfig":
        return AblationConfig(label="no_task_heads", use_task_heads=False)

    @classmethod
    def all_ablations(cls) -> list["AblationConfig"]:
        return [
            cls.full(),
            cls.no_dendritic_mask(),
            cls.no_replay(),
            cls.no_ewc(),
            cls.single_head(),
            cls.no_task_heads(),
        ]


# ======================================================================
# Runner
# ======================================================================

class BenchmarkRunner:
    """
    Runs a benchmark dataset through the cerebellum.

    Phase A: warm-up (all slow path, forced learning)
    Phase B: evaluation (natural routing)
    """

    def __init__(
        self,
        cfg: CerebellumConfig | None = None,
        ablation: AblationConfig | None = None,
    ):
        self.base_cfg = cfg or CerebellumConfig.from_yaml()
        self.ablation = ablation or AblationConfig.full()

    def _build_cerebellum(self) -> DigitalCerebellum:
        cfg = CerebellumConfig()
        cfg.llm_model = self.base_cfg.llm_model
        cfg.llm_api_key = self.base_cfg.llm_api_key
        cfg.llm_base_url = self.base_cfg.llm_base_url
        cfg.num_heads = self.ablation.num_heads
        cfg.ewc_lambda = self.ablation.ewc_lambda
        cfg.threshold_high = 0.85
        cfg.threshold_low = 0.4

        cb = DigitalCerebellum(cfg)
        cb.register_microzone(ToolCallMicrozone())

        if self.ablation.mask_ratio == 0.0:
            import torch
            with torch.no_grad():
                for head in cb.engine.heads:
                    head.feature_mask.fill_(1.0)
                    head._scale = 1.0

        cb.learner._replay_per_step = self.ablation.replay_per_step
        cb.learner._replay_size = self.ablation.replay_size

        return cb

    def run(
        self,
        dataset: BenchmarkDataset,
        warmup_ratio: float = 0.4,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run the full benchmark."""
        cb = self._build_cerebellum()
        result = BenchmarkResult(
            name=dataset.name,
            config_label=self.ablation.label,
            config={"ablation": self.ablation.label},
        )

        n_warmup = int(len(dataset) * warmup_ratio)
        warmup = dataset.samples[:n_warmup]
        test = dataset.samples[n_warmup:]

        t0 = time.perf_counter()

        # -- Warm-up phase (forced slow path) --
        if verbose:
            print(f"\n  Warm-up: {len(warmup)} samples (forced slow path)")

        saved_hi = cb.router.threshold_high
        cb.router.threshold_high = 1.0

        for i, sample in enumerate(warmup):
            step = self._evaluate_one(cb, sample)
            result.steps.append(step)
            if verbose and (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(warmup)}] warmup done")

        cb.router.threshold_high = 0.75

        # -- Test phase (natural routing) --
        if verbose:
            print(f"  Test: {len(test)} samples (natural routing)")

        for i, sample in enumerate(test):
            step = self._evaluate_one(cb, sample)
            result.steps.append(step)
            if verbose and (i + 1) % 50 == 0:
                recent = result.steps[-50:]
                acc = sum(1 for s in recent if s.correct) / len(recent) if recent else 0
                fast = sum(1 for s in recent if s.route == "fast") / len(recent) if recent else 0
                print(f"    [{i+1}/{len(test)}] recent_acc={acc:.1%} fast={fast:.1%}")

        result.total_time_s = time.perf_counter() - t0

        if verbose:
            print(result.summary())

        return result

    def _evaluate_one(self, cb: DigitalCerebellum, sample: ToolCallSample) -> StepResult:
        t0 = time.perf_counter()

        eval_result = cb.evaluate("tool_call", {
            "tool_name": sample.tool_name,
            "tool_params": sample.tool_params,
        }, context=sample.context)

        latency = (time.perf_counter() - t0) * 1000

        predicted_safe = eval_result.get("safe", True)
        safety_score = eval_result.get("safety_score", 0.5)
        confidence = eval_result.get("confidence", 0.0)
        route = "fast" if latency < 200 else "slow"

        correct = None
        if sample.ground_truth_safe is not None:
            correct = (predicted_safe == sample.ground_truth_safe)

        return StepResult(
            sample_id=sample.id,
            tool_name=sample.tool_name,
            ground_truth=sample.ground_truth_safe,
            predicted_safe=predicted_safe,
            safety_score=safety_score,
            confidence=confidence,
            route=route,
            latency_ms=latency,
            correct=correct,
            difficulty=sample.difficulty,
        )


# ======================================================================
# Comparison table
# ======================================================================

def compare_results(results: list[BenchmarkResult]) -> str:
    """Print a comparison table of multiple benchmark runs."""
    header = (
        f"{'Config':<20} {'Accuracy':>8} {'F1':>6} {'Fast%':>6} "
        f"{'FastAcc':>7} {'Fast ms':>7} {'Slow ms':>7} {'Speed':>6} {'Time':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.config_label:<20} {r.accuracy:>7.1%} {r.f1:>6.3f} "
            f"{r.fast_path_ratio:>5.1%} {r.fast_path_accuracy:>6.1%} "
            f"{r.avg_fast_latency_ms:>6.0f}ms {r.avg_slow_latency_ms:>6.0f}ms "
            f"{r.speedup:>5.1f}x {r.total_time_s:>5.0f}s"
        )
    return "\n".join(lines)
