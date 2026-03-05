#!/usr/bin/env python3
"""
Tank Battle Benchmark — quantitative validation of cortex-cerebellum collaboration.

Runs the TankBattleEnv in two modes:
  1. cortex+cerebellum: GUIController with cerebellar correction enabled
  2. cortex_only:       Same controller with correction disabled (ablation)

Measures across multiple rounds:
  - SPE convergence (forward model learning)
  - Hit rate improvement (aiming accuracy)
  - Reward trajectory (overall performance)
  - Cerebellum confidence growth
  - Effective LLM call reduction (adaptive interval)

Usage:
    python benchmarks/tank_benchmark.py
    python benchmarks/tank_benchmark.py --rounds 15 --json
    python benchmarks/tank_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

from digital_cerebellum.micro_ops.tank_env import (
    TankBattleEnv,
    TankConfig,
    TankController,
)
from digital_cerebellum.micro_ops.gui_controller import GUIControlConfig


@dataclass
class RoundMetrics:
    round_num: int
    kills: int = 0
    shots_fired: int = 0
    shots_hit: int = 0
    hit_rate: float = 0.0
    total_reward: float = 0.0
    mean_spe: float = 0.0
    mean_correction_mag: float = 0.0
    confidence: float = 0.0
    adaptive_llm_interval: int = 90
    total_score: float = 0.0
    grade: str = "D"
    survive_ticks: int = 0


@dataclass
class BenchmarkResult:
    mode: str
    rounds: list[RoundMetrics] = field(default_factory=list)

    @property
    def mean_hit_rate_early(self) -> float:
        early = self.rounds[:3]
        return np.mean([r.hit_rate for r in early]) if early else 0.0

    @property
    def mean_hit_rate_late(self) -> float:
        late = self.rounds[-3:]
        return np.mean([r.hit_rate for r in late]) if late else 0.0

    @property
    def mean_spe_early(self) -> float:
        early = self.rounds[:3]
        return np.mean([r.mean_spe for r in early]) if early else 0.0

    @property
    def mean_spe_late(self) -> float:
        late = self.rounds[-3:]
        return np.mean([r.mean_spe for r in late]) if late else 0.0

    @property
    def total_kills(self) -> int:
        return sum(r.kills for r in self.rounds)

    @property
    def total_reward(self) -> float:
        return sum(r.total_reward for r in self.rounds)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "rounds": len(self.rounds),
            "total_kills": self.total_kills,
            "total_reward": round(self.total_reward, 1),
            "hit_rate_early": round(self.mean_hit_rate_early, 4),
            "hit_rate_late": round(self.mean_hit_rate_late, 4),
            "hit_rate_improvement": round(self.mean_hit_rate_late - self.mean_hit_rate_early, 4),
            "spe_early": round(self.mean_spe_early, 4),
            "spe_late": round(self.mean_spe_late, 4),
            "spe_reduction": round(
                (1 - self.mean_spe_late / max(self.mean_spe_early, 1e-6)) * 100, 1
            ),
            "final_confidence": round(self.rounds[-1].confidence, 4) if self.rounds else 0.0,
            "final_llm_interval": self.rounds[-1].adaptive_llm_interval if self.rounds else 90,
        }


TANK_CFG = TankConfig(
    round_max_ticks=600,
    enemy_count=3,
    noise=0.05,
    randomize_spawns=False,
)

CTRL_CFG = GUIControlConfig(
    cortex_gain=0.6,
    cortex_noise=0.8,
    correction_lr=0.002,
    correction_hidden=64,
    forward_model_lr=0.02,
    forward_model_hidden=128,
    noise_decay=0.90,
    correction_scale=0.15,
    alignment_weight=0.3,
)


def run_mode(mode: str, n_rounds: int, verbose: bool = False) -> BenchmarkResult:
    cerebellum_on = mode == "cortex+cerebellum"

    env = TankBattleEnv(TANK_CFG)
    ctrl = TankController(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=CTRL_CFG,
        cerebellum_enabled=cerebellum_on,
    )

    result = BenchmarkResult(mode=mode)

    for rnd in range(1, n_rounds + 1):
        env.reset()
        round_reward = 0.0
        round_spes: list[float] = []
        round_corrections: list[float] = []

        while not env.done:
            step_result = ctrl.step(env)
            round_reward += step_result.reward
            round_spes.append(step_result.spe)
            round_corrections.append(float(np.linalg.norm(step_result.correction)))

        ctrl.decay_noise()
        score = env.get_round_score()

        metrics = RoundMetrics(
            round_num=rnd,
            kills=score["kills"],
            shots_fired=score["shots_fired"],
            shots_hit=score["shots_hit"],
            hit_rate=score["hit_rate"],
            total_reward=round_reward,
            mean_spe=float(np.mean(round_spes)) if round_spes else 0.0,
            mean_correction_mag=float(np.mean(round_corrections)) if round_corrections else 0.0,
            confidence=ctrl.cerebellum_confidence,
            adaptive_llm_interval=ctrl.should_call_cortex(90),
            total_score=score["total_score"],
            grade=score["grade"],
            survive_ticks=score["time_ticks"],
        )
        result.rounds.append(metrics)

        if verbose:
            tag = "CB" if cerebellum_on else "CX"
            print(
                f"  [{tag}] Round {rnd:2d}: "
                f"kills={metrics.kills}  "
                f"hit={metrics.hit_rate*100:4.1f}%  "
                f"SPE={metrics.mean_spe:.4f}  "
                f"corr={metrics.mean_correction_mag:.4f}  "
                f"conf={metrics.confidence:.3f}  "
                f"score={metrics.total_score:.0f}({metrics.grade})"
            )

    return result


def main():
    parser = argparse.ArgumentParser(description="Tank Battle Benchmark")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per mode")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  TANK BATTLE BENCHMARK -- Cortex vs Cortex+Cerebellum")
    print("=" * 65)
    print()

    results = {}
    for mode in ("cortex+cerebellum", "cortex_only"):
        label = "Cortex + Cerebellum" if "cerebellum" in mode else "Cortex Only (ablation)"
        print(f"  -- {label} ({args.rounds} rounds) --")
        t0 = time.perf_counter()
        res = run_mode(mode, args.rounds, verbose=args.verbose)
        elapsed = time.perf_counter() - t0
        results[mode] = res
        s = res.summary()
        print(f"     Kills: {s['total_kills']}  "
              f"Reward: {s['total_reward']}  "
              f"Hit rate: {s['hit_rate_early']*100:.1f}% -> {s['hit_rate_late']*100:.1f}%  "
              f"SPE: {s['spe_early']:.4f} -> {s['spe_late']:.4f} ({s['spe_reduction']:.0f}% down)  "
              f"[{elapsed:.1f}s]")
        print()

    cb = results["cortex+cerebellum"].summary()
    cx = results["cortex_only"].summary()

    print("  -- Comparison --")
    print(f"     Hit rate improvement:    CB={cb['hit_rate_improvement']*100:+.1f}%  "
          f"CX={cx['hit_rate_improvement']*100:+.1f}%")
    print(f"     SPE reduction:           CB={cb['spe_reduction']:.0f}%  "
          f"CX={cx['spe_reduction']:.0f}%")
    print(f"     Total reward:            CB={cb['total_reward']:.0f}  "
          f"CX={cx['total_reward']:.0f}  "
          f"(delta={cb['total_reward'] - cx['total_reward']:+.0f})")
    print(f"     Total kills:             CB={cb['total_kills']}  CX={cx['total_kills']}")
    print(f"     Final CB confidence:     {cb['final_confidence']:.4f}")
    print(f"     Final LLM interval:      {cb['final_llm_interval']} ticks "
          f"(base=90, {cb['final_llm_interval']/90:.1f}x reduction)")
    print()

    if cb["total_reward"] > cx["total_reward"]:
        print("  [PASS] CEREBELLUM HELPS: higher total reward than cortex-only")
    else:
        print("  [WARN] CEREBELLUM DID NOT HELP in this run")

    if cb["spe_reduction"] > 10:
        print(f"  [PASS] FORWARD MODEL LEARNED: {cb['spe_reduction']:.0f}% SPE reduction")
    else:
        print(f"  [WARN] LIMITED LEARNING: only {cb['spe_reduction']:.0f}% SPE reduction")

    if cb["hit_rate_late"] > cb["hit_rate_early"]:
        print(f"  [PASS] ACCURACY IMPROVED: "
              f"{cb['hit_rate_early']*100:.1f}% -> {cb['hit_rate_late']*100:.1f}%")
    print()

    if args.json:
        output = {
            "cortex_cerebellum": cb,
            "cortex_only": cx,
            "per_round": {
                "cortex_cerebellum": [
                    {
                        "round": m.round_num,
                        "kills": m.kills,
                        "hit_rate": round(m.hit_rate, 4),
                        "mean_spe": round(m.mean_spe, 4),
                        "confidence": round(m.confidence, 4),
                        "total_score": round(m.total_score, 1),
                        "grade": m.grade,
                    }
                    for m in results["cortex+cerebellum"].rounds
                ],
                "cortex_only": [
                    {
                        "round": m.round_num,
                        "kills": m.kills,
                        "hit_rate": round(m.hit_rate, 4),
                        "mean_spe": round(m.mean_spe, 4),
                        "total_score": round(m.total_score, 1),
                        "grade": m.grade,
                    }
                    for m in results["cortex_only"].rounds
                ],
            },
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
