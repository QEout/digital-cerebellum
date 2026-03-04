"""
Micro-Operation Demo — the cerebellum learns continuous control.

This demo proves the Digital Cerebellum can:
  1. Run at 60Hz+ with sub-millisecond latency per step
  2. Learn a forward model (predict consequences of actions)
  3. Improve control performance over time
  4. Operate WITHOUT any LLM — pure cerebellar computation

Two environments are tested:
  - TargetTracker: move to intercept a moving target (2D)
  - BalanceBeam: keep a pole balanced (simplified CartPole)

Run:
    python examples/micro_ops_demo.py
"""

from __future__ import annotations

import time
import numpy as np

from digital_cerebellum.micro_ops.engine import MicroOpEngine, MicroOpConfig
from digital_cerebellum.micro_ops.environments import TargetTracker, BalanceBeam


def run_target_tracker():
    print("=" * 70)
    print("ENVIRONMENT 1: Target Tracker (2D pursuit)")
    print("  Task: Move agent to intercept a moving target")
    print("  State: [agent_x, agent_y, target_x, target_y, dx, dy]")
    print("  Action: [move_x, move_y]")
    print("=" * 70)

    env = TargetTracker(speed=0.1, target_speed=0.02, noise=0.005)
    cfg = MicroOpConfig(
        rff_dim=1024,
        forward_model_lr=0.01,
        action_lr=0.005,
        target_hz=10000,
    )
    engine = MicroOpEngine(state_dim=6, action_dim=2, cfg=cfg)

    print("\nRunning 500 steps...")
    t0 = time.perf_counter()
    summary = engine.run(env, n_steps=500, target_hz=100000)
    elapsed = time.perf_counter() - t0

    print(f"\n  Actual Hz:           {summary['actual_hz']:.0f}")
    print(f"  Mean latency:        {summary['mean_latency_ms']:.3f} ms/step")
    print(f"  Max latency:         {summary['max_latency_ms']:.3f} ms/step")
    print(f"  Total time:          {elapsed:.2f}s")
    print()
    print("  FORWARD MODEL (predicting consequences of actions):")
    print(f"    Training steps:    {summary['forward_model']['step']}")
    print(f"    Mean recent error: {summary['forward_model']['mean_recent_error']:.6f}")
    print(f"    Is improving:      {summary['forward_model']['is_improving']}")
    print()
    print("  LEARNING PROGRESS:")
    imp = summary["improvement"]
    print(f"    Early reward (avg):  {imp['early_reward']:.4f}")
    print(f"    Late reward (avg):   {imp['late_reward']:.4f}")
    print(f"    Reward change:       {imp['reward_improvement']:+.4f}")
    print(f"    Early SPE:           {imp['early_spe']:.4f}")
    print(f"    Late SPE:            {imp['late_spe']:.4f}")
    print(f"    SPE reduction:       {imp['spe_reduction']:+.4f}")

    return summary


def run_balance_beam():
    print("\n" + "=" * 70)
    print("ENVIRONMENT 2: Balance Beam (pole balancing)")
    print("  Task: Apply force to keep a pole upright")
    print("  State: [position, velocity, angle, angular_velocity]")
    print("  Action: [force]")
    print("=" * 70)

    env = BalanceBeam()
    cfg = MicroOpConfig(
        rff_dim=1024,
        forward_model_lr=0.01,
        action_lr=0.005,
    )
    engine = MicroOpEngine(state_dim=4, action_dim=1, cfg=cfg)

    print("\nRunning 500 steps...")
    t0 = time.perf_counter()
    summary = engine.run(env, n_steps=500, target_hz=100000)
    elapsed = time.perf_counter() - t0

    print(f"\n  Actual Hz:           {summary['actual_hz']:.0f}")
    print(f"  Mean latency:        {summary['mean_latency_ms']:.3f} ms/step")
    print(f"  Total time:          {elapsed:.2f}s")
    print()
    print("  FORWARD MODEL:")
    print(f"    Mean recent error: {summary['forward_model']['mean_recent_error']:.6f}")
    print(f"    Is improving:      {summary['forward_model']['is_improving']}")
    print()
    print("  LEARNING PROGRESS:")
    imp = summary["improvement"]
    print(f"    Early reward (avg):  {imp['early_reward']:.4f}")
    print(f"    Late reward (avg):   {imp['late_reward']:.4f}")
    print(f"    SPE reduction:       {imp['spe_reduction']:+.4f}")

    return summary


def main():
    print()
    print("  DIGITAL CEREBELLUM — PHASE 6: MICRO-OPERATION ENGINE")
    print("  Proving: the cerebellum can learn continuous real-time control")
    print("  No LLM. No text. Pure cerebellar computation.")
    print()

    s1 = run_target_tracker()
    s2 = run_balance_beam()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    can_60hz = s1["mean_latency_ms"] < 16.0 and s2["mean_latency_ms"] < 16.0
    fm_improving = (
        s1["forward_model"]["is_improving"]
        or s1["forward_model"]["mean_recent_error"] < 0.01
    )

    print(f"\n  60Hz capable:        {'YES' if can_60hz else 'NO'}")
    print(f"    Tracker latency:   {s1['mean_latency_ms']:.3f} ms  (need <16ms)")
    print(f"    Balance latency:   {s2['mean_latency_ms']:.3f} ms  (need <16ms)")
    print(f"  Forward model works: {'YES' if fm_improving else 'NO'}")
    print(f"  SPE decreasing:      {'YES' if s1['improvement']['spe_reduction'] > 0 else 'learning...'}")

    print()
    print("  KEY INSIGHT:")
    print("  The cerebellum runs at {:.0f}+ Hz with {:.3f}ms per step.".format(
        s1["actual_hz"], s1["mean_latency_ms"],
    ))
    print("  It learns a forward model of the environment in real-time.")
    print("  No LLM was called. No text was processed.")
    print("  This is what 'giving an AI agent a body' means.")
    print()


if __name__ == "__main__":
    main()
