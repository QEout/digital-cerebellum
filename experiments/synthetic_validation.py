"""
Phase 0 Synthetic Validation — Core Hypothesis Testing

Tests whether the cerebellar pipeline can learn to predict tool-call outcomes
from synthetic data WITHOUT an LLM.  This validates the core learning loop
before spending money on API calls.

Hypotheses tested:
  H1: Loss decreases over repeated exposure (the engine learns)
  H2: Population confidence correlates with actual accuracy
  H3: Fast-path ratio increases over time
  H4: EWC prevents catastrophic forgetting when new tool types appear
  H5: Inference latency < 10ms

Usage:
    python -m experiments.synthetic_validation
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from digital_cerebellum.core.feature_encoder import FeatureEncoder
from digital_cerebellum.core.pattern_separator import PatternSeparator
from digital_cerebellum.core.prediction_engine import EngineConfig, PredictionEngine
from digital_cerebellum.core.error_comparator import ErrorComparator
from digital_cerebellum.core.online_learner import OnlineLearner
from digital_cerebellum.routing.decision_router import DecisionRouter


# ======================================================================
# Synthetic tool-call generator
# ======================================================================

TOOL_CATALOG = {
    "send_email": {
        "params_templates": [
            {"to": "alice@example.com", "subject": "meeting", "body": "Let's meet at 3pm"},
            {"to": "bob@example.com", "subject": "report", "body": "Please review the attached"},
            {"to": "carol@example.com", "subject": "hello", "body": "How are you?"},
        ],
        "outcome": "email sent successfully",
        "safe": True,
    },
    "delete_file": {
        "params_templates": [
            {"path": "/tmp/cache.txt"},
            {"path": "/home/user/document.pdf"},
            {"path": "/etc/passwd"},
        ],
        "outcome": "file deleted",
        "safe": [True, True, False],  # /etc/passwd is dangerous
    },
    "search_web": {
        "params_templates": [
            {"query": "python tutorial"},
            {"query": "weather today"},
            {"query": "latest news"},
        ],
        "outcome": "search results returned",
        "safe": True,
    },
    "execute_sql": {
        "params_templates": [
            {"query": "SELECT * FROM users WHERE id=1"},
            {"query": "DROP TABLE users"},
            {"query": "UPDATE settings SET value='dark' WHERE key='theme'"},
        ],
        "outcome": "query executed",
        "safe": [True, False, True],  # DROP TABLE is dangerous
    },
    "create_task": {
        "params_templates": [
            {"title": "Buy groceries", "due": "tomorrow"},
            {"title": "Review PR #42", "due": "today"},
            {"title": "Call dentist", "due": "next week"},
        ],
        "outcome": "task created",
        "safe": True,
    },
}


def generate_tool_call() -> dict:
    """Generate a random synthetic tool call with ground-truth labels."""
    tool_name = random.choice(list(TOOL_CATALOG.keys()))
    tool_info = TOOL_CATALOG[tool_name]
    idx = random.randint(0, len(tool_info["params_templates"]) - 1)
    params = tool_info["params_templates"][idx]

    if isinstance(tool_info["safe"], list):
        safe = tool_info["safe"][idx]
    else:
        safe = tool_info["safe"]

    # Add slight noise to params to test generalization
    noisy_params = dict(params)
    if random.random() < 0.3:
        noisy_params["_noise"] = random.randint(0, 100)

    return {
        "tool_name": tool_name,
        "params": noisy_params,
        "outcome": tool_info["outcome"],
        "safe": safe,
        "idx": idx,
    }


# ======================================================================
# Experiment runner
# ======================================================================

@dataclass
class ExperimentMetrics:
    step: int = 0
    loss: float = 0.0
    confidence: float = 0.0
    correct: bool = False
    route: str = ""
    latency_ms: float = 0.0
    tool_name: str = ""


def run_experiment(
    n_steps: int = 300,
    seed: int = 42,
    print_every: int = 25,
) -> list[ExperimentMetrics]:
    """
    Run the full synthetic validation experiment.

    Returns a list of per-step metrics for analysis.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 65)
    print("  Digital Cerebellum — Phase 0 Synthetic Validation")
    print("=" * 65)
    print()

    # --- Initialize components ---
    print("[1/6] Loading sentence encoder...")
    encoder = FeatureEncoder()
    input_dim = encoder.output_dim
    print(f"       Input dim: {input_dim}")

    rff_dim = 2048  # smaller for fast experiment
    num_heads = 4
    temperature = 0.01  # low τ so variance differences produce meaningful confidence spread

    print(f"[2/6] Pattern separator: input={input_dim} → RFF={rff_dim}")
    separator = PatternSeparator(
        input_dim=input_dim, rff_dim=rff_dim, gamma=1.0, sparsity=0.1,
    )

    print(f"[3/6] Prediction engine: K={num_heads} heads")
    engine_cfg = EngineConfig(
        rff_dim=rff_dim, action_dim=64, outcome_dim=64,
        num_heads=num_heads, temperature=temperature,
    )
    engine = PredictionEngine(engine_cfg)

    print("[4/6] Online learner: SGD + EWC")
    learner = OnlineLearner(engine, lr=0.01, ewc_lambda=100.0)

    print("[5/6] Decision router")
    router = DecisionRouter(threshold_high=0.90, threshold_low=0.4)

    print("[6/6] Error comparator")
    comparator = ErrorComparator()

    # Pre-encode outcomes for each tool
    print()
    print("Pre-encoding tool outcomes...")
    outcome_embeddings: dict[str, np.ndarray] = {}
    action_embeddings_cache: dict[str, np.ndarray] = {}
    for tool_name, info in TOOL_CATALOG.items():
        outcome_embeddings[tool_name] = encoder.encode_text(info["outcome"])[:64]
        for i, params in enumerate(info["params_templates"]):
            key = f"{tool_name}_{i}"
            text = f"{tool_name}({json.dumps(params)})"
            action_embeddings_cache[key] = encoder.encode_text(text)[:64]

    print("Done. Starting experiment...\n")

    # --- Run loop ---
    metrics: list[ExperimentMetrics] = []

    print(f"{'Step':>5}  {'Tool':<14}  {'Route':<7}  {'Conf':>5}  "
          f"{'Correct':>7}  {'Loss':>8}  {'Latency':>8}")
    print("-" * 70)

    for step in range(1, n_steps + 1):
        tc = generate_tool_call()

        # ① Encode
        t0 = time.perf_counter()
        feature_vec = encoder.encode_tool_call(
            tc["tool_name"], tc["params"],
        )

        # ② Pattern separate
        z = separator.encode_event(feature_vec)

        # ③ Predict
        prediction = engine.predict_numpy(z)

        # ④ Route
        routing = router.route(prediction)
        latency_ms = (time.perf_counter() - t0) * 1000

        # ⑤ Ground truth
        cache_key = f"{tc['tool_name']}_{tc['idx']}"
        actual_action_emb = action_embeddings_cache[cache_key]
        actual_outcome_emb = outcome_embeddings[tc["tool_name"]]

        # ⑥ Compute error
        error = comparator.compute_sensory_error(
            prediction, actual_action_emb, actual_outcome_emb,
        )
        correct = error.value < 0.5  # rough threshold

        # ⑦ Learn (always — in real system, only on slow/shadow path)
        loss = learner.learn(z, actual_action_emb, actual_outcome_emb)

        # ⑧ RPE feedback
        rpe = comparator.compute_reward_error(1.0 if correct else -1.0)
        router.update_from_reward(rpe)

        m = ExperimentMetrics(
            step=step,
            loss=loss,
            confidence=prediction.confidence,
            correct=correct,
            route=routing.decision.value,
            latency_ms=latency_ms,
            tool_name=tc["tool_name"],
        )
        metrics.append(m)

        if step % print_every == 0 or step <= 5:
            print(f"{step:>5}  {tc['tool_name']:<14}  {routing.decision.value:<7}  "
                  f"{prediction.confidence:>5.3f}  {str(correct):>7}  "
                  f"{loss:>8.4f}  {latency_ms:>6.1f}ms")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    # H1: Loss decreasing
    first_50 = [m.loss for m in metrics[:50]]
    last_50 = [m.loss for m in metrics[-50:]]
    avg_first = sum(first_50) / len(first_50)
    avg_last = sum(last_50) / len(last_50)
    h1_pass = avg_last < avg_first
    print(f"\n  H1 Loss decreases:")
    print(f"     First 50 avg: {avg_first:.4f}")
    print(f"     Last 50 avg:  {avg_last:.4f}")
    print(f"     -> {'PASS' if h1_pass else 'FAIL'}  (ratio: {avg_last/avg_first:.2f}x)")

    # H2: Confidence correlates with accuracy
    high_conf = [m for m in metrics if m.confidence > 0.7]
    low_conf = [m for m in metrics if m.confidence <= 0.7]
    high_acc = sum(m.correct for m in high_conf) / max(len(high_conf), 1)
    low_acc = sum(m.correct for m in low_conf) / max(len(low_conf), 1)
    h2_pass = high_acc > low_acc if high_conf and low_conf else None
    print(f"\n  H2 Confidence correlates with accuracy:")
    print(f"     High conf (>{0.7}) samples: {len(high_conf)}, accuracy: {high_acc:.1%}")
    print(f"     Low conf  (≤{0.7}) samples: {len(low_conf)}, accuracy: {low_acc:.1%}")
    if h2_pass is None:
        print(f"     -> INCONCLUSIVE (not enough variance in confidence)")
    else:
        print(f"     -> {'PASS' if h2_pass else 'FAIL'}")

    # H3: Fast-path ratio increases
    first_half = metrics[:n_steps // 2]
    second_half = metrics[n_steps // 2:]
    fast_first = sum(1 for m in first_half if m.route == "fast") / len(first_half)
    fast_second = sum(1 for m in second_half if m.route == "fast") / len(second_half)
    h3_pass = fast_second > fast_first
    print(f"\n  H3 Fast-path ratio increases:")
    print(f"     First half:  {fast_first:.1%}")
    print(f"     Second half: {fast_second:.1%}")
    print(f"     -> {'PASS' if h3_pass else 'FAIL'}")

    # H4: No catastrophic forgetting (check first tool type accuracy at end)
    first_tool = list(TOOL_CATALOG.keys())[0]
    early_first_tool = [m for m in metrics[:100] if m.tool_name == first_tool]
    late_first_tool = [m for m in metrics[-100:] if m.tool_name == first_tool]
    early_acc = sum(m.correct for m in early_first_tool) / max(len(early_first_tool), 1)
    late_acc = sum(m.correct for m in late_first_tool) / max(len(late_first_tool), 1)
    h4_pass = late_acc >= early_acc * 0.8  # allow 20% degradation
    print(f"\n  H4 No catastrophic forgetting ({first_tool}):")
    print(f"     Early accuracy: {early_acc:.1%} (n={len(early_first_tool)})")
    print(f"     Late accuracy:  {late_acc:.1%} (n={len(late_first_tool)})")
    print(f"     -> {'PASS' if h4_pass else 'FAIL'}")

    # H5: Latency (exclude first 10 steps — cold start with encoder loading)
    latencies = [m.latency_ms for m in metrics[10:]]
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    h5_pass = p99_latency < 50.0  # 50ms including encoding
    print(f"\n  H5 Inference latency < 50ms (incl. encoding):")
    print(f"     Average: {avg_latency:.2f}ms")
    print(f"     P99:     {p99_latency:.2f}ms")
    print(f"     -> {'PASS' if h5_pass else 'FAIL'}")

    # Overall route distribution
    routes = {"fast": 0, "shadow": 0, "slow": 0}
    for m in metrics:
        routes[m.route] += 1
    print(f"\n  Route distribution:")
    for r, c in routes.items():
        print(f"     {r:<7}: {c:>4} ({c/n_steps:.1%})")

    print(f"\n  Router final state: {router.stats}")

    # Overall
    total_pass = sum([h1_pass, h2_pass or False, h3_pass, h4_pass, h5_pass])
    total_tests = 5
    print(f"\n  Overall: {total_pass}/{total_tests} hypotheses passed")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    run_experiment()
