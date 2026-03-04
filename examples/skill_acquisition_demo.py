"""
Skill Acquisition Demo — proving the cerebellum can learn and automate.

This demo shows the core value proposition of the Digital Cerebellum:
  1. First encounter: LLM handles everything (slow, expensive)
  2. Cerebellum learns from the interaction
  3. Similar queries later: cerebellum handles directly (fast, free)
  4. Over time: automation ratio increases, cost decreases

No real LLM is needed — we simulate the cortex to make the demo
self-contained and reproducible.

Run:
    python examples/skill_acquisition_demo.py
"""

from __future__ import annotations

import time
from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum


SIMULATED_QUERIES = [
    # Round 1: First encounters (all will be slow path → learn)
    ("What's the weather in Tokyo?", "The weather in Tokyo is sunny, 25°C with light winds."),
    ("What's the weather in Paris?", "The weather in Paris is cloudy, 18°C with chance of rain."),
    ("How do I sort a list in Python?", "Use sorted(my_list) for a new list or my_list.sort() for in-place sorting."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain recursion briefly", "Recursion is when a function calls itself to solve smaller subproblems until reaching a base case."),

    # Round 2: Similar queries (should match learned skills)
    ("What's the weather in London?", "The weather in London is rainy, 14°C."),
    ("Tell me the weather in Tokyo", ""),  # should match the Tokyo skill
    ("How to sort a list in Python?", ""),  # should match the sorting skill
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What's the weather in Paris today?", ""),  # should match the Paris skill

    # Round 3: More repetition (automation should increase)
    ("Weather in Tokyo please", ""),
    ("Sort a Python list", ""),
    ("Capital of France?", ""),
    ("Explain recursion", ""),
    ("Weather in Paris", ""),

    # Round 4: Novel queries (should NOT match)
    ("How does photosynthesis work?", "Photosynthesis converts CO2 and water into glucose using sunlight."),
    ("What is machine learning?", "ML is a subset of AI where systems learn patterns from data."),

    # Round 5: Back to familiar territory
    ("Weather Tokyo", ""),
    ("Sort list Python", ""),
    ("Capital France", ""),
]


def main():
    print("=" * 70)
    print("SKILL ACQUISITION DEMO")
    print("Proving: LLM + Cerebellum > LLM alone")
    print("=" * 70)
    print()

    cfg = CerebellumConfig()
    cb = DigitalCerebellum(cfg)

    total = 0
    skill_hits = 0
    skill_misses = 0
    total_simulated_cost = 0.0

    LLM_COST_PER_CALL = 0.003  # $0.003 per LLM call
    CEREBELLUM_COST = 0.00001  # $0.00001 per cerebellum call

    print(f"{'#':>3}  {'Query':<40}  {'Path':<12}  {'Sim':>5}  {'Conf':>5}  {'Cost':>8}")
    print("-" * 85)

    for i, (query, simulated_response) in enumerate(SIMULATED_QUERIES, 1):
        total += 1
        t0 = time.perf_counter()

        match = cb.match_skill(query)

        if match is not None and match.should_execute:
            skill_hits += 1
            latency = (time.perf_counter() - t0) * 1000
            cost = CEREBELLUM_COST
            total_simulated_cost += cost

            # Positive feedback: skill worked → reinforce
            cb.skill_store.reinforce(match.skill.id)

            print(
                f"{i:3d}  {query:<40}  {'CEREBELLUM':<12}  "
                f"{match.similarity:5.3f}  {match.match_confidence:5.3f}  "
                f"${cost:.5f}"
            )
        else:
            skill_misses += 1
            response = simulated_response or f"[simulated response for: {query}]"

            sid = cb.learn_skill(
                input_text=query,
                response_text=response,
                domain="response",
            )

            # Positive feedback: LLM response was good → reinforce the new skill
            cb.skill_store.reinforce(sid)

            latency = (time.perf_counter() - t0) * 1000
            cost = LLM_COST_PER_CALL
            total_simulated_cost += cost

            sim_str = f"{match.similarity:5.3f}" if match else "  N/A"
            conf_str = f"{match.match_confidence:5.3f}" if match else "  N/A"
            print(
                f"{i:3d}  {query:<40}  {'LLM (learn)':<12}  "
                f"{sim_str}  {conf_str}  "
                f"${cost:.5f}"
            )

    print("-" * 85)
    print()

    automation_ratio = skill_hits / total if total > 0 else 0
    pure_llm_cost = total * LLM_COST_PER_CALL
    savings = (1 - total_simulated_cost / pure_llm_cost) * 100

    print("RESULTS")
    print(f"  Total queries:      {total}")
    print(f"  Skill hits:         {skill_hits} ({automation_ratio:.0%} automation)")
    print(f"  LLM calls needed:   {skill_misses}")
    print(f"  Skills learned:     {len(cb.skill_store)}")
    print()
    print("COST ANALYSIS")
    print(f"  Pure LLM cost:      ${pure_llm_cost:.4f}")
    print(f"  With cerebellum:    ${total_simulated_cost:.4f}")
    print(f"  Savings:            {savings:.1f}%")
    print()

    print("SKILL STORE")
    for skill in cb.skill_store.get_skills():
        print(
            f"  [{skill.confidence:.2f}] {skill.input_text[:50]:<50}  "
            f"accessed={skill.access_count}"
        )

    print()
    print("KEY INSIGHT:")
    print(f"  After {total} interactions, the cerebellum handles {automation_ratio:.0%}")
    print("  of queries without calling the LLM.")
    print("  Each new interaction makes the system smarter.")
    print("  This is what 'growing through experience' looks like.")


if __name__ == "__main__":
    main()
