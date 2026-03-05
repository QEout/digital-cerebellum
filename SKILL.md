# Digital Cerebellum — Give Your Agent a Cerebellum

> LLMs are the cortex — they think. This is the cerebellum — it acts.

## What This Is

A cerebellum. One organ, one install.

Your AI agent today is "cortex-only" — every action needs full LLM reasoning: 2-5 seconds of thinking per click. That's like a pianist thinking about each finger movement. The biological solution? A cerebellum: a separate organ that handles execution, timing, error correction, and learned patterns — so the cortex can focus on planning.

**Digital Cerebellum does the same thing for your AI agent.** After installation, your agent automatically gains five capabilities — not as separate tools you wire up, but as emergent properties of having a cerebellum:

1. **Learned actions execute instantly** — the cerebellum remembers how you did it last time (<10ms vs 2-5s)
2. **Errors get caught before they cascade** — prediction errors trigger early stopping, not post-mortem debugging
3. **Experiences persist across sessions** — episodic memory that decays, consolidates, and reconsolidates like biological memory
4. **Gut feelings warn before reasoning does** — somatic markers from past patterns provide pre-rational risk signals
5. **Curiosity drives exploration** — learning progress tracking reveals blind spots and suggests what to try next

These aren't five separate products. They're what happens when you give an agent a cerebellum.

## Why It Matters

| Without cerebellum | With cerebellum |
|---|---|
| Every action = full LLM call (2-5s) | Learned patterns = <10ms |
| Error at step 3 → cascades to step 10 | Error at step 3 → caught at step 4 |
| Restart = blank slate | Restart = picks up where it left off |
| No intuition about risk | "This feels wrong" before reasoning starts |
| Repeats same exploration forever | Knows what it knows, seeks what it doesn't |

**Measured on real OpenClaw tasks:**
- **3x faster** overall, **66% fewer tokens** (repeated tasks skip LLM entirely)
- **100% cascade detection**, **71% waste prevention** (stops before damage)
- **285Hz continuous control** (real-time mouse/keyboard, not step-by-step)

## Quick Start

Install the skill in OpenClaw:

```
openclaw skills install digital-cerebellum
```

That's it. Your agent now has a cerebellum.

The cerebellum works as a **transparent sidecar** — it monitors every step your agent takes, learns from successes, intercepts failures, and accelerates repeated patterns. No changes to your agent's prompts or workflow needed.

## How It Works (You Don't Need to Know This)

The cerebellum exposes 22 MCP tools, but you rarely call them explicitly. The two most important integrate automatically:

- **`monitor_before_step`** — called before every action. Predicts outcome, injects relevant memories, warns of danger.
- **`monitor_after_step`** — called after every action. Compares prediction to reality, detects cascades, stores experience.

Everything else emerges from this loop:
- Repeated patterns get absorbed into procedural memory (skills)
- Skills accelerate future execution (bypass LLM entirely)
- Failed patterns get flagged in failure memory
- Memory consolidates during sleep cycles
- Somatic markers build intuition from experience
- Curiosity tracks what's been explored vs. unexplored

This is exactly how the biological cerebellum works: a prediction-correction loop that learns timing, sequences, and error patterns through experience.

## Example: Desktop Automation

```
Round 1: Agent processes email (LLM path, 5s)
         → Cerebellum watches, learns the pattern

Round 2: Same type of email arrives
         → Cerebellum: "I know this" → executes in 8ms
         → LLM never called

Round 3: Agent tries to delete system files
         → Cerebellum: "This feels wrong" (somatic marker)
         → Prediction error spikes → cascade warning
         → Agent stops, asks for help instead of causing damage
```

No configuration. No prompt engineering. The cerebellum just makes your agent better.

## For Developers Who Want Fine Control

While the cerebellum works automatically, power users can call any of the 22 tools directly:

| What you want | Tool |
|---|---|
| Predict before acting | `monitor_before_step` |
| Evaluate after acting | `monitor_after_step` |
| Get rollback plan | `monitor_rollback_plan` |
| Teach a new skill (including multi-step sequences) | `learn_skill` |
| Check for known skill | `match_skill` |
| Reinforce/weaken a skill | `skill_feedback` |
| Store an experience | `store_memory` |
| Recall relevant experiences | `retrieve_memories` |
| Consolidate memory offline | `run_sleep_cycle` |
| Ask gut feeling | `get_gut_feeling` |
| Get exploration ideas | `get_exploration_suggestions` |
| Evaluate tool call safety | `evaluate_tool_call` |
| Evaluate payment risk | `evaluate_payment` |
| Evaluate shell command | `evaluate_shell_command` |
| Evaluate file operation | `evaluate_file_operation` |
| Evaluate API call | `evaluate_api_call` |
| Post-execution feedback | `feedback` |
| Self-assessment | `introspect` |
| System stats | `get_stats` |
| Curiosity ranking | `get_curiosity_ranking` |
| Reset for new task | `monitor_reset` |
| Monitor status | `monitor_status` |

But the whole point of a cerebellum is that you shouldn't have to think about it. Install it and let it work.

## Requirements

- Python 3.10+
- No API key required for core functionality (pure local inference)
- Optional: LLM API key for slow-path evaluation on novel inputs

## Links

- [GitHub](https://github.com/QEout/digital-cerebellum)
- [PyPI](https://pypi.org/project/digital-cerebellum/)
- [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18850778)
- [Architecture Deep Dive](https://github.com/QEout/digital-cerebellum/blob/main/docs/architecture.md)

## License

MIT
