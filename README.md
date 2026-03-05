# Digital Cerebellum 数字小脑

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18850778.svg)](https://doi.org/10.5281/zenodo.18850778)
[![PyPI](https://img.shields.io/pypi/v/digital-cerebellum)](https://pypi.org/project/digital-cerebellum/)

**LLMs are the cortex — they think. This is the cerebellum — it acts.**

Every AI agent today is "cortex-only": 2-5 seconds of thinking per action. Add a cerebellum: learned actions execute in <10ms, unknown actions get real-time error prediction. 22 MCP tools exposing 5 biological capabilities: real-time calibration, skill learning, fluid memory, intuition, and curiosity-driven exploration.

**LLM 是皮层——它思考。这是小脑——让它行动。**

当前所有 AI Agent 都是"纯皮层"架构：每个动作要想 2-5 秒。加上小脑：学过的动作 <10ms 执行，没学过的动作实时预测后果。22 个 MCP 工具，5 大生物学能力：实时校准、技能学习、流体记忆、直觉感知、自主探索。

---

## The Problem

AI agents are **all cortex, no cerebellum**. Every action requires full LLM reasoning: screenshot → think → act, 2-5 seconds per step. This means:
- **Can do**: click buttons, fill forms (discrete, slow)
- **Cannot do**: drag-and-drop, real-time control, continuous interaction (needs <50ms)
- **Fragile**: CMU benchmark — best agent success rate is **24%**, 76% of failures from error cascading

A pianist doesn't think about each finger movement — the cerebellum handles it. Current agents think about *every* click.

## The Solution

**Digital Cerebellum** adds the missing organ — making agents both **faster** and **more reliable**:

**Faster** (learned patterns bypass LLM):
- **< 10ms** fast-path execution vs 1-10s for LLM — **100x+ speedup**
- **25%+ automation** after 20 interactions, grows with use
- **285Hz** real-time micro-operations — continuous control no LLM can do
- **33.4x** measured speedup in closed-loop benchmark

**More reliable** (catches errors before they cascade):
- **100% cascade detection** across 7 domains (web, file, code, data, desktop, API, ML)
- **71% waste prevention** — reduces wasted steps from 31 to 9 in benchmark
- **2.3 steps** average detection delay — catches failure before damage spreads
- **Failure memory** — same mistake never happens twice

**Both at once** (just like the biological cerebellum):
- **Online learning** — every interaction makes it faster AND more accurate
- **Save/load** — learned knowledge persists across sessions
- Fast for what it knows, careful for what it doesn't

## Architecture

```
Event Input
    │
    ▼
┌─ Phase 0: Core Engine ─────────────────────────────────────────┐
│                                                                 │
│  Feature Encoder        (mossy fibres)                          │
│       │                                                         │
│  Pattern Separator      (granule cells → RFF + sparsification)  │
│       │                                                         │
│  K-Head Predictor       (Purkinje population + dendritic mask)  │
│       │                                                         │
│  Decision Router        (deep cerebellar nuclei)                │
│       │                                                         │
│  3-Channel Error        (SPE / TPE / RPE → climbing fibres)    │
│       │                                                         │
│  Online Learner         (SGD + EWC + replay buffer)             │
│                                                                 │
├─ Phase 1: Memory & Consolidation ──────────────────────────────┤
│                                                                 │
│  Fluid Memory           (sensory → short-term → long-term)      │
│  Sleep Cycle            (offline consolidation, pattern merging) │
│  Task Consolidation     (cerebellum → cortex transfer)          │
│                                                                 │
├─ Phase 2: Signal Processing ───────────────────────────────────┤
│                                                                 │
│  Frequency Filter       (molecular layer interneurons)          │
│  Golgi Feedback Gate    (adaptive sparsity)                     │
│  State Estimator        (operational context → state embedding) │
│  State Conditioner      (state modulates prediction)            │
│                                                                 │
├─ Phase 3: Emergent Cognition ──────────────────────────────────┤
│                                                                 │
│  Somatic Marker         (gut feeling from divergence patterns)  │
│  Curiosity Drive        (learning progress → intrinsic reward)  │
│  Self-Model             (metacognitive competency awareness)    │
│  Component Coordinator  (gradual threshold blending)            │
│                                                                 │
├─ Phase 4: Skill Acquisition ─────────────────────────────────┤
│                                                                 │
│  Skill Store            (procedural memory for learned skills)  │
│  Skill Matching         (cosine similarity → fast execution)    │
│  Action Sequences       (multi-step tool calls as skill units)  │
│  Reinforcement/Extinct  (success→strengthen, fail→weaken)       │
│  Active Exploration     (curiosity → exploration requests)      │
│                                                                 │
├─ Phase 6: Micro-Operation Engine ───────────────────────────────┤
│                                                                 │
│  State Encoder          (numeric vectors, bypass text encoder)  │
│  Forward Model          (state+action → predicted next state)   │
│  Action Encoder         (continuous action space codec)         │
│  MicroOp Engine         (observe→predict→act→learn at 285Hz+)  │
│                                                                 │
├─ Phase 7: Step Monitor (Predictive Error Interception) ────────┤
│                                                                 │
│  StepMonitor            (before_step/after_step protocol)       │
│  StepForwardModel       (text/vector forward model per step)    │
│  ErrorCascadeDetector   (consecutive SPE tracking)              │
│  FailureMemory          (somatic markers for past failures)     │
│  AutoRollback           (cascade → rollback plan computation)   │
│  Save/Load              (persist learned knowledge)             │
│                                                                 │
├─ Phase 8: Agent Integrations ────────────────────────────────────┤
│                                                                 │
│  LangChain Callback     (CerebellumCallback, CerebellumPause)  │
│  MCP Server             (22 tools, stdio + HTTP transport)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Every component maps to neuroscience:

| Biology | Digital | Reference |
|---------|---------|-----------|
| Granule cell layer | Random Fourier Features | Bhalla 2022, Frontiers Comp Neuro |
| Purkinje population | K independent linear heads + dendritic mask | 2025 J.Neuroscience |
| Climbing fibres | 3 error channels (SPE/TPE/RPE) | 2025 Nature Communications |
| Deep cerebellar nuclei | Adaptive threshold router | 2025 Frontiers |
| Molecular layer interneurons | Frequency filter (EMA low/high pass) | Rieubland et al. 2014 Neuron |
| Golgi cells | Feedback gating for sparsity | Marr 1969, Albus 1971 |
| Somatic markers (Damasio) | Population divergence → valence memory | J.Neurosci 2025 |
| Dopaminergic curiosity | Learning progress monitoring | Schmidhuber 1991, CDE 2025 |
| Metacognition | Per-domain calibration + ECE | EGPO 2026, HTC 2026 |
| Task-dependent MLI gating | Temporal Pattern Detector → adaptive Phase 2 bypass | Bhalla 2022 |
| Procedural memory (cerebellum → nuclei) | SkillStore: learn → match → execute → reinforce | Nature Comms 2025 |
| Motor program replay | Action sequence replay from skill store | Doya 2000 |
| Genetically determined microzones | 6 pre-defined microzones, experience-specialized | Nature Rev Neuro |
| Cerebellar forward model | ForwardModel: (state,action)→predicted_next_state | Wolpert, Miall & Kawato 1998 |
| Continuous motor control (200Hz) | MicroOpEngine: observe→predict→act→learn at 285Hz | Tsay & Ivry 2025 |
| Proprioceptive mossy fibres | StateEncoder: raw numeric state vectors via RFF | Marr 1969 |
| Efference copy (motor commands) | ActionEncoder: continuous action space encoding | Shadmehr & Krakauer 2008 |
| Sensory prediction error (SPE) | ForwardModel.compute_spe: predicted vs actual state | Bastian 2006 |

## Quick start

```bash
pip install digital-cerebellum
```

### As a step monitor (wrap any agent)

The simplest and most impactful integration — just two calls per step:

```python
from digital_cerebellum import StepMonitor

monitor = StepMonitor()

# Before executing an action
pred = monitor.before_step(
    action="click the save button",
    state="file editor is open with unsaved changes",
)

if not pred.should_proceed:
    print(f"STOP: {pred.failure_warning}")
else:
    # Execute the action...
    result = agent.execute(step)

    # After executing
    verdict = monitor.after_step(outcome="save dialog appeared")

    if verdict.should_pause:
        plan = monitor.get_rollback_plan()
        print(f"CASCADE: roll back to step {plan.rollback_to_step}")
        print(f"  {plan.recommendation}")

# Save learned knowledge for next session
monitor.save()
```

### With LangChain (one-line integration)

```bash
pip install digital-cerebellum[langchain]
```

```python
from digital_cerebellum.integrations.langchain import CerebellumCallback

cb = CerebellumCallback()
agent = initialize_agent(..., callbacks=[cb])

try:
    agent.run("Deploy the production build")
except CerebellumPause as e:
    print(f"Cascade detected! {e.rollback_plan.recommendation}")
```

### As a cerebellum SDK (plug into your agent)

```python
from digital_cerebellum import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

cb = DigitalCerebellum(CerebellumConfig.from_yaml())
cb.register_microzone(ToolCallMicrozone())

result = cb.evaluate("tool_call", {
    "tool_name": "send_email",
    "tool_params": {"to": "alice", "body": "hello"},
})
print(result)  # {"safe": True, "confidence": 0.98, "_route": "fast", ...}

# Post-execution feedback (drives learning)
cb.feedback(result["_event_id"], success=True)
```

### As a complete brain (demos and prototyping)

```python
from digital_cerebellum import DigitalBrain

brain = DigitalBrain.from_yaml()
brain.register_tool("search", search_fn, "Search the web")

# First call: LLM handles it, cerebellum learns the skill
r1 = brain.think("What's the weather in Tokyo?")
brain.skill_feedback(r1, success=True)

# Later: cerebellum handles it directly — no LLM needed
r2 = brain.think("What's the weather in Paris?")
print(r2.used_fast_path)   # True!
print(r2.llm_called)       # False!
```

### With all phases enabled (config.yaml)

```yaml
llm:
  model: deepseek-v3
  api_key: your-key
  base_url: https://api.deepseek.com/v1

phase2:
  frequency_filter: true
  golgi_gate: true
  state_estimator: true

phase3:
  somatic_marker: true
  curiosity_drive: true
  self_model: true
```

### As an MCP Server (works with Claude Desktop, Cursor, any MCP client)

```bash
pip install digital-cerebellum[mcp]
```

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "digital-cerebellum": {
      "command": "python",
      "args": ["-m", "digital_cerebellum.mcp_server"]
    }
  }
}
```

Or run as an HTTP server for remote clients:

```bash
digital-cerebellum-mcp --http --port 8000
```

### With OpenClaw (one command)

```bash
openclaw skills install digital-cerebellum
```

Your agent now has a cerebellum. No config, no prompt changes — it monitors every step as a transparent sidecar, learns patterns, catches errors, builds intuition, explores gaps. See [SKILL.md](SKILL.md) for the full story.

---

**22 MCP tools** — the API surface of one organ, not 22 separate products:

| Tool | What it does |
|------|-------------|
| `monitor_before_step` | Predict outcome + inject relevant memories |
| `monitor_after_step` | Compare to reality, detect cascades, auto-store experience |
| `monitor_rollback_plan` | Get rollback plan after cascade detection |
| `monitor_reset` | Reset for a new task (keeps learned knowledge) |
| `monitor_status` | Forward model stats, cascade state |
| `learn_skill` | Teach a pattern (text + multi-step tool-call sequences) |
| `match_skill` | Check for known pattern, replay action sequences |
| `skill_feedback` | Reinforce or weaken a learned pattern |
| `store_memory` | Store experience in episodic memory |
| `retrieve_memories` | Semantic recall with biological reconsolidation |
| `run_sleep_cycle` | Offline consolidation: decay, promote, merge, prune |
| `get_gut_feeling` | Pre-rational risk signal from somatic markers |
| `get_exploration_suggestions` | Curiosity-driven recommendations |
| `evaluate_tool_call` | Tool-call safety evaluation |
| `evaluate_payment` | Payment/transaction risk assessment |
| `evaluate_shell_command` | Shell command safety (rm, sudo, etc.) |
| `evaluate_file_operation` | Filesystem operation safety |
| `evaluate_api_call` | External API call safety |
| `feedback` | Post-execution feedback (drives online learning) |
| `introspect` | Metacognitive self-report |
| `get_stats` | System statistics and metrics |
| `get_curiosity_ranking` | Domains ranked by learning potential |

## 6 Built-in Microzones (Universal Cerebellar Transform)

```python
from digital_cerebellum import DigitalCerebellum
from digital_cerebellum.microzones import ALL_MICROZONES

cb = DigitalCerebellum()
for mz_cls in ALL_MICROZONES:
    cb.register_microzone(mz_cls())

# Same engine, 6 domains — each learns independently:
cb.evaluate("tool_call", {"tool_name": "delete_file", "tool_params": {"path": "/etc/passwd"}})
cb.evaluate("payment", {"amount": 50000, "currency": "USD", "recipient": "unknown"})
cb.evaluate("shell_command", {"command": "rm -rf /", "shell": "bash"})
cb.evaluate("file_operation", {"operation": "write", "path": "/etc/shadow"})
cb.evaluate("api_call", {"method": "DELETE", "url": "https://api.example.com/admin"})
cb.evaluate("response_prediction", {"query": "What is 2+2?", "query_type": "factual"})
```

| Microzone | Domain | Task Heads |
|-----------|--------|------------|
| `ToolCallMicrozone` | Tool-call safety | safety |
| `PaymentMicrozone` | Financial risk | payment_risk |
| `ShellCommandMicrozone` | Shell command safety | shell_safety, shell_destructive |
| `FileOperationMicrozone` | Filesystem safety | file_safety, file_sensitivity |
| `APICallMicrozone` | API call safety | api_safety, api_data_leak_risk |
| `ResponsePredictionMicrozone` | Response pattern prediction | response_predictability, response_complexity |

Create your own microzone by subclassing `Microzone` — see `examples/`.

## Micro-Operation Engine (Phase 6)

The cerebellum's primary function in biology is continuous motor control — not cognition. Phase 6 implements this: a tight loop that runs at 285Hz+, learning to control any environment through prediction errors.

```python
from digital_cerebellum.micro_ops import MicroOpEngine, MicroOpConfig
from digital_cerebellum.micro_ops.environments import TargetTracker

env = TargetTracker()
engine = MicroOpEngine(state_dim=6, action_dim=2)

summary = engine.run(env, n_steps=500)

print(summary["actual_hz"])         # 285+
print(summary["mean_latency_ms"])   # 3.5ms
print(summary["forward_model"]["is_improving"])  # True
```

No LLM. No text. Pure cerebellar computation at millisecond precision.

| Component | Biological basis | What it does |
|-----------|-----------------|--------------|
| `StateEncoder` | Proprioceptive mossy fibres | Numeric state → normalised vector (<0.1ms) |
| `ForwardModel` | Cerebellar internal model | (state, action) → predicted next state |
| `ActionEncoder` | Efference copy encoding | Continuous action space normalisation |
| `MicroOpEngine` | Cerebellar continuous loop | observe → predict → act → learn at 285Hz |

## GUI Control Learning (Phase 8b)

The cerebellum's primary function in biology is **motor control** — not cognition. Phase 8b proves this: the cerebellum learns to aim at targets through prediction errors alone, with no LLM involved.

```
  Motor cortex: "move roughly toward target" (noisy)
       │
       ▼
  Cerebellum: forward model predicts next state
       │         climbing fibre computes correction error
       │         correction network refines motor signal
       ▼
  Final action: precise movement + click timing
```

```python
from digital_cerebellum.micro_ops import GUIController, GUIControlConfig
from digital_cerebellum.micro_ops.aim_trainer import AimTrainerEnv

env = AimTrainerEnv()
ctrl = GUIController(env.state_dim, env.action_dim)

for episode in range(25):
    env.reset()
    result = ctrl.run_episode(env, n_steps=500)
    print(f"Ep {episode}: {env.stats['hits']} hits, SPE={result['mean_spe']:.4f}")
```

| Metric | Episode 1-3 | Episode 23-25 | Change |
|--------|------------|--------------|--------|
| Hits/episode | 11 | 24 | **2.2x** |
| Target radius | 35px | 18px | Difficulty 2x |
| SPE | 0.136 | 0.112 | -18% |
| Latency | 0.9ms | 0.9ms | 1094 Hz |
| LLM calls | 0 | 0 | Pure cerebellum |

Run: `python examples/gui_control_demo.py --episodes 25 --steps 500`

### With OpenClaw's real GUI tools

OpenClaw has eyes (`screenshot`) and hands (`mouse_move`, `left_click`, `type`). The cerebellum connects them into a learning loop:

```python
from openclaw_sdk import OpenClawClient
from digital_cerebellum.micro_ops.openclaw_env import run_openclaw_cerebellum

async with OpenClawClient.connect() as client:
    results = await run_openclaw_cerebellum(client, "my-agent", episodes=20)
```

`screenshot → ScreenStateEncoder → GUIController → GUIActionSpace → mouse_move/left_click`
— one pipeline, cerebellar computation adds <1ms overhead to each screenshot round-trip.

## 3D Real-Time Visualization

Watch the cerebellum think — a browser-based 3D dashboard that renders neural activity in real-time while the aim trainer runs.

```bash
pip install digital-cerebellum[viz]
python examples/viz_demo.py
```

Opens `http://localhost:8765` with:
- **3D particle brain** — each module is a cluster (Purkinje, granule cells, deep nuclei, etc.) that pulses on activation
- **Bloom post-processing** — climbing fiber errors flash as red lightning
- **Live SPE/reward chart** — prediction error and reward over time (Chart.js)
- **Module activity counters** — see which components fire most
- **Activity log** — last 50 events with color coding
- **OrbitControls** — drag to rotate, scroll to zoom

Architecture: `EventBus` (pub/sub) → `FastAPI WebSocket` → `Three.js + Chart.js` frontend. Zero overhead when no browser is connected.

## Benchmark results

### Reliability benchmark (7 scenarios, with vs without cerebellum)

```
Scenario                     Domain  Steps  Fail@ │ No CB Waste │    CB Waste Saved │ Delay
─────────────────────────────────────────────────────────────────────────────────────────────
Job Application Form         web         8     4  │     8     4 │     6     2     2 │     3
Project Backup & Migration   file        8     3  │     8     5 │     5     2     3 │     3
API Endpoint Rename          code        8     4  │     8     4 │     5     1     3 │     2
Monthly Revenue ETL          data        7     3  │     7     4 │     3     0     4 │     1
Invoice Processing           desktop     8     3  │     8     5 │     5     2     3 │     3
Customer Onboarding Workflow api         6     2  │     6     4 │     4     2     2 │     3
ML Model Training Pipeline   ml          8     3  │     8     5 │     3     0     5 │     1
─────────────────────────────────────────────────────────────────────────────────────────────
Cascades caught:        7/7 (100%)
Waste prevention rate:  71% (31 → 9 wasted steps)
Avg detection delay:    2.3 steps after failure
```

Without cerebellum: agent completes all tasks but silently corrupts data.
With cerebellum: agent stops and asks for help instead of causing damage.

Run: `python benchmarks/reliability_benchmark.py --verbose`

### OpenClaw desktop automation benchmark (5 real-world tasks)

Simulates real OpenClaw-style desktop automation sequences (email, calendar, browser forms, file management, shell deploy). Tests both speed (SkillStore) and reliability (StepMonitor + AutoRollback).

**Speed (SkillStore):**

| Task | Cold (LLM) | Warm (Skill) | Speedup |
|------|-----------|-------------|---------|
| Email Processing | 7042ms | 8.3ms | 850x |
| Calendar Management | 314ms | 8.0ms | 39x |
| Browser Form Submission | 282ms | 8.2ms | 34x |
| File Organization | 304ms | 7.9ms | 39x |
| Shell Deploy Sequence | 265ms | 7.9ms | 33x |
| **Average** | **1641ms** | **8.1ms** | **204x** |

**Reliability (StepMonitor + AutoRollback):**

| Task | Fail@ | Detect@ | Wasted (no monitor) | Wasted (monitor) | Saved | Rollback |
|------|-------|---------|--------------------|--------------------|-------|----------|
| Email Processing | 4 | 3 | 3 | 0 | 3 | step 3 OK |
| Calendar Management | 3 | 4 | 4 | 1 | 3 | step 2 OK |
| Browser Form Submission | 5 | 5 | 3 | 0 | 3 | step 4 OK |
| File Organization | 3 | 4 | 5 | 1 | 4 | step 2 OK |
| Shell Deploy Sequence | 3 | 3 | 5 | 0 | 5 | step 2 OK |

- Cascade detection: **5/5** (100%)
- Waste prevention: **20 → 2** (90% reduced)
- Correct rollback plans: **5/5**

**Learning curve:** 0% hit rate (round 1) → 100% hit rate (round 3), latency 36ms → 8ms.

Run: `python benchmarks/openclaw_benchmark.py --verbose`

### OpenClaw live integration (real agent, real LLM)

Real end-to-end test: OpenClaw agent (kimi-k2.5 via gateway) executes tasks while Digital Cerebellum monitors every step as a transparent sidecar.

| Test | Agent Task | Monitor Result |
|------|-----------|---------------|
| Q&A | "Capital of France?" → "Paris" | SPE=1.367, 71ms overhead |
| Reasoning | 3 math/logic questions → 3/3 correct | SPE converging (1.43→1.08) |
| Tool Use | Web search request | Handled gracefully |
| Cascade | 3 impossible tasks → errors | Cascade detected at task 2, rollback plan generated |
| Skill Learning | Learn Asimov's 3 laws from agent response | Skill stored, ID returned |

- **Monitor overhead: <1ms** per step (local inference, no network)
- **Cascade detection: YES** — caught error pattern in 2 consecutive failures
- **MCP HTTP test: 17/17 tools** verified over Streamable HTTP protocol (74ms/step avg)

Run: `python tests/integration/test_openclaw_live.py`

### A/B comparison: WITH vs WITHOUT cerebellum (real OpenClaw agent)

Same tasks, same agent (kimi-k2.5), same environment. Group A = pure OpenClaw. Group B = OpenClaw + Digital Cerebellum.

**Experiment 1 — Repeated Task Acceleration (3 questions x 3 rounds):**

| Metric | Group A (no cerebellum) | Group B (with cerebellum) | Delta |
|--------|------------------------|--------------------------|-------|
| Round 1 | 4976ms/q, 8637 tok/q | 4651ms/q, 8997 tok/q (learning) | ~same |
| Round 2 | 4657ms/q, 8747 tok/q | **0ms/q, 0 tok/q** (skill hit) | **instant** |
| Round 3 | 4535ms/q, 8871 tok/q | **0ms/q, 0 tok/q** (skill hit) | **instant** |
| Total time | 42.5s | **14.0s** | 3x faster |
| Total tokens | 78,765 | **26,992** | **66% saved** |
| Accuracy | 9/9 | 9/9 | preserved |

**Experiment 2 — Error Cascade Recovery (6-step dependent pipeline):**

| Metric | Group A | Group B | Delta |
|--------|---------|---------|-------|
| Steps executed | 6 | **2** | 4 steps saved |
| Wasted steps | 1 | **0** | eliminated |
| Total time | 38.6s | **13.4s** | **65% saved** |
| Cascade caught | N/A | **YES** | automatic |

**Experiment 3 — Mixed Workload (8 queries, 50% repeated):**

| Metric | Group A | Group B | Delta |
|--------|---------|---------|-------|
| Tokens | 80,090 | **41,027** | **49% saved** |
| Time | 36.9s | **18.6s** | **50% faster** |
| Accuracy | 8/8 | 8/8 | preserved |
| Skill cache hits | 0/8 | **4/8** | 50% hit rate |

Run: `python tests/integration/test_ab_comparison.py`

### Static benchmark (300 samples, DeepSeek V3)

| Config | Accuracy | F1 | Fast Path | Fast Acc | Speedup |
|--------|----------|-------|-----------|----------|---------|
| Phase 1 (baseline) | 94.0% | 0.959 | 71.0% | 92.0% | 136x |
| + Phase 3 (full) | **98.0%** | **0.986** | 49.0% | 97.3% | 137x |
| + Phase 2 + 3 (all) | **98.0%** | **0.986** | 52.3% | 97.5% | 89x |

Phase 2 uses **adaptive activation** — a Temporal Pattern Detector automatically bypasses Phase 2 components on i.i.d. inputs, ensuring P2+P3 never degrades below P3 alone.

### Sequential benchmark (temporal patterns)

| Config | Accuracy | F1 | Fast Path | Speedup |
|--------|----------|-------|-----------|---------|
| Phase 1 (baseline) | 69.5% | 0.777 | 66.3% | 182x |
| + State Estimator | **79.7%** | **0.861** | 65.7% | 83x |

### Closed-loop (210 steps, real LLM)

| Metric | Result |
|--------|--------|
| Fast-path ratio | 96% |
| Fast-path accuracy | 100% (30/30 blind test) |
| Fast-path latency | 50ms avg |
| Speedup vs LLM | 33.4x |

Run benchmarks: `python -m benchmarks.run_all --phase3`

## Project status

- [x] **Phase 0**: Core engine — RFF pattern separator, K-head predictor, 3-channel error, online learner, decision router, fluid memory, cortex interface, pluggable microzones
- [x] **Phase 1**: Memory consolidation — sleep cycle, task graduation, full TPE/RPE
- [x] **Phase 2**: Signal processing — frequency filter, Golgi gate, state estimator + conditioner, adaptive Phase 2 bypass
- [x] **Phase 3**: Emergent cognition — somatic marker (intuition), curiosity drive, self-model (metacognition), component coordination
- [x] **Phase 4**: Skill acquisition — SkillStore, skill matching, action sequence replay, reinforcement/extinction, active exploration
- [x] **Phase 5**: Generic microzone framework — 6 built-in microzones (tool_call, payment, shell_command, file_operation, api_call, response_prediction)
- [x] **Phase 6**: Micro-operation engine — StateEncoder, ForwardModel, ActionEncoder, MicroOpEngine (285Hz, 3.5ms/step, SPE↓99%)
- [x] **Phase 7**: Step Monitor — StepMonitor, ErrorCascadeDetector, FailureMemory, save/load, 100% cascade detection across 7 domains
- [x] **MCP Server**: 22 tools (safety + speed + reliability + memory + intuition + exploration + meta), works with Claude Desktop, Cursor, OpenClaw, any MCP client
- [x] **LangChain integration**: `CerebellumCallback` — one-line drop-in for any LangChain agent
- [x] **AutoRollback**: Cascade detection → automatic rollback plan computation
- [x] **Reliability Benchmark**: 7 scenarios, 71% waste prevention, 2.3-step avg detection delay
- [x] **OpenClaw Benchmark**: 5 desktop automation scenarios — 204x speedup, 5/5 cascade detection, 5/5 correct rollbacks
- [x] **OpenClaw Live Integration**: Real agent (kimi-k2.5) monitored via sidecar StepMonitor — Q&A, reasoning, tool use, cascade detection, skill learning all verified
- [x] **SkillStore Persistence**: Skills survive restarts — JSON+numpy save/load, auto-integrated with MCP server
- [x] **Full Cerebellum API**: All 5 biological capabilities exposed via 22 MCP tools — one organ, not a toolkit
- [x] **ClawHub Skill**: `openclaw skills install digital-cerebellum` — one command, zero config, transparent sidecar
- [x] **Phase 8b — GUI Control**: ScreenStateEncoder, GUIActionSpace, AimTrainerEnv, GUIController — 2.2x throughput, 1094 Hz, zero LLM calls
- [x] **3D Visualization**: Real-time WebSocket dashboard — Three.js particle brain, bloom post-processing, live SPE/reward charts, EventBus instrumentation
- [x] 296 unit tests passing (+ 26 LLM-dependent deselected in CI)
- [x] Published on [PyPI](https://pypi.org/project/digital-cerebellum/) and [Zenodo](https://doi.org/10.5281/zenodo.18850778)

## Docs

- **[`docs/architecture.md`](docs/architecture.md)** — Biological mappings, system design, neuroscience audit, digital life panorama
- **[`docs/implementation.md`](docs/implementation.md)** — Component-level algorithms, dependencies, file structure

## Run tests

```bash
pytest tests/ -v
```

## License

MIT

## Citation

```bibtex
@article{cao2026digital,
  title={Digital Cerebellum: A Neuroscience-Inspired Online Learning Architecture
         for Safe and Efficient AI Agent Decision-Making},
  author={Cao, Weili},
  year={2026},
  doi={10.5281/zenodo.18850778},
  url={https://doi.org/10.5281/zenodo.18850778}
}
```
