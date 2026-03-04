# Digital Cerebellum 数字小脑

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18850778.svg)](https://doi.org/10.5281/zenodo.18850778)
[![PyPI](https://img.shields.io/pypi/v/digital-cerebellum)](https://pypi.org/project/digital-cerebellum/)

A cerebellar-inspired cognitive architecture that makes AI agents reliable — predictive error interception, skill learning, real-time micro-operations, and experience accumulation.

一个受生物小脑启发的认知架构：通过预测性错误拦截、技能学习、实时微操和经验积累，让 AI Agent 不翻车。

---

## The Problem

AI agents are fragile. CMU's TheAgentCompany benchmark: best agent success rate is **24%**. 76% of failures come from **error cascading** — one step goes wrong, every subsequent step fails, and the agent doesn't even notice.

## The Solution

Biological brains don't have this problem. When you trip while carrying a glass of water, your **cerebellum** detects the posture deviation in 50ms and corrects — before your cortex even knows something happened.

**Digital Cerebellum** does the same for AI agents — making them both **faster** and **more reliable**:

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
│  MCP Server             (17 tools, stdio + HTTP transport)      │
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

**Exposed tools (17):**

| Tool | Category | Description |
|------|----------|-------------|
| `evaluate_tool_call` | Safety | Evaluate tool-call safety before execution |
| `evaluate_payment` | Safety | Assess payment/transaction risk |
| `evaluate_shell_command` | Safety | Evaluate shell command safety (rm, sudo, etc.) |
| `evaluate_file_operation` | Safety | Evaluate filesystem operation safety |
| `evaluate_api_call` | Safety | Evaluate external API call safety |
| `learn_skill` | Speed | Teach the cerebellum a new skill from an interaction |
| `match_skill` | Speed | Check if the cerebellum can handle a query directly |
| `skill_feedback` | Speed | Reinforce/weaken a learned skill |
| `monitor_before_step` | Reliability | Predict outcome before agent executes an action |
| `monitor_after_step` | Reliability | Compare actual outcome, detect error cascades |
| `monitor_rollback_plan` | Reliability | Get auto-rollback plan after cascade detection |
| `monitor_reset` | Reliability | Reset monitor for a new task (keeps learned knowledge) |
| `monitor_status` | Reliability | Get forward model stats, cascade state, failure count |
| `feedback` | Learning | Post-execution feedback (drives online learning) |
| `introspect` | Meta | Metacognitive self-report |
| `get_stats` | Meta | System statistics and metrics |
| `get_curiosity_ranking` | Meta | Domains ranked by learning potential |

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
- [x] **MCP Server**: 17 tools (safety + speed + reliability + meta), works with Claude Desktop, Cursor, any MCP client
- [x] **LangChain integration**: `CerebellumCallback` — one-line drop-in for any LangChain agent
- [x] **AutoRollback**: Cascade detection → automatic rollback plan computation
- [x] **Reliability Benchmark**: 7 scenarios, 71% waste prevention, 2.3-step avg detection delay
- [x] **OpenClaw Benchmark**: 5 desktop automation scenarios — 204x speedup, 5/5 cascade detection, 5/5 correct rollbacks
- [x] **OpenClaw Live Integration**: Real agent (kimi-k2.5) monitored via sidecar StepMonitor — Q&A, reasoning, tool use, cascade detection, skill learning all verified
- [x] **SkillStore Persistence**: Skills survive restarts — JSON+numpy save/load, auto-integrated with MCP server
- [x] 275 unit tests passing (+ 26 LLM-dependent deselected in CI)
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
