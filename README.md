# Digital Cerebellum 数字小脑

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18850778.svg)](https://doi.org/10.5281/zenodo.18850778)
[![PyPI](https://img.shields.io/pypi/v/digital-cerebellum)](https://pypi.org/project/digital-cerebellum/)

A cerebellar-inspired cognitive architecture for AI agents — prediction, correction, intuition, curiosity, and self-awareness.

一个受生物小脑启发的认知架构：为 AI Agent 提供预测、纠错、直觉、好奇心和自我意识。

---

## What is this?

Current AI agents are all "cerebral cortex" — slow, expensive, flexible reasoning. But biological intelligence runs on **two engines**: the cerebral cortex *and* the cerebellum.

The cerebellum holds ~50% of all brain neurons. It doesn't think — it **predicts, corrects, and automates**. It's why you can catch a ball without calculating parabolas, why a pianist's fingers move faster than conscious thought.

**Digital Cerebellum** brings this to AI agents:

- **< 10ms** prediction latency (vs 1-10s for LLM)
- **90%+ cost reduction** for routine decisions
- **Online learning** — gets better with every interaction
- **Gut feeling** — population divergence patterns trigger intuitive alarms
- **Curiosity** — actively seeks learnable domains for efficient exploration
- **Self-awareness** — knows what it's good at, defers what it can't handle

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

## Quick start

```bash
pip install digital-cerebellum
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

# Metacognitive self-report
report = cb.introspect()
print(report.to_prompt())
```

### As a complete brain (no framework needed)

```python
from digital_cerebellum import DigitalBrain

brain = DigitalBrain.from_yaml()
brain.register_tool("search", search_fn, "Search the web")

response = brain.think("What's the weather in Tokyo?")
print(response.text)           # LLM response
print(response.used_fast_path) # True if cerebellum handled it
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

**Exposed tools:**

| Tool | Description |
|------|-------------|
| `evaluate_tool_call` | Evaluate tool-call safety before execution |
| `evaluate_payment` | Assess payment/transaction risk |
| `feedback` | Provide post-execution feedback (drives learning) |
| `introspect` | Get metacognitive self-report |
| `get_stats` | System statistics and metrics |
| `get_curiosity_ranking` | Domains ranked by learning potential |

## Multiple microzones (Universal Cerebellar Transform)

```python
from digital_cerebellum import DigitalCerebellum
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.microzones.payment import PaymentMicrozone

cb = DigitalCerebellum()
cb.register_microzone(ToolCallMicrozone())
cb.register_microzone(PaymentMicrozone())

# Same engine, different domains:
cb.evaluate("tool_call", {"tool_name": "delete_file", "tool_params": {"path": "/etc/passwd"}})
cb.evaluate("payment", {"amount": 50000, "currency": "USD", "recipient": "unknown"})
```

Create your own microzone by subclassing `Microzone` — see `examples/`.

## Benchmark results

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
- [x] **Phase 2**: Signal processing — frequency filter, Golgi gate, state estimator + conditioner
- [x] **Phase 3**: Emergent cognition — somatic marker (intuition), curiosity drive, self-model (metacognition), component coordination
- [x] **MCP Server**: Works with Claude Desktop, Cursor, any MCP-compatible client
- [x] **Adaptive Phase 2**: Temporal Pattern Detector auto-bypasses Phase 2 on i.i.d. inputs
- [x] 132 unit tests passing
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
