# Digital Cerebellum 数字小脑

A cerebellar-inspired prediction-correction engine for LLM Agents.

一个受生物小脑启发的轻量级预测-校正系统，作为大语言模型的"另一半大脑"。

---

## What is this?

Current AI agents (ChatGPT, OpenClaw, LangChain, etc.) are all "cerebral cortex" — slow, expensive, flexible reasoning. But biological intelligence runs on **two engines**: the cerebral cortex *and* the cerebellum.

The cerebellum holds ~50% of all brain neurons. It doesn't think — it **predicts, corrects, and automates**. It's why you can catch a ball without calculating parabolas, why a pianist's fingers move faster than conscious thought.

**Digital Cerebellum** brings this missing half to AI agents:

- **< 10ms** prediction latency (vs 1-10s for LLM)
- **Near-zero cost** for routine decisions (vs $0.01+ per LLM call)
- **Online learning** — gets better with every interaction
- **Uncertainty quantification** — knows what it doesn't know

## How it works

```
LLM Agent prepares a tool call
        │
        ▼
┌─ Digital Cerebellum ──────────────────────┐
│                                            │
│  Feature Encoder     (mossy fibres)        │
│       │                                    │
│  Pattern Separator   (granule cells, RFF)  │
│       │                                    │
│  K-Head Predictor    (Purkinje population) │
│       │                                    │
│  Decision Router     (deep cerebellar      │
│       │               nuclei)              │
│       ▼                                    │
│  High confidence → ALLOW  (fast path)      │
│  Low confidence  → ASK LLM (slow path)    │
│       │                                    │
│  Result → Error Signal → Online Learning   │
└────────────────────────────────────────────┘
```

Every biological component maps to a neuroscience-validated mechanism:

| Biology | Digital | Reference |
|---------|---------|-----------|
| Granule cell layer | Random Fourier Features | Bhalla 2022, Frontiers Comp Neuro |
| Purkinje population | K independent linear heads | 2025 J.Neuroscience |
| Climbing fibres | 3 error channels (SPE/TPE/RPE) | 2025 Nature Communications |
| Deep cerebellar nuclei | Adaptive threshold router | 2025 Frontiers |
| Cortico-cerebellar loop | Task consolidation pipeline | Boven et al. 2024 Nature Comms |

Full scientific audit with honest limitations: [`docs/architecture.md`](docs/architecture.md) §15.

## Quick start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/digital-cerebellum.git
cd digital-cerebellum

# Install
pip install -r requirements.txt

# Configure (fill in your API key)
cp config.yaml config.local.yaml
# Edit config.local.yaml → llm.api_key

# Use
python -c "
from src.main import DigitalCerebellum, CerebellumConfig
cfg = CerebellumConfig.from_yaml()
cb = DigitalCerebellum(cfg)
result = cb.evaluate_tool_call('send_email', {'to': 'alice', 'body': 'hello'})
print(result)
"
```

Supports any OpenAI-compatible LLM: Qwen, GPT, Claude, Ollama, etc.

## Project status

**Phase 0 — Core engine + validation** (in progress)

- [x] Pattern separator (RFF with top-k sparsification)
- [x] K-head prediction engine (population coding → emergent confidence)
- [x] 3-channel error comparator (SPE implemented; TPE/RPE interfaces ready)
- [x] Online learner (SGD + EWC, multi-site interface)
- [x] Adaptive decision router (RPE-driven threshold)
- [x] Fluid memory v0 (decay + reconsolidation)
- [x] Cortex interface (OpenAI-compatible LLM)
- [x] Main pipeline (`DigitalCerebellum` class)
- [x] 23 unit tests passing
- [ ] End-to-end validation with real tool-call data
- [ ] Learning curve analysis (accuracy vs sample count)
- [ ] Benchmark: population coding confidence vs sigmoid confidence

See [roadmap](docs/architecture.md#十三应用场景与路线图) for Phase 1-3.

## Architecture

Two documents cover everything:

- **[`docs/architecture.md`](docs/architecture.md)** — Biological mappings, system design, neuroscience corrections (v2), digital life panorama, application scenarios, scientific audit, competitive landscape, honest boundaries
- **[`docs/implementation.md`](docs/implementation.md)** — Component-level technical details, algorithms, dependencies, file structure

## Key design decisions

**Population coding, not sigmoid confidence.** K=4 independent prediction heads. Confidence emerges from agreement — when heads disagree, uncertainty is real, not a learned artifact.

**Three error channels, not one.** Sensory prediction error updates the predictor. Temporal error updates the rhythm system. Reward error updates the router. Each drives different learning.

**Continuous prediction, not classification.** The engine predicts *outcomes* (continuous vectors), not labels. The safe/unsafe decision derives from continuous confidence. This satisfies the cerebellum's [continuity constraint](https://arxiv.org/abs/2509.09818) (Tsay & Ivry 2025).

**EWC is an engineering approximation.** Biological cerebellum prevents catastrophic forgetting via systems consolidation (cerebellum→cortex transfer), not weight regularization. Our task consolidation pipeline (Phase 1) is the more faithful mechanism. We're honest about this.

## Run tests

```bash
pytest tests/ -v
```

## License

MIT

## Citation

If you use this work in research:

```bibtex
@software{digital_cerebellum_2026,
  title={Digital Cerebellum: A Cerebellar-Inspired Prediction-Correction Engine for LLM Agents},
  year={2026},
  url={https://github.com/YOUR_USERNAME/digital-cerebellum}
}
```
