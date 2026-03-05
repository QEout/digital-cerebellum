# Changelog

## 0.6.1 — 2026-03-05

### Added
- **ClawHub Skill release**: `openclaw skills install digital-cerebellum` — one command,
  zero config, transparent sidecar. Published as a unified organ, not a toolkit.
  - `skill.yaml` manifest (ClawHub metadata)
  - `SKILL.md` (unified cerebellum narrative for marketplace listing)
  - `run.sh` entry point (auto-creates venv, launches MCP server)
  - `examples/clawhub_demo.py` — end-to-end demo: learn → accelerate → catch cascade

### Changed
- **Narrative correction**: marketing no longer lists "StepMonitor + SkillStore + Fluid Memory"
  as separate products. The cerebellum is one organ; all 22 tools are the API surface of
  that organ, not 22 separate features.
- Updated README.md with ClawHub install section and unified tool table.
- Updated `docs/vision.md` Phase 8a description to reflect unified positioning.

## 0.6.0 — 2026-03-04

### Added
- **SkillStore persistence**: `save()` / `load()` methods for persisting learned skills
  across sessions. Uses portable JSON + numpy format. Integrated into `DigitalCerebellum.save()`
  and MCP server (auto-loads on startup, auto-saves on `learn_skill`).
- **OpenClaw live integration**: Real end-to-end tests with OpenClaw agent (kimi-k2.5)
  monitored by Digital Cerebellum as a transparent sidecar. Q&A, multi-step reasoning,
  tool use, cascade detection, and skill learning all verified in production.
- **A/B comparison experiments**: Quantitative proof of product value — same agent,
  same tasks, with vs without cerebellum:
  - Repeated tasks: 3x faster, 66% fewer tokens (skills bypass LLM from round 2)
  - Error cascade: 4 steps saved, 65% time reduction (early stopping)
  - Mixed workload: 49% token savings, 50% time savings, accuracy preserved
- **Sidecar monitoring example** (`examples/sidecar_monitor.py`): Copy-paste quickstart
  for the most impactful integration pattern validated by A/B testing.

- **Fluid Memory MCP tools**: `store_memory` and `retrieve_memories` — episodic memory
  with semantic retrieval and biological reconsolidation. `monitor_after_step` auto-stores
  outcomes; `monitor_before_step` injects relevant memories into predictions.
- **Sleep Cycle MCP tool**: `run_sleep_cycle` — offline consolidation (decay weak memories,
  promote strong ones, merge/prune skills). Call between sessions for lean memory.
- **Gut Feeling MCP tool**: `get_gut_feeling` — somatic marker intuition. Returns pre-rational
  risk assessment (valence, intensity, alarm/positive/uneasy) based on past experience patterns.
- **Exploration Suggestions MCP tool**: `get_exploration_suggestions` — curiosity-driven
  actionable recommendations ("explore domain X", "investigate Y", "abandon Z").
- **Multi-step skill sequences**: `learn_skill` now accepts `tool_calls` parameter for
  storing complete action sequences; `match_skill` returns `tool_calls` for replay.

### Changed
- **MCP server expanded to 22 tools** (was 17): +5 new tools for memory, sleep, intuition,
  exploration, and multi-step skills. All 5 biological cerebellum capabilities now exposed.
- **CI hardened**: Added `conftest.py` with auto-markers for LLM-dependent tests.
  CI now runs `pytest -m "not llm and not slow"` to avoid hangs on model loading.
- **pytest markers**: `llm`, `slow`, `integration` markers defined in `pyproject.toml`.
- Version bumped to 0.6.0.

## 0.5.0 — 2026-03-04

### Added
- **AutoRollback**: When StepMonitor detects an error cascade, it now automatically
  computes a `RollbackPlan` — which step to revert to, what failed, and a
  human-readable recovery recommendation.
- **LangChain integration**: `CerebellumCallback` wraps any LangChain agent with
  cerebellum monitoring. Raises `CerebellumPause` on cascade with attached rollback
  plan. Install with `pip install digital-cerebellum[langchain]`.
- **MCP tool `monitor_rollback_plan`**: Exposes the rollback plan over MCP (17 tools total).
- **Combined demo** (`examples/fast_and_reliable_demo.py`): End-to-end demonstration
  of SkillStore (122x speedup) and StepMonitor (cascade detection + auto-rollback)
  working together in a customer support scenario.

### Changed
- MCP server now exposes 17 tools (was 16), organized by category (Safety, Speed,
  Reliability, Learning, Meta).
- `StepVerdict.details` now includes a `rollback_plan` dict when cascade is detected.
- Version bumped to 0.5.0.

## 0.4.0

### Added
- **StepMonitor**: Universal agent step monitoring protocol with predictive error
  interception. Framework-agnostic — works with any agent that can describe actions,
  states, and outcomes as text or vectors.
- **StepForwardModel**: Neural forward model that predicts outcomes from (state, action)
  pairs. Online learning — gets better with every step.
- **ErrorCascadeDetector**: Tracks consecutive prediction errors to detect cascading
  failures before they compound.
- **FailureMemory**: Records past failure patterns and provides preemptive warnings
  when similar situations arise.
- **Reliability Benchmark**: 7-scenario benchmark comparing agent performance with
  and without StepMonitor: 100% cascade detection, 71% waste prevention.
- **Save/Load for StepMonitor**: Persist forward model weights and failure memory
  across sessions.
- MCP server expanded to 16 tools (4 new StepMonitor tools).

## 0.3.0

### Added
- **MicroOpEngine**: Real-time continuous control at 285Hz for fine-grained
  environments (games, robotics, desktop automation).
- **DigitalBrain**: Unified LLM (cortex) + cerebellum architecture with automatic
  skill acquisition and tool replay.
- **SkillStore**: Procedural memory for learned skills — enables 100x speedup by
  bypassing the LLM for known patterns.

## 0.2.0

### Added
- **Emergence module**: SomaticMarker (gut feelings), CuriosityDrive (exploration
  ranking), SelfModel (metacognitive self-reports).
- **Microzones**: Specialized evaluation domains (tool call, payment, shell command,
  file operation, API call).

## 0.1.0

### Added
- Initial release with core prediction-correction engine, feature encoder, online
  learning, confidence routing, and basic MCP server.
