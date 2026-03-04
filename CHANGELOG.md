# Changelog

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
