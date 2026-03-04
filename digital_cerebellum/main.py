"""
Digital Cerebellum — General-Purpose Prediction-Correction Engine.

A universal cerebellar computation module that can serve ANY domain
through pluggable microzones.  The core pipeline is always the same::

    event → encode → pattern separate → predict → route → learn

What differs per domain is HOW inputs are encoded and WHAT the outputs mean.

Usage::

    from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig
    from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

    cb = DigitalCerebellum()
    cb.register_microzone(ToolCallMicrozone())

    # Generic API — works with any microzone:
    result = cb.evaluate("tool_call", {
        "tool_name": "send_email",
        "tool_params": {"to": "alice", "body": "hi"},
    })

    # Convenience shortcut for the tool_call microzone:
    result = cb.evaluate_tool_call("send_email", {"to": "alice"})
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from digital_cerebellum.core.feature_encoder import FeatureEncoder
from digital_cerebellum.core.pattern_separator import PatternSeparator
from digital_cerebellum.core.prediction_engine import EngineConfig, PredictionEngine
from digital_cerebellum.core.error_comparator import ErrorComparator
from digital_cerebellum.core.online_learner import OnlineLearner
from digital_cerebellum.core.microzone import LearningSignal, Microzone
from digital_cerebellum.core.types import (
    ErrorType,
    PredictionOutput,
    RouteDecision,
    ToolCallEvaluation,
)
from digital_cerebellum.routing.decision_router import DecisionRouter
from digital_cerebellum.memory.fluid_memory import FluidMemory
from digital_cerebellum.core.types import MemorySlot

log = logging.getLogger(__name__)


@dataclass
class CerebellumConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    rff_dim: int = 4096
    rff_gamma: float = 1.0
    rff_sparsity: float = 0.1
    num_heads: int = 4           # K
    action_dim: int = 128
    outcome_dim: int = 128
    confidence_temperature: float = 1.0
    learning_rate: float = 0.01
    ewc_lambda: float = 400.0
    task_lr: float = 0.005
    threshold_high: float = 0.95
    threshold_low: float = 0.5
    save_dir: str = ".digital-cerebellum"

    # Phase 2 components
    enable_frequency_filter: bool = False
    frequency_alpha: float = 0.1
    enable_golgi_gate: bool = False
    golgi_target_sparsity: float = 0.1
    enable_state_estimator: bool = False
    state_dim: int = 64

    # Phase 3 emergence
    enable_somatic_marker: bool = False
    enable_curiosity_drive: bool = False
    enable_self_model: bool = False

    # LLM (slow path)
    llm_model: str = "qwen3.5-flash"
    llm_api_key: str | None = None
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "CerebellumConfig":
        """Load config from YAML, with config.local.yaml override."""
        import yaml

        cfg = cls()

        for p in [Path(path), Path("config.local.yaml")]:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            llm = data.get("llm", {})
            if llm.get("model"):
                cfg.llm_model = llm["model"]
            if llm.get("api_key"):
                cfg.llm_api_key = llm["api_key"]
            if llm.get("base_url"):
                cfg.llm_base_url = llm["base_url"]

            ps = data.get("pattern_separator", {})
            if ps.get("rff_dim"):
                cfg.rff_dim = ps["rff_dim"]
            if ps.get("gamma"):
                cfg.rff_gamma = ps["gamma"]
            if ps.get("sparsity"):
                cfg.rff_sparsity = ps["sparsity"]

            pe = data.get("prediction_engine", {})
            if pe.get("num_heads"):
                cfg.num_heads = pe["num_heads"]
            if pe.get("action_dim"):
                cfg.action_dim = pe["action_dim"]
            if pe.get("outcome_dim"):
                cfg.outcome_dim = pe["outcome_dim"]
            if pe.get("temperature"):
                cfg.confidence_temperature = pe["temperature"]

            lr = data.get("learning", {})
            if lr.get("lr"):
                cfg.learning_rate = lr["lr"]
            if lr.get("ewc_lambda"):
                cfg.ewc_lambda = lr["ewc_lambda"]
            if lr.get("task_lr"):
                cfg.task_lr = lr["task_lr"]

            rt = data.get("router", {})
            if rt.get("threshold_high"):
                cfg.threshold_high = rt["threshold_high"]
            if rt.get("threshold_low"):
                cfg.threshold_low = rt["threshold_low"]

            emb = data.get("embedding", {})
            if emb.get("model"):
                cfg.embedding_model = emb["model"]

            st = data.get("storage", {})
            if st.get("save_dir"):
                cfg.save_dir = st["save_dir"]

            # Phase 2 components
            p2 = data.get("phase2", {})
            if "frequency_filter" in p2:
                cfg.enable_frequency_filter = p2["frequency_filter"]
            if "frequency_alpha" in p2:
                cfg.frequency_alpha = p2["frequency_alpha"]
            if "golgi_gate" in p2:
                cfg.enable_golgi_gate = p2["golgi_gate"]
            if "golgi_target_sparsity" in p2:
                cfg.golgi_target_sparsity = p2["golgi_target_sparsity"]
            if "state_estimator" in p2:
                cfg.enable_state_estimator = p2["state_estimator"]
            if "state_dim" in p2:
                cfg.state_dim = p2["state_dim"]

            # Phase 3 emergence
            p3 = data.get("phase3", {})
            if "somatic_marker" in p3:
                cfg.enable_somatic_marker = p3["somatic_marker"]
            if "curiosity_drive" in p3:
                cfg.enable_curiosity_drive = p3["curiosity_drive"]
            if "self_model" in p3:
                cfg.enable_self_model = p3["self_model"]

        return cfg


class DigitalCerebellum:
    """
    General-purpose cerebellar engine.

    The core pipeline (encode → separate → predict → route → learn) is
    domain-agnostic.  Domain-specific behaviour comes from registered
    microzones.
    """

    def __init__(self, cfg: CerebellumConfig | None = None):
        self.cfg = cfg or CerebellumConfig()

        # ① Feature encoder (mossy fibre) — shared across microzones
        self.encoder = FeatureEncoder(self.cfg.embedding_model)
        input_dim = self.encoder.output_dim

        # ② Pattern separator (granule cell layer) — universal
        self.separator = PatternSeparator(
            input_dim=input_dim,
            rff_dim=self.cfg.rff_dim,
            gamma=self.cfg.rff_gamma,
            sparsity=self.cfg.rff_sparsity,
        )

        # ③ Prediction engine (Purkinje population) — universal
        engine_cfg = EngineConfig(
            rff_dim=self.cfg.rff_dim,
            action_dim=self.cfg.action_dim,
            outcome_dim=self.cfg.outcome_dim,
            num_heads=self.cfg.num_heads,
            temperature=self.cfg.confidence_temperature,
        )
        self.engine = PredictionEngine(engine_cfg)

        # ④ Decision router (DCN)
        self.router = DecisionRouter(
            threshold_high=self.cfg.threshold_high,
            threshold_low=self.cfg.threshold_low,
        )

        # ⑤ Error comparator (climbing fibre)
        self.comparator = ErrorComparator()

        # ⑥ Online learner (synaptic plasticity)
        self.learner = OnlineLearner(
            self.engine,
            lr=self.cfg.learning_rate,
            ewc_lambda=self.cfg.ewc_lambda,
            task_lr=self.cfg.task_lr,
        )

        # Phase 2: Frequency filter (molecular layer interneurons)
        self._freq_filter = None
        if self.cfg.enable_frequency_filter:
            from digital_cerebellum.core.frequency_filter import FrequencyFilter
            self._freq_filter = FrequencyFilter(
                dim=self.cfg.rff_dim, alpha=self.cfg.frequency_alpha, mode="gate",
            )

        # Phase 2: Golgi feedback gate
        self._golgi_gate = None
        if self.cfg.enable_golgi_gate:
            from digital_cerebellum.core.golgi_gate import GolgiGate
            self._golgi_gate = GolgiGate(
                dim=self.cfg.rff_dim,
                target_sparsity=self.cfg.golgi_target_sparsity,
            )

        # Phase 2: State estimator
        self._state_estimator = None
        if self.cfg.enable_state_estimator:
            from digital_cerebellum.core.state_estimator import StateEstimator
            self._state_estimator = StateEstimator(
                state_dim=self.cfg.state_dim,
            )
            self.engine.enable_state_conditioning(self.cfg.state_dim)

        # Phase 3: Somatic marker (gut feeling)
        self._somatic_marker = None
        if self.cfg.enable_somatic_marker:
            from digital_cerebellum.emergence.somatic_marker import SomaticMarker
            self._somatic_marker = SomaticMarker()

        # Phase 3: Curiosity drive (intrinsic motivation)
        self._curiosity_drive = None
        if self.cfg.enable_curiosity_drive:
            from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive
            self._curiosity_drive = CuriosityDrive()

        # Phase 3: Self-model (metacognition)
        self._self_model = None
        if self.cfg.enable_self_model:
            from digital_cerebellum.emergence.self_model import SelfModel
            self._self_model = SelfModel()

        # ⑦ Fluid memory + sleep cycle
        self.memory = FluidMemory()
        self._sleep_cycle: _SleepCycle | None = None

        # ⑧ Task consolidation pipeline
        self._consolidation: _ConsolidationPipeline | None = None

        # ⑨ Cortex interface (lazy)
        self._cortex = None

        # Microzone registry
        self._microzones: dict[str, Microzone] = {}

        # Tracking
        self._pending: dict[str, dict] = {}
        self._history: list[dict] = []
        self._step = 0

    # ==================================================================
    # Lazy-init Phase 1 components
    # ==================================================================
    @property
    def sleep_cycle(self):
        if self._sleep_cycle is None:
            from digital_cerebellum.memory.sleep_cycle import SleepCycle as _SC
            self._sleep_cycle = _SC()
        return self._sleep_cycle

    @property
    def consolidation(self):
        if self._consolidation is None:
            from digital_cerebellum.cortex.consolidation import ConsolidationPipeline as _CP
            self._consolidation = _CP()
        return self._consolidation

    def sleep(self):
        """Run one sleep cycle (offline memory maintenance)."""
        return self.sleep_cycle.run(self.memory)

    # ==================================================================
    # Microzone registration
    # ==================================================================
    def register_microzone(self, mz: Microzone) -> None:
        """
        Register a new microzone (domain).

        Creates the corresponding task heads in the prediction engine.
        """
        self._microzones[mz.name] = mz
        for th_cfg in mz.task_heads():
            self.engine.register_task_head(
                th_cfg.name, th_cfg.output_dim, th_cfg.activation,
            )
        self.learner._sync_task_optimizers()
        log.info("Registered microzone '%s' with task heads: %s",
                 mz.name, [h.name for h in mz.task_heads()])

    # ==================================================================
    # Generic evaluation API
    # ==================================================================
    def evaluate(
        self,
        microzone_name: str,
        payload: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        """
        Universal evaluation entry point.

        Works with any registered microzone.  Returns a domain-specific
        dict whose schema is defined by the microzone.
        """
        mz = self._microzones.get(microzone_name)
        if mz is None:
            raise ValueError(
                f"Unknown microzone '{microzone_name}'. "
                f"Registered: {list(self._microzones)}"
            )

        event_id = uuid.uuid4().hex
        t0 = time.time()
        self._step += 1

        # ① Encode — microzone formats, shared encoder embeds
        text = mz.format_input(payload, context)
        feature_vec = self.encoder.encode_tool_call_raw(text)

        # ② Pattern separate
        z = self.separator.encode_event(feature_vec)

        # Phase 2: Golgi feedback gate (adaptive sparsity)
        if self._golgi_gate is not None:
            with torch.no_grad():
                z_t = torch.from_numpy(z).float()
                z_t = self._golgi_gate(z_t)
                z = z_t.numpy()

        # Phase 2: Frequency filter (low/high band decomposition)
        if self._freq_filter is not None:
            with torch.no_grad():
                z_t = torch.from_numpy(z).float()
                z_t = self._freq_filter(z_t)
                z = z_t.numpy()

        # ③ Predict (with optional state conditioning)
        if self._state_estimator is not None:
            with torch.no_grad():
                state_vec = self._state_estimator()
                prediction = self.engine.forward(
                    torch.from_numpy(z).float(), state=state_vec,
                )
        else:
            prediction = self.engine.predict_numpy(z)

        # Phase 3: Somatic marker — gut feeling from head divergence pattern
        gut_feeling = None
        if self._somatic_marker is not None:
            from digital_cerebellum.emergence.somatic_marker import GutFeeling
            gut_feeling = self._somatic_marker.feel(
                prediction.head_predictions, domain=microzone_name,
            )

        # ④ Route (gut feeling can override)
        routing = self.router.route(prediction)

        if gut_feeling is not None and gut_feeling.should_override:
            if routing.decision == RouteDecision.FAST:
                routing = routing.__class__(
                    decision=RouteDecision.SLOW,
                    confidence=prediction.confidence,
                    reason=f"gut_override: {gut_feeling.label} "
                           f"(v={gut_feeling.valence:.2f}, i={gut_feeling.intensity:.2f})",
                )

        latency_ms = (time.time() - t0) * 1000

        if routing.decision == RouteDecision.FAST:
            result = mz.fast_path_evaluate(payload, prediction)
            result["_route"] = "fast"
            result["_latency_ms"] = latency_ms
        elif routing.decision == RouteDecision.SHADOW:
            llm_response = self._call_cortex_generic(mz, payload, context, prediction)
            result = mz.slow_path_evaluate(payload, prediction, llm_response)
            result["_route"] = "shadow"
            result["_latency_ms"] = latency_ms
            self._learn_from_response(mz, z, text, llm_response)
        else:
            llm_response = self._call_cortex_generic(mz, payload, context, prediction)
            result = mz.slow_path_evaluate(payload, prediction, llm_response)
            result["_route"] = "slow"
            result["_latency_ms"] = latency_ms
            self._learn_from_response(mz, z, text, llm_response)

        result["_step"] = self._step
        result["_event_id"] = event_id

        # Phase 3: Attach gut feeling
        if gut_feeling is not None:
            result["_gut_feeling"] = {
                "valence": gut_feeling.valence,
                "intensity": gut_feeling.intensity,
                "label": gut_feeling.label,
            }

        # Phase 3: Curiosity assessment
        if self._curiosity_drive is not None:
            spe_mag = float(np.linalg.norm(
                prediction.action_embedding - prediction.outcome_embedding
            )) if prediction.action_embedding is not None else 0.0
            curiosity = self._curiosity_drive.assess(
                domain=microzone_name,
                error=spe_mag,
                feature_vec=feature_vec,
            )
            result["_curiosity"] = {
                "novelty": curiosity.novelty,
                "learning_progress": curiosity.learning_progress,
                "intrinsic_reward": curiosity.intrinsic_reward,
                "recommendation": curiosity.recommendation,
            }

        # Phase 2: State estimator tracking
        if self._state_estimator is not None:
            tool_name = payload.get("tool_name", microzone_name)
            risk = prediction.task_outputs.get("safety_score", 0.0)
            self._state_estimator.record_event(
                tool_name=tool_name,
                route=result.get("_route", "slow"),
                confidence=prediction.confidence,
                risk_score=risk,
            )

        # Store for feedback
        self._pending[event_id] = {
            "z": z,
            "prediction": prediction,
            "routing": routing.decision.value,
            "microzone": microzone_name,
            "payload": payload,
            "timestamp": t0,
            "confidence": prediction.confidence,
        }

        # Memory
        self._store_memory(text, feature_vec)

        # History
        self._history.append({
            "step": self._step,
            "event_id": event_id,
            "microzone": microzone_name,
            "route": routing.decision.value,
            "confidence": prediction.confidence,
            "task_outputs": dict(prediction.task_outputs),
            "latency_ms": latency_ms,
        })

        log.info(
            "step=%d zone=%s route=%s conf=%.3f tasks=%s latency=%.1fms",
            self._step, microzone_name, routing.decision.value,
            prediction.confidence, prediction.task_outputs, latency_ms,
        )

        return result

    # ==================================================================
    # Backward-compatible tool_call convenience API
    # ==================================================================
    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        context: str = "",
    ) -> ToolCallEvaluation:
        """
        Convenience wrapper for the tool_call microzone.

        Auto-registers ToolCallMicrozone if not already present.
        """
        if "tool_call" not in self._microzones:
            from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
            self.register_microzone(ToolCallMicrozone())

        result = self.evaluate("tool_call", {
            "tool_name": tool_name,
            "tool_params": tool_params,
        }, context)

        return ToolCallEvaluation(
            safe=result.get("safe", True),
            risk_type=result.get("risk_type", "none"),
            confidence=result.get("confidence", 0.0),
            predicted_outcome=None,
            details=f"{result.get('_route', '?')} path | "
                    f"safety={result.get('safety_score', 0):.3f} | "
                    f"step {result.get('_step', 0)}",
        )

    # ==================================================================
    # Feedback
    # ==================================================================
    def feedback(self, event_id: str, success: bool):
        """Provide post-execution feedback (RPE) for any microzone."""
        ctx = self._pending.pop(event_id, None)
        if ctx is None:
            return
        actual = 1.0 if success else -1.0
        expected = ctx.get("confidence", 0.5) if isinstance(ctx, dict) else 0.5
        rpe = self.comparator.compute_reward_error(expected, actual, event_id)
        self.router.update_from_reward(rpe)

        # Phase 3: Record somatic marker (valence pattern)
        prediction = ctx.get("prediction")
        if self._somatic_marker is not None and prediction is not None:
            self._somatic_marker.record(
                prediction.head_predictions,
                valence=actual,
                domain=ctx.get("microzone", ""),
            )

        # Phase 3: Self-model observation
        if self._self_model is not None:
            self._self_model.record(
                domain=ctx.get("microzone", "unknown"),
                correct=success,
                confidence=ctx.get("confidence", 0.5),
                route=ctx.get("routing", "slow"),
            )

    # ==================================================================
    # Internal — domain-agnostic
    # ==================================================================
    def _get_cortex(self):
        if self._cortex is None:
            from digital_cerebellum.cortex.cortex_interface import CortexInterface
            self._cortex = CortexInterface(
                model=self.cfg.llm_model,
                api_key=self.cfg.llm_api_key,
                base_url=self.cfg.llm_base_url,
            )
        return self._cortex

    def _call_cortex_generic(
        self,
        mz: Microzone,
        payload: dict,
        context: str,
        prediction: PredictionOutput,
    ) -> dict:
        """Ask the LLM to evaluate, using the microzone's prompt template."""
        req = mz.build_slow_path_request(payload, context, prediction)
        cortex = self._get_cortex()
        return cortex.evaluate_raw(req.system_prompt, req.user_message)

    def _learn_from_response(
        self,
        mz: Microzone,
        z: np.ndarray,
        text: str,
        llm_response: dict,
    ):
        """Extract training signal from the LLM response and learn."""
        signal: LearningSignal = mz.parse_slow_path_response(llm_response)

        if not signal.outcome_text:
            return

        action_emb = self.encoder.encode_text(text)
        outcome_emb = self.encoder.encode_text(signal.outcome_text)

        action_emb = self._fit_dim(action_emb, self.cfg.action_dim)
        outcome_emb = self._fit_dim(outcome_emb, self.cfg.outcome_dim)

        loss = self.learner.learn(
            z, action_emb, outcome_emb,
            task_labels=signal.task_labels,
        )
        log.debug("online learning step loss=%.4f", loss)

    def _store_memory(self, text: str, feature_vec: np.ndarray):
        emb = feature_vec[:384] if len(feature_vec) >= 384 else feature_vec
        slot = MemorySlot(
            content=text,
            embedding=emb.copy(),
            layer="short_term",
        )
        self.memory.store(slot)

    @staticmethod
    def _fit_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
        if len(vec) == target_dim:
            return vec
        if len(vec) > target_dim:
            return vec[:target_dim]
        return np.pad(vec, (0, target_dim - len(vec)))

    # ==================================================================
    # Persistence
    # ==================================================================
    def save(self, path: str | Path | None = None):
        path = Path(path or self.cfg.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "engine": self.engine.state_dict(),
            "separator": self.separator.state_dict(),
            "router_stats": self.router.stats,
            "step": self._step,
            "history": self._history,
        }, path / "checkpoint.pt")
        log.info("Saved checkpoint to %s", path)

    def load(self, path: str | Path | None = None):
        path = Path(path or self.cfg.save_dir)
        ckpt_path = path / "checkpoint.pt"
        if not ckpt_path.exists():
            log.info("No checkpoint found at %s", ckpt_path)
            return
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.engine.load_state_dict(ckpt["engine"])
        self.separator.load_state_dict(ckpt["separator"])
        self._step = ckpt.get("step", 0)
        self._history = ckpt.get("history", [])
        log.info("Loaded checkpoint from %s (step %d)", path, self._step)

    # ==================================================================
    # Diagnostics
    # ==================================================================
    @property
    def somatic_marker(self):
        """Access the somatic marker (lazy init if not configured)."""
        if self._somatic_marker is None:
            from digital_cerebellum.emergence.somatic_marker import SomaticMarker
            self._somatic_marker = SomaticMarker()
        return self._somatic_marker

    @property
    def curiosity_drive(self):
        """Access the curiosity drive (lazy init if not configured)."""
        if self._curiosity_drive is None:
            from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive
            self._curiosity_drive = CuriosityDrive()
        return self._curiosity_drive

    @property
    def self_model(self):
        """Access the self-model (lazy init if not configured)."""
        if self._self_model is None:
            from digital_cerebellum.emergence.self_model import SelfModel
            self._self_model = SelfModel()
        return self._self_model

    def introspect(self, domain: str | None = None):
        """Generate a metacognitive self-report."""
        return self.self_model.introspect(domain)

    @property
    def stats(self) -> dict:
        routes = {"fast": 0, "shadow": 0, "slow": 0}
        for h in self._history:
            routes[h["route"]] = routes.get(h["route"], 0) + 1
        result = {
            "step": self._step,
            "microzones": list(self._microzones),
            "task_heads": list(self.engine.task_heads.keys()),
            "routes": routes,
            "router": self.router.stats,
            "memory": self.memory.stats,
        }
        if self._somatic_marker is not None:
            result["somatic_marker"] = self._somatic_marker.stats
        if self._curiosity_drive is not None:
            result["curiosity"] = self._curiosity_drive.stats
        if self._self_model is not None:
            result["self_model"] = self._self_model.stats
        return result
