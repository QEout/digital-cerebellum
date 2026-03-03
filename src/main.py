"""
Digital Cerebellum — Phase 0 main pipeline.

Wires together all components into the core prediction-correction loop
for LLM tool-call pre-evaluation.

Usage::

    from src.main import DigitalCerebellum

    cb = DigitalCerebellum()

    # Agent is about to call a tool:
    result = cb.evaluate_tool_call("send_email", {"to": "alice", "body": "hi"})
    print(result)
    # ToolCallEvaluation(safe=True, risk_type='none', confidence=0.87, ...)

    # After execution, feed back the actual result:
    cb.feedback(result.event_id, success=True)
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

from src.core.feature_encoder import FeatureEncoder
from src.core.pattern_separator import PatternSeparator
from src.core.prediction_engine import EngineConfig, PredictionEngine
from src.core.error_comparator import ErrorComparator
from src.core.online_learner import OnlineLearner
from src.core.types import (
    ErrorType,
    PredictionOutput,
    RouteDecision,
    ToolCallEvaluation,
)
from src.routing.decision_router import DecisionRouter
from src.memory.fluid_memory import FluidMemory
from src.core.types import MemorySlot

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
    threshold_high: float = 0.95
    threshold_low: float = 0.5
    save_dir: str = ".digital-cerebellum"

    # LLM (slow path)
    llm_model: str = "qwen-turbo"
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

        return cfg


class DigitalCerebellum:
    """The main cerebellum pipeline: perceive → separate → predict → route."""

    def __init__(self, cfg: CerebellumConfig | None = None):
        self.cfg = cfg or CerebellumConfig()

        # ① Feature encoder (mossy fibre)
        self.encoder = FeatureEncoder(self.cfg.embedding_model)
        input_dim = self.encoder.output_dim  # ~388

        # ② Pattern separator (granule cell layer)
        self.separator = PatternSeparator(
            input_dim=input_dim,
            rff_dim=self.cfg.rff_dim,
            gamma=self.cfg.rff_gamma,
            sparsity=self.cfg.rff_sparsity,
        )

        # ③ Prediction engine (Purkinje population)
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
        )

        # ⑦ Fluid memory
        self.memory = FluidMemory()

        # ⑧ Cortex interface (lazy — only created when slow path needed)
        self._cortex = None

        # Tracking
        self._pending: dict[str, dict] = {}   # event_id → context for feedback
        self._history: list[dict] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        context: str = "",
    ) -> ToolCallEvaluation:
        """
        Core pipeline: evaluate whether a tool call is safe.

        Returns a ToolCallEvaluation with safe/risk/confidence.
        """
        event_id = uuid.uuid4().hex
        t0 = time.time()
        self._step += 1

        # ① Encode
        feature_vec = self.encoder.encode_tool_call(tool_name, tool_params, context)

        # ② Pattern separate
        z = self.separator.encode_event(feature_vec)

        # ③ Predict
        prediction = self.engine.predict_numpy(z)

        # ④ Route
        routing = self.router.route(prediction)
        latency_ms = (time.time() - t0) * 1000

        if routing.decision == RouteDecision.FAST:
            # Fast path — trust the cerebellum
            safe = bool(prediction.confidence > 0.7)
            risk_type = "none" if safe else "low_confidence"
            evaluation = ToolCallEvaluation(
                safe=safe,
                risk_type=risk_type,
                confidence=prediction.confidence,
                predicted_outcome=prediction.outcome_embedding,
                details=f"fast path | {latency_ms:.1f}ms | step {self._step}",
            )
        elif routing.decision == RouteDecision.SHADOW:
            # Shadow path — use LLM but also learn from comparison
            llm_result = self._call_cortex(tool_name, tool_params, context, prediction)
            evaluation = ToolCallEvaluation(
                safe=llm_result.get("safe", True),
                risk_type=llm_result.get("risk_type", "none"),
                confidence=prediction.confidence,
                predicted_outcome=prediction.outcome_embedding,
                details=f"shadow path | {latency_ms:.1f}ms | step {self._step}",
            )
            # Learn from the shadow comparison
            self._learn_from_llm(z, tool_name, tool_params, llm_result)
        else:
            # Slow path — fully delegate to LLM
            llm_result = self._call_cortex(tool_name, tool_params, context, prediction)
            evaluation = ToolCallEvaluation(
                safe=llm_result.get("safe", True),
                risk_type=llm_result.get("risk_type", "none"),
                confidence=prediction.confidence,
                predicted_outcome=prediction.outcome_embedding,
                details=f"slow path | step {self._step}",
            )
            self._learn_from_llm(z, tool_name, tool_params, llm_result)

        # Store for later feedback
        self._pending[event_id] = {
            "z": z,
            "prediction": prediction,
            "routing": routing.decision.value,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "timestamp": t0,
        }

        # Record to memory
        self._store_memory(tool_name, tool_params, feature_vec)

        # Record history
        self._history.append({
            "step": self._step,
            "event_id": event_id,
            "tool_name": tool_name,
            "route": routing.decision.value,
            "confidence": prediction.confidence,
            "safe": evaluation.safe,
            "latency_ms": latency_ms,
        })

        log.info(
            "step=%d tool=%s route=%s conf=%.3f safe=%s latency=%.1fms",
            self._step, tool_name, routing.decision.value,
            prediction.confidence, evaluation.safe, latency_ms,
        )

        return evaluation

    # ------------------------------------------------------------------
    def feedback(self, event_id: str, success: bool):
        """
        Provide feedback after a tool call has executed.

        Generates RPE and (if we have the pending context) SPE.
        """
        ctx = self._pending.pop(event_id, None)
        if ctx is None:
            return

        rpe_value = 1.0 if success else -1.0
        rpe = self.comparator.compute_reward_error(rpe_value, event_id)
        self.router.update_from_reward(rpe)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_cortex(self):
        if self._cortex is None:
            from src.cortex.cortex_interface import CortexInterface
            self._cortex = CortexInterface(
                model=self.cfg.llm_model,
                api_key=self.cfg.llm_api_key,
                base_url=self.cfg.llm_base_url,
            )
        return self._cortex

    def _call_cortex(
        self,
        tool_name: str,
        tool_params: dict,
        context: str,
        prediction: PredictionOutput,
    ) -> dict:
        cortex = self._get_cortex()
        hint = {
            "confidence": round(prediction.confidence, 3),
            "note": "cerebellar pre-prediction, may be inaccurate",
        }
        return cortex.evaluate_tool_call(tool_name, tool_params, context, hint)

    def _learn_from_llm(
        self,
        z: np.ndarray,
        tool_name: str,
        tool_params: dict,
        llm_result: dict,
    ):
        """Use the LLM's output as training signal for the prediction engine."""
        action_text = f"{tool_name}({json.dumps(tool_params, ensure_ascii=False)})"
        outcome_text = llm_result.get("expected_outcome", "")
        if not outcome_text:
            return

        actual_action_emb = self.encoder.encode_text(action_text)
        actual_outcome_emb = self.encoder.encode_text(outcome_text)

        # Pad/truncate to match engine dimensions
        actual_action_emb = self._fit_dim(actual_action_emb, self.cfg.action_dim)
        actual_outcome_emb = self._fit_dim(actual_outcome_emb, self.cfg.outcome_dim)

        loss = self.learner.learn(z, actual_action_emb, actual_outcome_emb)
        log.debug("online learning step loss=%.4f", loss)

    def _store_memory(
        self,
        tool_name: str,
        tool_params: dict,
        feature_vec: np.ndarray,
    ):
        content = f"{tool_name}({json.dumps(tool_params, ensure_ascii=False)})"
        # Use the text embedding portion (first embedding_dim elements)
        emb = feature_vec[:384] if len(feature_vec) >= 384 else feature_vec
        slot = MemorySlot(
            content=content,
            embedding=emb.copy(),
            layer="short_term",
            metadata={"tool_name": tool_name},
        )
        self.memory.store(slot)

    @staticmethod
    def _fit_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
        if len(vec) == target_dim:
            return vec
        if len(vec) > target_dim:
            return vec[:target_dim]
        return np.pad(vec, (0, target_dim - len(vec)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        routes = {"fast": 0, "shadow": 0, "slow": 0}
        for h in self._history:
            routes[h["route"]] = routes.get(h["route"], 0) + 1
        return {
            "step": self._step,
            "routes": routes,
            "router": self.router.stats,
            "memory": self.memory.stats,
        }
