"""
StepMonitor — universal agent step monitoring protocol.

This is the core of Phase 7: predictive error interception.

Any AI agent framework can use StepMonitor with just two calls::

    # Before executing an action
    prediction = monitor.before_step(
        action="click the save button",
        state="file editor is open with unsaved changes",
    )

    if not prediction.should_proceed:
        handle_warning(prediction.failure_warning)

    # Execute the action...
    result = agent.execute(step)

    # After executing
    verdict = monitor.after_step(
        outcome="save dialog appeared",
        success=True,
    )

    if verdict.should_pause:
        handle_cascade(verdict)

The StepMonitor is framework-agnostic.  It doesn't know whether
the agent is automating a desktop, playing a game, or controlling
a robot.  It only sees text descriptions (or pre-computed vectors)
of states, actions, and outcomes.

Biological mapping:
  - before_step  →  efference copy: cerebellum receives the motor
                     command before it's executed, predicts consequences
  - after_step   →  reafference comparison: cerebellum compares
                     predicted vs actual sensory feedback
  - cascade_detector  →  climbing fibre burst: sustained SPE triggers
                          motor program interruption
  - failure_memory    →  somatic markers: past failures bias future
                          decisions preemptively
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from digital_cerebellum.monitor.types import (
    RollbackPlan,
    StepPrediction,
    StepRecord,
    StepVerdict,
)
from digital_cerebellum.monitor.step_forward_model import StepForwardModel
from digital_cerebellum.monitor.cascade_detector import ErrorCascadeDetector
from digital_cerebellum.monitor.failure_memory import FailureMemory
from digital_cerebellum.memory.habit_observer import HabitObserver

log = logging.getLogger(__name__)


class StepMonitor:
    """
    Universal agent step monitor.

    Wraps any agent's execution loop with prediction and error
    detection.  Maintains an internal forward model that learns
    from every step, getting better over time.

    Can be used standalone or with an existing DigitalCerebellum
    instance (shares the encoder for consistency).
    """

    def __init__(
        self,
        cerebellum: Any | None = None,
        embedding_dim: int = 384,
        spe_threshold: float = 0.9,
        cascade_consecutive_limit: int = 3,
        cascade_risk_threshold: float = 0.7,
        auto_rollback: bool = True,
    ):
        if cerebellum is not None:
            self._encoder = cerebellum.encoder
            embedding_dim = self._encoder.output_dim - 4
        else:
            self._encoder = None

        self._embedding_dim = embedding_dim
        self._spe_threshold = spe_threshold
        self._auto_rollback = auto_rollback

        self._forward_model = StepForwardModel(
            embedding_dim=embedding_dim,
        )
        self._cascade_detector = ErrorCascadeDetector(
            spe_threshold=spe_threshold,
            consecutive_limit=cascade_consecutive_limit,
            cascade_risk_threshold=cascade_risk_threshold,
        )
        self._failure_memory = FailureMemory()
        self.habit_observer = HabitObserver()

        self._current_step: StepRecord | None = None
        self._step_count = 0
        self._history: list[StepRecord] = []
        self._total_pauses = 0
        self._last_rollback_plan: RollbackPlan | None = None

    # ==================================================================
    # Encoding — accepts text, dict, or pre-computed vectors
    # ==================================================================

    def _encode(self, input_data: str | dict | np.ndarray) -> np.ndarray:
        """
        Encode any input into a fixed-length embedding vector.

        Accepts:
        - str: encoded via sentence-transformer
        - dict: serialized to text, then encoded
        - np.ndarray: used directly (padded/truncated to embedding_dim)
        """
        if isinstance(input_data, np.ndarray):
            return self._fit_dim(input_data)

        if isinstance(input_data, dict):
            text = self._dict_to_text(input_data)
        else:
            text = str(input_data)

        encoder = self._get_encoder()
        vec = encoder.encode_text(text)
        return self._fit_dim(vec)

    def _get_encoder(self):
        if self._encoder is None:
            from digital_cerebellum.core.feature_encoder import FeatureEncoder
            self._encoder = FeatureEncoder()
        return self._encoder

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        """Pad or truncate to embedding_dim."""
        if len(vec) == self._embedding_dim:
            return vec
        if len(vec) > self._embedding_dim:
            return vec[:self._embedding_dim]
        return np.pad(vec, (0, self._embedding_dim - len(vec)))

    @staticmethod
    def _dict_to_text(d: dict) -> str:
        parts = []
        for k, v in d.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)

    # ==================================================================
    # Core protocol: before_step / after_step
    # ==================================================================

    def before_step(
        self,
        action: str | dict | np.ndarray,
        state: str | dict | np.ndarray | None = None,
        context: str = "",
    ) -> StepPrediction:
        """
        Call BEFORE executing an agent action.

        The cerebellum:
        1. Encodes the action and state
        2. Predicts the expected outcome via forward model
        3. Checks failure memory for known bad patterns
        4. Returns a prediction with risk assessment

        Parameters
        ----------
        action : What the agent intends to do.
            - str: "click the save button"
            - dict: {"type": "click", "target": "save_button"}
            - np.ndarray: pre-computed action embedding
        state : Current state of the world (optional).
            Same flexible types as action.
            If None, uses a zero vector.
        context : Additional context string (optional).

        Returns
        -------
        StepPrediction with should_proceed recommendation.
        """
        self._step_count += 1

        action_text = str(action) if not isinstance(action, np.ndarray) else ""
        state_text = str(state) if state is not None and not isinstance(state, np.ndarray) else ""

        if context and action_text:
            action_text = f"{context} → {action_text}"

        action_emb = self._encode(action)
        state_emb = (
            self._encode(state) if state is not None
            else np.zeros(self._embedding_dim, dtype=np.float32)
        )

        predicted_outcome, confidence = self._forward_model.predict(
            state_emb, action_emb,
        )

        failure_warning = self._failure_memory.check(state_emb, action_emb)

        cascade_status = self._cascade_detector._compute_risk()

        should_proceed = True
        risk_score = cascade_status

        if failure_warning is not None:
            risk_score = max(risk_score, failure_warning.severity)
            if failure_warning.severity > 0.8:
                should_proceed = False

        if cascade_status > 0.7:
            should_proceed = False

        self._current_step = StepRecord(
            step_number=self._step_count,
            action_embedding=action_emb,
            state_embedding=state_emb,
            predicted_outcome=predicted_outcome,
            timestamp=time.time(),
            action_text=action_text,
            state_text=state_text,
        )

        log.info(
            "before_step #%d: confidence=%.3f risk=%.3f proceed=%s",
            self._step_count, confidence, risk_score, should_proceed,
        )

        from digital_cerebellum.viz.event_bus import event_bus as _eb
        _eb.emit("step", "StepMonitor", phase="before",
                 step=self._step_count, confidence=float(confidence),
                 risk=float(risk_score), proceed=should_proceed)

        return StepPrediction(
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            risk_score=risk_score,
            should_proceed=should_proceed,
            failure_warning=failure_warning,
            cascade_risk=cascade_status,
            step_number=self._step_count,
        )

    def after_step(
        self,
        outcome: str | dict | np.ndarray,
        success: bool | None = None,
    ) -> StepVerdict:
        """
        Call AFTER executing an agent action.

        The cerebellum:
        1. Encodes the actual outcome
        2. Computes SPE (predicted vs actual)
        3. Updates the error cascade detector
        4. Learns from this step (updates forward model)
        5. Records failure if unsuccessful

        Parameters
        ----------
        outcome : What actually happened.
            Same flexible types as before_step's action/state.
        success : Whether the step succeeded (optional).
            If None, the cerebellum infers from SPE.

        Returns
        -------
        StepVerdict with should_pause recommendation.
        """
        if self._current_step is None:
            return StepVerdict(
                spe=0.0,
                suggestion="no_prediction_to_compare",
                step_number=self._step_count,
            )

        outcome_text = str(outcome) if not isinstance(outcome, np.ndarray) else ""
        outcome_emb = self._encode(outcome)

        spe = StepForwardModel.compute_spe(
            self._current_step.predicted_outcome, outcome_emb,
        )

        cascade_status = self._cascade_detector.observe(spe)

        error = self._forward_model.learn(
            self._current_step.state_embedding
            if self._current_step.state_embedding is not None
            else np.zeros(self._embedding_dim, dtype=np.float32),
            self._current_step.action_embedding,
            outcome_emb,
        )

        inferred_success = success if success is not None else (spe < self._spe_threshold)

        if not inferred_success:
            self._failure_memory.record(
                state_emb=self._current_step.state_embedding
                if self._current_step.state_embedding is not None
                else np.zeros(self._embedding_dim, dtype=np.float32),
                action_emb=self._current_step.action_embedding,
                action_text=self._current_step.action_text,
                state_text=self._current_step.state_text,
                error_description=f"SPE={spe:.3f}, outcome='{outcome_text[:100]}'",
                severity=min(spe / max(self._spe_threshold, 1e-9), 1.0),
                step_number=self._step_count,
            )

        self._current_step.actual_outcome = outcome_emb
        self._current_step.spe = spe
        self._current_step.success = inferred_success
        self._current_step.outcome_text = outcome_text
        self._history.append(self._current_step)

        should_pause = cascade_status.is_cascading
        if should_pause:
            self._total_pauses += 1

        rollback_plan = None
        suggestion = "continue"

        if should_pause:
            rollback_plan = self._compute_rollback_plan(cascade_status)
            self._last_rollback_plan = rollback_plan

            safe_step = rollback_plan.rollback_to_step
            suggestion = (
                f"pause: error cascade detected "
                f"({cascade_status.consecutive_high} consecutive errors, "
                f"trend={cascade_status.trend}). "
                f"Roll back to step {safe_step}."
            )
            if rollback_plan.last_safe_state:
                suggestion += f" Last safe state: {rollback_plan.last_safe_state[:80]}"
        elif cascade_status.risk > 0.4:
            suggestion = (
                f"caution: cascade risk rising "
                f"(risk={cascade_status.risk:.2f}, "
                f"trend={cascade_status.trend})"
            )

        log.info(
            "after_step #%d: spe=%.3f cascade_risk=%.3f pause=%s",
            self._step_count, spe, cascade_status.risk, should_pause,
        )

        self._current_step = None

        details: dict[str, Any] = {
            "forward_model_loss": round(error, 6),
            "success": inferred_success,
            "cascade_trend": cascade_status.trend,
        }
        if rollback_plan is not None:
            details["rollback_plan"] = {
                "rollback_to_step": rollback_plan.rollback_to_step,
                "last_safe_state": rollback_plan.last_safe_state,
                "last_safe_outcome": rollback_plan.last_safe_outcome,
                "steps_wasted": rollback_plan.steps_wasted,
                "failed_steps": rollback_plan.failed_steps,
                "recommendation": rollback_plan.recommendation,
            }

        # Record action for habit learning
        action_text_for_habit = (
            self._history[-1].action_text if self._history else ""
        )
        if action_text_for_habit:
            domain = ""
            if self._history:
                state_text = self._history[-1].state_text or ""
                for kw, d in [("email", "email"), ("calendar", "calendar"),
                              ("file", "file"), ("deploy", "devops"),
                              ("slack", "chat"), ("browser", "web"),
                              ("terminal", "shell"), ("code", "code")]:
                    if kw in action_text_for_habit.lower() or kw in state_text.lower():
                        domain = d
                        break
            self.habit_observer.record(
                action=action_text_for_habit,
                domain=domain,
                success=inferred_success,
            )

        from digital_cerebellum.viz.event_bus import event_bus as _eb
        _eb.emit("error", "ErrorComparator", spe=float(spe), step=self._step_count)
        if should_pause:
            _eb.emit("cascade", "ErrorCascadeDetector",
                     risk=float(cascade_status.risk), step=self._step_count)
        _eb.emit("step", "StepMonitor", phase="after",
                 step=self._step_count, spe=float(spe),
                 cascade_risk=float(cascade_status.risk), pause=should_pause)

        return StepVerdict(
            spe=spe,
            should_pause=should_pause,
            cascade_risk=cascade_status.risk,
            consecutive_errors=cascade_status.consecutive_high,
            suggestion=suggestion,
            step_number=self._step_count,
            learned=True,
            details=details,
        )

    # ==================================================================
    # AutoRollback
    # ==================================================================

    def _compute_rollback_plan(self, cascade_status: Any) -> RollbackPlan:
        """Build a rollback plan from step history when cascade is detected."""
        last_safe_idx = -1
        for i, record in enumerate(self._history):
            if record.success:
                last_safe_idx = i

        if last_safe_idx >= 0:
            safe_record = self._history[last_safe_idx]
            rollback_to = safe_record.step_number
            last_safe_state = safe_record.state_text or safe_record.outcome_text
            last_safe_outcome = safe_record.outcome_text
        else:
            rollback_to = 0
            last_safe_state = "(initial state — before any steps)"
            last_safe_outcome = ""

        failed_steps = []
        for record in self._history:
            if not record.success:
                failed_steps.append({
                    "step": record.step_number,
                    "action": record.action_text,
                    "outcome": record.outcome_text[:120],
                    "spe": round(record.spe, 3),
                })

        steps_wasted = len(failed_steps)
        total = len(self._history)

        if steps_wasted == 0:
            recommendation = "No failed steps found; cascade may be due to cumulative drift."
        elif steps_wasted == 1:
            recommendation = f"Undo step {failed_steps[0]['step']} and retry with different approach."
        else:
            recommendation = (
                f"Roll back to step {rollback_to}, discard {steps_wasted} failed steps, "
                f"and retry from state: {last_safe_state[:60]}"
            )

        return RollbackPlan(
            rollback_to_step=rollback_to,
            last_safe_state=last_safe_state,
            last_safe_outcome=last_safe_outcome,
            failed_steps=failed_steps,
            total_steps=total,
            steps_wasted=steps_wasted,
            cascade_risk=cascade_status.risk if hasattr(cascade_status, 'risk') else 0.0,
            recommendation=recommendation,
        )

    def get_rollback_plan(self) -> RollbackPlan | None:
        """
        Get the most recent rollback plan (if any).

        A plan is computed automatically when a cascade is detected.
        Returns None if no cascade has been detected in this episode.
        """
        return self._last_rollback_plan

    # ==================================================================
    # Task management
    # ==================================================================

    def reset(self) -> dict[str, Any]:
        """
        Reset for a new task/episode.

        Clears step history and cascade detector, but KEEPS the
        forward model and failure memory (learned knowledge persists).

        Returns summary of the completed episode.
        """
        summary = self.episode_summary
        self._current_step = None
        self._step_count = 0
        self._cascade_detector.reset()
        self._history.clear()
        self._last_rollback_plan = None
        return summary

    @property
    def episode_summary(self) -> dict[str, Any]:
        """Summary of the current/last episode."""
        if not self._history:
            return {"steps": 0}

        spes = [r.spe for r in self._history]
        successes = [r.success for r in self._history if r.success is not None]

        return {
            "steps": len(self._history),
            "mean_spe": round(float(np.mean(spes)), 4) if spes else 0.0,
            "max_spe": round(float(np.max(spes)), 4) if spes else 0.0,
            "success_rate": (
                round(sum(successes) / len(successes), 3)
                if successes else None
            ),
            "pauses": self._total_pauses,
        }

    # ==================================================================
    # Diagnostics
    # ==================================================================

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "total_pauses": self._total_pauses,
            "forward_model": self._forward_model.stats,
            "cascade_detector": self._cascade_detector.stats,
            "failure_memory": self._failure_memory.stats,
            "episode": self.episode_summary,
        }

    # ==================================================================
    # Persistence
    # ==================================================================

    def save(self, path: str | Path = ".digital-cerebellum/monitor") -> None:
        """
        Save learned knowledge (forward model + failure memory).

        Step history and cascade state are NOT saved (they're ephemeral).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "forward_model": self._forward_model.state_dict(),
            "forward_model_step": self._forward_model._step,
            "forward_model_loss": self._forward_model._cumulative_loss,
            "forward_model_errors": self._forward_model._recent_errors,
            "failure_records": [
                {
                    "fingerprint": r.fingerprint,
                    "action_text": r.action_text,
                    "state_text": r.state_text,
                    "error_description": r.error_description,
                    "severity": r.severity,
                    "step_number": r.step_number,
                    "strength": r.strength,
                }
                for r in self._failure_memory._records
            ],
            "total_pauses": self._total_pauses,
            "embedding_dim": self._embedding_dim,
            "spe_threshold": self._spe_threshold,
        }, path / "monitor.pt")

        log.info("StepMonitor saved to %s", path)

    def load(self, path: str | Path = ".digital-cerebellum/monitor") -> None:
        """Load previously saved knowledge."""
        ckpt_path = Path(path) / "monitor.pt"
        if not ckpt_path.exists():
            log.info("No monitor checkpoint at %s", ckpt_path)
            return

        ckpt = torch.load(ckpt_path, weights_only=False)

        self._forward_model.load_state_dict(ckpt["forward_model"])
        self._forward_model._step = ckpt.get("forward_model_step", 0)
        self._forward_model._cumulative_loss = ckpt.get("forward_model_loss", 0.0)
        self._forward_model._recent_errors = ckpt.get("forward_model_errors", [])
        self._total_pauses = ckpt.get("total_pauses", 0)

        from digital_cerebellum.monitor.failure_memory import _FailureRecord
        self._failure_memory._records.clear()
        for rec in ckpt.get("failure_records", []):
            self._failure_memory._records.append(_FailureRecord(
                fingerprint=rec["fingerprint"],
                action_text=rec["action_text"],
                state_text=rec["state_text"],
                error_description=rec["error_description"],
                severity=rec["severity"],
                step_number=rec["step_number"],
                strength=rec["strength"],
            ))

        log.info(
            "StepMonitor loaded from %s (fm_step=%d, failures=%d)",
            path, self._forward_model._step, len(self._failure_memory._records),
        )

    @property
    def cascade_detector(self) -> ErrorCascadeDetector:
        return self._cascade_detector

    @property
    def failure_memory(self) -> FailureMemory:
        return self._failure_memory

    @property
    def forward_model(self) -> StepForwardModel:
        return self._forward_model
