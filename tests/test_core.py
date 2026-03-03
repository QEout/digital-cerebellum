"""
Core component tests for Digital Cerebellum Phase 0.

Run with:  pytest tests/test_core.py -v
"""

import numpy as np
import torch
import pytest

from src.core.pattern_separator import PatternSeparator
from src.core.prediction_engine import EngineConfig, PredictionEngine
from src.core.error_comparator import ErrorComparator, cosine_distance
from src.core.online_learner import OnlineLearner
from src.core.types import ErrorType, PredictionOutput, RouteDecision
from src.routing.decision_router import DecisionRouter
from src.memory.fluid_memory import FluidMemory
from src.core.types import MemorySlot


# ======================================================================
# Pattern Separator
# ======================================================================

class TestPatternSeparator:

    def test_output_shape(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256, sparsity=0.1)
        x = torch.randn(32)
        z = sep(x)
        assert z.shape == (256,)

    def test_batch_output_shape(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256)
        x = torch.randn(8, 32)
        z = sep(x)
        assert z.shape == (8, 256)

    def test_sparsity(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256, sparsity=0.1)
        x = torch.randn(32)
        z = sep(x)
        nonzero = (z != 0).sum().item()
        # top_k = 256 * 0.1 = 25 (approximately)
        assert nonzero <= 26

    def test_deterministic_for_same_input(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256)
        x = torch.randn(32)
        z1 = sep(x)
        z2 = sep(x)
        assert torch.allclose(z1, z2)

    def test_different_inputs_different_outputs(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256)
        x1 = torch.randn(32)
        x2 = torch.randn(32)
        z1 = sep(x1)
        z2 = sep(x2)
        assert not torch.allclose(z1, z2)

    def test_golgi_gate(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256, enable_golgi=True)
        x = torch.randn(32)
        z = sep(x)
        assert z.shape == (256,)

    def test_numpy_convenience(self):
        sep = PatternSeparator(input_dim=32, rff_dim=256)
        x = np.random.randn(32).astype(np.float32)
        z = sep.encode_event(x)
        assert isinstance(z, np.ndarray)
        assert z.shape == (256,)


# ======================================================================
# Prediction Engine
# ======================================================================

class TestPredictionEngine:

    def _make_engine(self, K=4) -> PredictionEngine:
        cfg = EngineConfig(rff_dim=256, action_dim=64, outcome_dim=64, num_heads=K)
        return PredictionEngine(cfg)

    def test_output_structure(self):
        engine = self._make_engine()
        z = torch.randn(256)
        pred = engine(z)
        assert isinstance(pred, PredictionOutput)
        assert pred.action_embedding.shape == (64,)
        assert pred.outcome_embedding.shape == (64,)
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.head_predictions) == 4

    def test_confidence_higher_for_similar_heads(self):
        """If heads agree, confidence should be high."""
        engine = self._make_engine(K=4)
        # Make all heads produce similar output by using same weights
        with torch.no_grad():
            for head in engine.heads[1:]:
                head.action_proj.weight.copy_(engine.heads[0].action_proj.weight)
                head.action_proj.bias.copy_(engine.heads[0].action_proj.bias)
                head.outcome_proj.weight.copy_(engine.heads[0].outcome_proj.weight)
                head.outcome_proj.bias.copy_(engine.heads[0].outcome_proj.bias)

        z = torch.randn(256)
        pred = engine(z)
        assert pred.confidence > 0.9

    def test_numpy_convenience(self):
        engine = self._make_engine()
        z = np.random.randn(256).astype(np.float32)
        pred = engine.predict_numpy(z)
        assert isinstance(pred, PredictionOutput)


# ======================================================================
# Error Comparator
# ======================================================================

class TestErrorComparator:

    def test_cosine_distance_identical(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_distance_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_sensory_error(self):
        comp = ErrorComparator()
        pred = PredictionOutput(
            action_embedding=np.array([1.0, 0.0, 0.0]),
            outcome_embedding=np.array([0.0, 1.0, 0.0]),
            confidence=0.8,
            head_predictions=[],
        )
        err = comp.compute_sensory_error(
            pred,
            actual_action_emb=np.array([1.0, 0.0, 0.0]),
            actual_outcome_emb=np.array([0.0, 1.0, 0.0]),
        )
        assert err.error_type == ErrorType.SENSORY
        assert err.value == pytest.approx(0.0, abs=1e-5)  # perfect prediction

    def test_reward_error(self):
        comp = ErrorComparator()
        err = comp.compute_reward_error(-1.0, "evt-1")
        assert err.error_type == ErrorType.REWARD
        assert err.value == -1.0


# ======================================================================
# Decision Router
# ======================================================================

class TestDecisionRouter:

    def test_fast_route(self):
        router = DecisionRouter(threshold_high=0.8)
        pred = PredictionOutput(
            action_embedding=np.zeros(64),
            outcome_embedding=np.zeros(64),
            confidence=0.9,
            head_predictions=[],
        )
        result = router.route(pred)
        assert result.decision == RouteDecision.FAST

    def test_slow_route(self):
        router = DecisionRouter(threshold_low=0.5)
        pred = PredictionOutput(
            action_embedding=np.zeros(64),
            outcome_embedding=np.zeros(64),
            confidence=0.3,
            head_predictions=[],
        )
        result = router.route(pred)
        assert result.decision == RouteDecision.SLOW

    def test_shadow_route(self):
        router = DecisionRouter(threshold_high=0.8, threshold_low=0.3)
        pred = PredictionOutput(
            action_embedding=np.zeros(64),
            outcome_embedding=np.zeros(64),
            confidence=0.6,
            head_predictions=[],
        )
        result = router.route(pred)
        assert result.decision == RouteDecision.SHADOW

    def test_rpe_adapts_threshold(self):
        router = DecisionRouter(threshold_high=0.90)
        # Positive RPE → threshold should decrease
        comp = ErrorComparator()
        for _ in range(10):
            err = comp.compute_reward_error(1.0)
            router.update_from_reward(err)
        assert router.threshold_high < 0.90


# ======================================================================
# Online Learner
# ======================================================================

class TestOnlineLearner:

    def test_loss_decreases(self):
        cfg = EngineConfig(rff_dim=128, action_dim=32, outcome_dim=32, num_heads=2)
        engine = PredictionEngine(cfg)
        learner = OnlineLearner(engine, lr=0.01, ewc_lambda=0)

        z = np.random.randn(128).astype(np.float32)
        target_a = np.random.randn(32).astype(np.float32)
        target_o = np.random.randn(32).astype(np.float32)

        losses = []
        for _ in range(20):
            loss = learner.learn(z, target_a, target_o)
            losses.append(loss)

        assert losses[-1] < losses[0]


# ======================================================================
# Fluid Memory
# ======================================================================

class TestFluidMemory:

    def test_store_and_retrieve(self):
        mem = FluidMemory()
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        slot = MemorySlot(content="test", embedding=emb)
        mem.store(slot)
        results = mem.retrieve(emb, top_k=3)
        assert len(results) == 1
        assert results[0].content == "test"

    def test_reconsolidation_boosts_strength(self):
        mem = FluidMemory()
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        slot = MemorySlot(content="test", embedding=emb, strength=0.3)
        mem.store(slot)
        results = mem.retrieve(emb)
        assert results[0].strength >= 0.8
        assert results[0].access_count == 1

    def test_capacity_enforcement(self):
        mem = FluidMemory()
        for i in range(60):
            emb = np.random.randn(384).astype(np.float32)
            slot = MemorySlot(content=f"mem-{i}", embedding=emb, strength=0.5)
            mem.store(slot)
        assert len(mem) <= FluidMemory.SHORT_TERM_CAPACITY

    def test_consolidation_promotes(self):
        mem = FluidMemory()
        emb = np.random.randn(384).astype(np.float32)
        slot = MemorySlot(
            content="repeated", embedding=emb,
            access_count=5, strength=0.9,
        )
        mem.store(slot)
        mem.consolidate()
        assert slot.layer == "long_term"
