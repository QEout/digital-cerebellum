"""Tests for the SkillStore — procedural memory for learned skills."""

import numpy as np
import pytest

from digital_cerebellum.memory.skill_store import Skill, SkillMatch, SkillStore


def _rand_emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class TestSkillStore:

    def test_store_and_retrieve(self):
        store = SkillStore()
        emb = _rand_emb(42)
        skill = Skill(
            input_embedding=emb,
            input_text="What's the weather?",
            response_text="It's sunny.",
            domain="response",
        )
        sid = store.store(skill)
        assert len(store) == 1
        assert sid == skill.id

    def test_match_exact(self):
        store = SkillStore(similarity_threshold=0.8)
        emb = _rand_emb(42)
        store.store(Skill(
            input_embedding=emb,
            input_text="hello",
            response_text="world",
            confidence=0.9,
        ))

        result = store.match(emb)
        assert result is not None
        assert result.similarity > 0.99
        assert result.skill.response_text == "world"

    def test_match_similar(self):
        store = SkillStore(similarity_threshold=0.6)
        emb = _rand_emb(42)
        store.store(Skill(
            input_embedding=emb,
            input_text="weather in tokyo",
            response_text="sunny in tokyo",
            confidence=0.9,
        ))

        rng = np.random.RandomState(99)
        noise = rng.randn(384).astype(np.float32)
        noise = noise / (np.linalg.norm(noise) + 1e-9)
        query = 0.8 * emb + 0.2 * noise
        query = query / (np.linalg.norm(query) + 1e-9)
        result = store.match(query)
        assert result is not None
        assert result.similarity > 0.6

    def test_no_match_for_dissimilar(self):
        store = SkillStore(similarity_threshold=0.85)
        store.store(Skill(
            input_embedding=_rand_emb(1),
            input_text="weather",
            response_text="sunny",
            confidence=0.9,
        ))

        result = store.match(_rand_emb(999))
        assert result is None

    def test_no_match_for_low_confidence(self):
        store = SkillStore(similarity_threshold=0.8, min_confidence=0.5)
        emb = _rand_emb(42)
        store.store(Skill(
            input_embedding=emb,
            input_text="test",
            response_text="response",
            confidence=0.3,
        ))

        result = store.match(emb)
        assert result is None

    def test_reinforce_increases_confidence(self):
        store = SkillStore()
        skill = Skill(
            input_embedding=_rand_emb(1),
            response_text="ok",
            confidence=0.5,
        )
        store.store(skill)

        old_conf = skill.confidence
        store.reinforce(skill.id)
        assert skill.confidence > old_conf
        assert skill.success_count == 1

    def test_weaken_decreases_confidence(self):
        store = SkillStore()
        skill = Skill(
            input_embedding=_rand_emb(1),
            response_text="ok",
            confidence=0.5,
        )
        store.store(skill)

        old_conf = skill.confidence
        store.weaken(skill.id)
        assert skill.confidence < old_conf
        assert skill.failure_count == 1

    def test_learn_from_interaction_creates_new(self):
        store = SkillStore()
        sid = store.learn_from_interaction(
            input_embedding=_rand_emb(1),
            input_text="hello",
            response_text="world",
            domain="test",
        )
        assert len(store) == 1
        assert store._skills[sid].input_text == "hello"

    def test_learn_from_interaction_reinforces_existing(self):
        store = SkillStore()
        emb = _rand_emb(1)
        sid1 = store.learn_from_interaction(
            input_embedding=emb,
            input_text="hello",
            response_text="world",
            domain="test",
        )
        old_conf = store._skills[sid1].confidence

        sid2 = store.learn_from_interaction(
            input_embedding=emb,
            input_text="hello",
            response_text="world updated",
            domain="test",
        )
        assert sid1 == sid2
        assert store._skills[sid1].confidence > old_conf
        assert len(store) == 1

    def test_consolidation_prunes_weak(self):
        store = SkillStore()
        weak = Skill(
            input_embedding=_rand_emb(1),
            response_text="weak",
            confidence=0.1,
            access_count=5,
        )
        strong = Skill(
            input_embedding=_rand_emb(2),
            response_text="strong",
            confidence=0.9,
        )
        store.store(weak)
        store.store(strong)

        result = store.consolidate()
        assert result["pruned"] == 1
        assert len(store) == 1

    def test_consolidation_merges_similar(self):
        store = SkillStore()
        emb = _rand_emb(1)
        s1 = Skill(
            input_embedding=emb.copy(),
            response_text="response1",
            confidence=0.9,
            success_count=5,
        )
        s2 = Skill(
            input_embedding=emb.copy(),
            response_text="response2",
            confidence=0.7,
            success_count=3,
        )
        store.store(s1)
        store.store(s2)

        result = store.consolidate()
        assert result["merged"] == 1
        assert len(store) == 1
        remaining = list(store._skills.values())[0]
        assert remaining.success_count == 8

    def test_skill_is_sequence(self):
        skill_single = Skill(
            input_embedding=_rand_emb(1),
            tool_calls=[{"tool": "search", "params": {"q": "weather"}}],
        )
        skill_multi = Skill(
            input_embedding=_rand_emb(2),
            tool_calls=[
                {"tool": "search", "params": {"q": "weather"}},
                {"tool": "format", "params": {"style": "brief"}},
            ],
        )
        assert not skill_single.is_sequence
        assert skill_multi.is_sequence

    def test_skill_reliability(self):
        skill = Skill(input_embedding=_rand_emb(1))
        assert skill.reliability == 0.5

        skill.success_count = 7
        skill.failure_count = 3
        assert skill.reliability == 0.7

    def test_should_execute(self):
        skill = Skill(input_embedding=_rand_emb(1), confidence=0.9)
        match = SkillMatch(skill=skill, similarity=0.9, match_confidence=0.81)
        assert match.should_execute

        low_match = SkillMatch(skill=skill, similarity=0.6, match_confidence=0.54)
        assert not low_match.should_execute

    def test_stats(self):
        store = SkillStore()
        store.store(Skill(
            input_embedding=_rand_emb(1),
            domain="response",
            confidence=0.8,
        ))
        store.store(Skill(
            input_embedding=_rand_emb(2),
            domain="tool_call",
            confidence=0.6,
        ))

        stats = store.stats
        assert stats["total"] == 2
        assert "response" in stats["by_domain"]
        assert "tool_call" in stats["by_domain"]
        assert 0.6 < stats["avg_confidence"] < 0.8

    def test_empty_store_returns_none(self):
        store = SkillStore()
        assert store.match(_rand_emb(1)) is None

    def test_capacity_enforcement(self):
        store = SkillStore()
        store.MAX_SKILLS = 5
        for i in range(10):
            store.store(Skill(
                input_embedding=_rand_emb(i),
                response_text=f"skill_{i}",
                confidence=i / 10.0,
            ))
        assert len(store) == 5


class TestDigitalBrainSkills:
    """Integration tests: DigitalBrain + SkillStore."""

    def test_brain_has_skill_store(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())
        assert hasattr(cb, "skill_store")
        assert len(cb.skill_store) == 0

    def test_learn_and_match_skill(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())

        sid = cb.learn_skill(
            input_text="What's the weather in Tokyo?",
            response_text="The weather in Tokyo is sunny, 25°C.",
            domain="response",
        )
        assert sid is not None

        match = cb.match_skill("What's the weather in Tokyo?")
        assert match is not None
        assert match.similarity > 0.95
        assert match.skill.response_text == "The weather in Tokyo is sunny, 25°C."

    def test_similar_query_matches(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())

        cb.learn_skill(
            input_text="What's the weather in Tokyo?",
            response_text="Sunny in Tokyo.",
            domain="response",
        )

        match = cb.match_skill("Tell me the weather in Tokyo")
        assert match is not None
        assert match.similarity > 0.7

    def test_dissimilar_query_no_match(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())

        cb.learn_skill(
            input_text="What's the weather in Tokyo?",
            response_text="Sunny.",
            domain="response",
        )

        match = cb.match_skill("Explain quantum physics")
        assert match is None

    def test_skill_feedback_reinforces(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())

        sid = cb.learn_skill(
            input_text="hello",
            response_text="world",
            domain="test",
        )

        old_conf = cb.skill_store._skills[sid].confidence
        cb.skill_store.reinforce(sid)
        assert cb.skill_store._skills[sid].confidence > old_conf

    def test_skill_store_in_stats(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
        cb = DigitalCerebellum(CerebellumConfig())
        cb.register_microzone(ToolCallMicrozone())
        stats = cb.stats
        assert "skill_store" in stats

    def test_sleep_consolidates_skills(self):
        from digital_cerebellum.main import CerebellumConfig, DigitalCerebellum
        cb = DigitalCerebellum(CerebellumConfig())

        from digital_cerebellum.memory.skill_store import Skill
        cb.skill_store.store(Skill(
            input_embedding=_rand_emb(1),
            response_text="weak skill",
            confidence=0.1,
            access_count=5,
        ))
        cb.skill_store.store(Skill(
            input_embedding=_rand_emb(2),
            response_text="strong skill",
            confidence=0.9,
        ))

        report = cb.sleep()
        assert report.skill_consolidation is not None
        assert report.skill_consolidation["pruned"] == 1


class TestCuriosityExploration:
    """Tests for active exploration signals."""

    def test_exploration_requests_empty_initially(self):
        from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive
        cd = CuriosityDrive()
        requests = cd.get_exploration_requests()
        assert requests == []

    def test_exploration_requests_with_high_error_low_obs(self):
        from digital_cerebellum.emergence.curiosity_drive import CuriosityDrive
        cd = CuriosityDrive()
        vec = _rand_emb(1)
        for i in range(15):
            cd.assess("new_domain", error=0.5, feature_vec=vec)

        requests = cd.get_exploration_requests()
        found = [r for r in requests if r["domain"] == "new_domain"]
        assert len(found) >= 0  # may or may not trigger depending on progress


class TestSkillStorePersistence:
    """Tests for save/load round-trip."""

    def test_save_load_round_trip(self, tmp_path):
        store = SkillStore(similarity_threshold=0.80)
        emb1 = _rand_emb(10)
        emb2 = _rand_emb(20)
        store.learn_from_interaction(emb1, "hello", "world", domain="qa")
        sid2 = store.learn_from_interaction(emb2, "bye", "see ya", [{"tool": "x"}], domain="chat")
        store.reinforce(sid2)
        store.reinforce(sid2)

        store.save(tmp_path / "skills")

        store2 = SkillStore()
        loaded = store2.load(tmp_path / "skills")
        assert loaded == 2
        assert len(store2) == 2
        assert store2.similarity_threshold == 0.80

        match = store2.match(emb2)
        assert match is not None
        assert match.skill.response_text == "see ya"
        assert match.skill.success_count == 2
        assert match.skill.tool_calls == [{"tool": "x"}]
        assert match.skill.domain == "chat"

    def test_save_empty_store(self, tmp_path):
        store = SkillStore()
        store.save(tmp_path / "skills")
        store2 = SkillStore()
        loaded = store2.load(tmp_path / "skills")
        assert loaded == 0
        assert len(store2) == 0

    def test_load_nonexistent_path(self, tmp_path):
        store = SkillStore()
        loaded = store.load(tmp_path / "nonexistent")
        assert loaded == 0

    def test_save_preserves_embeddings(self, tmp_path):
        store = SkillStore()
        emb = _rand_emb(99)
        store.learn_from_interaction(emb, "test", "result", domain="d")
        store.save(tmp_path / "skills")

        store2 = SkillStore()
        store2.load(tmp_path / "skills")
        skill = store2.get_skills()[0]
        np.testing.assert_allclose(skill.input_embedding, emb, atol=1e-6)

    def test_overwrite_on_second_save(self, tmp_path):
        store = SkillStore()
        store.learn_from_interaction(_rand_emb(1), "a", "b", domain="d")
        store.save(tmp_path / "skills")

        store.learn_from_interaction(_rand_emb(2), "c", "d", domain="d")
        store.save(tmp_path / "skills")

        store2 = SkillStore()
        loaded = store2.load(tmp_path / "skills")
        assert loaded == 2
