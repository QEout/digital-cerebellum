"""
Skill Store — procedural memory for learned skills.

Biological basis:
  In the cerebellum, repeated cortico-cerebellar interactions consolidate
  motor programs into procedural memory.  A pianist doesn't think about
  individual finger movements — the cerebellum replays the entire
  learned sequence as a single "skill unit".

  Memory traces transfer from the cerebellar cortex (complex, flexible)
  to the cerebellar nuclei (simple, fast) via systems consolidation.
  Reference: Nature Communications 2025, "A normative principle
  governing memory transfer in cerebellar motor learning"

Digital implementation:
  Skills are (input_pattern → output_action) pairs stored with their
  embedding vectors.  When a new query arrives, the SkillStore finds
  the most similar stored skill using cosine similarity.  If the match
  exceeds a threshold AND the skill has proven reliable, the system
  executes the skill directly without calling the LLM.

  Over time, successful skills gain confidence (reinforcement) and
  failed skills lose confidence (extinction), mirroring biological
  synaptic plasticity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Skill:
    """A learned procedural skill — a (pattern → action) mapping."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    input_embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    input_text: str = ""
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    domain: str = ""

    confidence: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    access_count: int = 0

    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def is_sequence(self) -> bool:
        return len(self.tool_calls) > 1


@dataclass
class SkillMatch:
    """Result of a skill lookup."""

    skill: Skill
    similarity: float
    match_confidence: float

    @property
    def should_execute(self) -> bool:
        return self.match_confidence >= 0.7


class SkillStore:
    """
    Procedural memory for learned skills.

    Uses brute-force cosine similarity for matching (fast enough for
    thousands of skills).  Can be upgraded to FAISS for larger stores.
    """

    SIMILARITY_THRESHOLD = 0.85
    MIN_CONFIDENCE = 0.4
    REINFORCEMENT_RATE = 0.1
    EXTINCTION_RATE = 0.15
    MAX_SKILLS = 10000

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self._skills: dict[str, Skill] = {}
        self._embeddings: np.ndarray | None = None
        self._id_index: list[str] = []
        self._dirty = True
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence

    def store(self, skill: Skill) -> str:
        """Store a new skill. Returns the skill id."""
        self._skills[skill.id] = skill
        self._dirty = True
        self._enforce_capacity()
        return skill.id

    def learn_from_interaction(
        self,
        input_embedding: np.ndarray,
        input_text: str,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None = None,
        domain: str = "",
    ) -> str:
        """
        Learn a new skill from an LLM interaction.

        If a very similar skill already exists, reinforce it instead
        of creating a duplicate.
        """
        existing = self._find_nearest(input_embedding)
        if existing is not None and existing.similarity > 0.95:
            skill = existing.skill
            skill.access_count += 1
            skill.confidence = min(1.0, skill.confidence + self.REINFORCEMENT_RATE)
            skill.last_used = time.time()
            if response_text and response_text != skill.response_text:
                skill.response_text = response_text
            if tool_calls is not None:
                skill.tool_calls = tool_calls
            self._dirty = True
            return skill.id

        skill = Skill(
            input_embedding=input_embedding.copy(),
            input_text=input_text,
            response_text=response_text,
            tool_calls=tool_calls or [],
            domain=domain,
            confidence=0.5,
        )
        return self.store(skill)

    def match(self, query_embedding: np.ndarray) -> SkillMatch | None:
        """
        Find the best matching skill for a query.

        Returns None if no skill exceeds the similarity threshold
        or if the best match has too low confidence.
        """
        result = self._find_nearest(query_embedding)
        if result is None:
            return None

        if result.similarity < self.similarity_threshold:
            return None
        if result.skill.confidence < self.min_confidence:
            return None

        return result

    def reinforce(self, skill_id: str) -> None:
        """Mark a skill execution as successful (synaptic potentiation)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.success_count += 1
        skill.access_count += 1
        skill.confidence = min(1.0, skill.confidence + self.REINFORCEMENT_RATE)
        skill.last_used = time.time()

    def weaken(self, skill_id: str) -> None:
        """Mark a skill execution as failed (synaptic depression)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.failure_count += 1
        skill.access_count += 1
        skill.confidence = max(0.0, skill.confidence - self.EXTINCTION_RATE)
        skill.last_used = time.time()

    def consolidate(self) -> dict[str, int]:
        """
        Sleep-cycle consolidation: merge similar skills, prune weak ones.

        Mirrors cerebellar cortex → nuclei memory transfer:
        simple, reliable skills are retained; noisy, unreliable ones
        are pruned.
        """
        pruned = 0
        merged = 0

        prune_ids = [
            sid for sid, s in self._skills.items()
            if s.confidence < 0.2 and s.access_count >= 3
        ]
        for sid in prune_ids:
            del self._skills[sid]
            pruned += 1

        skills_by_domain: dict[str, list[Skill]] = {}
        for s in self._skills.values():
            skills_by_domain.setdefault(s.domain, []).append(s)

        for domain, skills in skills_by_domain.items():
            if len(skills) < 2:
                continue
            i = 0
            while i < len(skills):
                j = i + 1
                while j < len(skills):
                    sim = self._cosine_sim(
                        skills[i].input_embedding, skills[j].input_embedding
                    )
                    if sim > 0.95:
                        winner, loser = (
                            (skills[i], skills[j])
                            if skills[i].confidence >= skills[j].confidence
                            else (skills[j], skills[i])
                        )
                        winner.success_count += loser.success_count
                        winner.failure_count += loser.failure_count
                        winner.access_count += loser.access_count
                        if loser.id in self._skills:
                            del self._skills[loser.id]
                            merged += 1
                        skills.pop(j)
                    else:
                        j += 1
                i += 1

        self._dirty = True
        return {"pruned": pruned, "merged": merged, "remaining": len(self._skills)}

    # ------------------------------------------------------------------
    # Retrieval internals
    # ------------------------------------------------------------------

    def _find_nearest(self, query: np.ndarray) -> SkillMatch | None:
        if not self._skills:
            return None

        self._rebuild_index()

        q_norm = np.linalg.norm(query)
        if q_norm < 1e-9:
            return None

        sims = self._embeddings @ query / (
            np.linalg.norm(self._embeddings, axis=1) * q_norm + 1e-9
        )
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        skill = self._skills[self._id_index[best_idx]]
        match_conf = best_sim * skill.confidence

        return SkillMatch(
            skill=skill,
            similarity=best_sim,
            match_confidence=match_conf,
        )

    def _rebuild_index(self):
        if not self._dirty:
            return
        self._id_index = list(self._skills.keys())
        if self._id_index:
            self._embeddings = np.stack(
                [self._skills[sid].input_embedding for sid in self._id_index]
            )
        else:
            self._embeddings = None
        self._dirty = False

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(dot / norm)

    def _enforce_capacity(self):
        if len(self._skills) <= self.MAX_SKILLS:
            return
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.confidence * (s.success_count + 1),
        )
        for s in sorted_skills[: len(sorted_skills) - self.MAX_SKILLS]:
            del self._skills[s.id]
        self._dirty = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._skills)

    @property
    def stats(self) -> dict[str, Any]:
        if not self._skills:
            return {"total": 0, "by_domain": {}, "avg_confidence": 0.0}

        by_domain: dict[str, int] = {}
        confidences = []
        for s in self._skills.values():
            by_domain[s.domain] = by_domain.get(s.domain, 0) + 1
            confidences.append(s.confidence)

        return {
            "total": len(self._skills),
            "by_domain": by_domain,
            "avg_confidence": float(np.mean(confidences)),
            "avg_reliability": float(np.mean([
                s.reliability for s in self._skills.values()
            ])),
        }

    def get_skills(self, domain: str | None = None) -> list[Skill]:
        """List all skills, optionally filtered by domain."""
        skills = list(self._skills.values())
        if domain:
            skills = [s for s in skills if s.domain == domain]
        skills.sort(key=lambda s: s.confidence, reverse=True)
        return skills
