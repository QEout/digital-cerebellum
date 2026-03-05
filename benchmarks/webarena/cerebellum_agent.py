"""
Cerebellum-enhanced browsing agent for WebArena benchmark.

Wraps the base WebAgent with Digital Cerebellum's StepMonitor
and SkillStore as a transparent sidecar. Three integration points:

1. Before each step: SkillStore fast-path (skip LLM if we've seen this before)
2. Before each step: StepMonitor risk check (block cascading failures)
3. After each step: StepMonitor learning + FailureMemory recording
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import Page, sync_playwright

from digital_cerebellum import DigitalCerebellum, StepMonitor
from benchmarks.webarena.agent import (
    AgentResult,
    StepRecord,
    WebAgent,
    extract_page_observation,
    execute_action,
    parse_llm_action,
    ACTION_TIMEOUT_MS,
    PAGE_LOAD_WAIT_MS,
)

log = logging.getLogger(__name__)


# ======================================================================
# Ablation config
# ======================================================================

@dataclass
class AblationConfig:
    """Toggle cerebellum components on/off for ablation study."""
    use_skill_store: bool = True
    use_step_monitor: bool = True
    use_failure_memory: bool = True
    label: str = "full"

    @classmethod
    def full(cls) -> AblationConfig:
        return cls(label="full")

    @classmethod
    def no_skill(cls) -> AblationConfig:
        return cls(use_skill_store=False, label="no_skill")

    @classmethod
    def no_monitor(cls) -> AblationConfig:
        return cls(use_step_monitor=False, label="no_monitor")

    @classmethod
    def no_memory(cls) -> AblationConfig:
        return cls(use_failure_memory=False, label="no_memory")


# ======================================================================
# Cerebellum stats for a single task run
# ======================================================================

@dataclass
class CerebellumStats:
    skill_hits: int = 0
    skill_misses: int = 0
    steps_blocked: int = 0
    cascades_caught: int = 0
    failures_recorded: int = 0
    llm_calls_saved: int = 0


# ======================================================================
# CerebellumWebAgent
# ======================================================================

class CerebellumWebAgent(WebAgent):
    """WebAgent enhanced with Digital Cerebellum sidecar."""

    def __init__(
        self,
        ablation: AblationConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.ablation = ablation or AblationConfig.full()

        self._cerebellum = DigitalCerebellum()
        self._monitor = StepMonitor(cerebellum=self._cerebellum)

        if not self.ablation.use_failure_memory:
            self._monitor._failure_memory = None  # type: ignore[assignment]

    @property
    def cerebellum_stats(self) -> CerebellumStats:
        return self._task_stats

    def run_task(
        self,
        intent: str,
        start_url: str,
        task_id: int = -1,
        page: Page | None = None,
    ) -> AgentResult:
        result = AgentResult(task_id=task_id, intent=intent, response="")
        self._task_stats = CerebellumStats()
        t0 = time.perf_counter()

        own_browser = page is None
        pw_ctx = None
        browser = None

        try:
            if own_browser:
                pw_ctx = sync_playwright().start()
                browser = pw_ctx.chromium.launch(headless=self.headless)
                ctx = browser.new_context()
                page = ctx.new_page()

            page.goto(start_url, timeout=ACTION_TIMEOUT_MS)
            page.wait_for_load_state("domcontentloaded", timeout=PAGE_LOAD_WAIT_MS)

            self._monitor.reset()

            for step_num in range(self.max_steps):
                obs = extract_page_observation(page)
                step_t0 = time.perf_counter()

                # --- Sidecar: SkillStore fast path ---
                if self.ablation.use_skill_store:
                    skill_match = self._cerebellum.match_skill(intent + " | " + obs[:500])
                    if skill_match and skill_match.should_execute:
                        action = parse_llm_action(skill_match.skill.response_text)
                        self._task_stats.skill_hits += 1
                        self._task_stats.llm_calls_saved += 1
                        self._cerebellum.skill_store.reinforce(skill_match.skill.id)
                        elapsed = (time.perf_counter() - step_t0) * 1000

                        act_type = action.get("action", "")
                        if act_type == "answer":
                            result.response = action.get("args", {}).get("text", "")
                            result.steps.append(StepRecord(
                                step=step_num, action=f"[SKILL] answer: {result.response}",
                                observation=obs[:200], llm_response="[cached]",
                                elapsed_ms=elapsed,
                            ))
                            break

                        success, desc = execute_action(page, action)
                        result.steps.append(StepRecord(
                            step=step_num, action=f"[SKILL] {desc}",
                            observation=obs[:200], llm_response="[cached]",
                            elapsed_ms=elapsed, success=success,
                        ))

                        if self.ablation.use_step_monitor:
                            self._monitor.after_step(
                                outcome=extract_page_observation(page)[:500],
                                success=success,
                            )
                        continue
                    else:
                        self._task_stats.skill_misses += 1

                # --- Sidecar: StepMonitor pre-check ---
                if self.ablation.use_step_monitor:
                    pred = self._monitor.before_step(
                        action=f"browsing for: {intent[:100]}",
                        state=obs[:500],
                    )
                    if not pred.should_proceed:
                        self._task_stats.steps_blocked += 1
                        result.steps.append(StepRecord(
                            step=step_num, action="[BLOCKED] cascade risk too high",
                            observation=obs[:200], llm_response="",
                            elapsed_ms=(time.perf_counter() - step_t0) * 1000,
                            success=False,
                        ))
                        result.error = "cascade_blocked"
                        result.response = ""
                        self._task_stats.cascades_caught += 1
                        break

                # --- Core: LLM decision ---
                raw_llm = self._call_llm(intent, obs, result.steps)
                result.total_llm_calls += 1
                elapsed = (time.perf_counter() - step_t0) * 1000

                action = parse_llm_action(raw_llm)
                act_type = action.get("action", "")

                if act_type == "answer":
                    result.response = action.get("args", {}).get("text", "")
                    result.steps.append(StepRecord(
                        step=step_num, action=f"answer: {result.response}",
                        observation=obs[:200], llm_response=raw_llm,
                        elapsed_ms=elapsed,
                    ))
                    if self.ablation.use_skill_store:
                        self._cerebellum.learn_skill(
                            intent + " | " + obs[:500],
                            raw_llm,
                            domain="webarena",
                        )
                    break

                if act_type == "fail":
                    result.response = ""
                    result.error = action.get("args", {}).get("reason", "unknown")
                    result.steps.append(StepRecord(
                        step=step_num, action=f"fail: {result.error}",
                        observation=obs[:200], llm_response=raw_llm,
                        elapsed_ms=elapsed, success=False,
                    ))
                    self._task_stats.failures_recorded += 1
                    break

                success, desc = execute_action(page, action)
                result.steps.append(StepRecord(
                    step=step_num, action=desc,
                    observation=obs[:200], llm_response=raw_llm,
                    elapsed_ms=elapsed, success=success,
                ))

                # --- Sidecar: StepMonitor post-check ---
                if self.ablation.use_step_monitor:
                    new_obs = extract_page_observation(page)[:500]
                    verdict = self._monitor.after_step(
                        outcome=new_obs,
                        success=success,
                    )
                    if verdict.should_pause:
                        self._task_stats.cascades_caught += 1
                        result.steps.append(StepRecord(
                            step=step_num + 1,
                            action="[CASCADE] stopping early",
                            observation="", llm_response="",
                            elapsed_ms=0, success=False,
                        ))
                        result.error = "cascade_detected"
                        break

                # Learn successful interaction as skill
                if success and self.ablation.use_skill_store:
                    skill_id = self._cerebellum.learn_skill(
                        intent + " | " + obs[:500],
                        raw_llm,
                        domain="webarena",
                    )
                    self._cerebellum.skill_store.reinforce(skill_id)

                if not success:
                    self._task_stats.failures_recorded += 1

            else:
                result.response = ""
                result.error = "max_steps_exceeded"

        except Exception as e:
            result.error = str(e)
            log.error("Task %d error: %s", task_id, e)

        finally:
            if own_browser:
                if browser:
                    browser.close()
                if pw_ctx:
                    pw_ctx.stop()

        result.total_time_s = time.perf_counter() - t0
        return result
