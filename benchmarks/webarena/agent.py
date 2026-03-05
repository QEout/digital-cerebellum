"""
Base browsing agent for WebArena benchmark.

Navigates web pages via Playwright and uses an LLM (OpenAI-compatible API)
to decide actions. Returns a text response when the task is complete.

This is the *baseline* agent — no cerebellum.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from playwright.sync_api import Page, sync_playwright

log = logging.getLogger(__name__)

MAX_STEPS = 15
ACTION_TIMEOUT_MS = 10_000
PAGE_LOAD_WAIT_MS = 2_000


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class StepRecord:
    step: int
    action: str
    observation: str
    llm_response: str
    elapsed_ms: float
    success: bool = True


@dataclass
class AgentResult:
    task_id: int
    intent: str
    response: str
    steps: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    total_llm_calls: int = 0
    error: str | None = None


# ======================================================================
# Observation extraction
# ======================================================================

def extract_page_observation(page: Page, max_chars: int = 4000) -> str:
    """Extract a text representation of the current page state."""
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    url = page.url

    try:
        text = page.inner_text("body")
    except Exception:
        text = ""

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    return f"URL: {url}\nTitle: {title}\n\n{text}"


# ======================================================================
# LLM action decision
# ======================================================================

SYSTEM_PROMPT = """You are a web browsing agent. You interact with web pages to complete tasks.

You will receive:
1. A TASK to complete
2. The current page OBSERVATION (URL, title, visible text)
3. Your previous actions

Based on this, decide your next action. Respond in EXACTLY this JSON format:

{"thought": "brief reasoning", "action": "ACTION_TYPE", "args": {"key": "value"}}

Available actions:
- {"action": "click", "args": {"text": "button or link text to click"}}
- {"action": "type", "args": {"text": "text to type", "field": "label or placeholder of input field"}}
- {"action": "goto", "args": {"url": "full URL to navigate to"}}
- {"action": "select", "args": {"text": "option text", "field": "select field label"}}
- {"action": "scroll", "args": {"direction": "down" or "up"}}
- {"action": "answer", "args": {"text": "your final answer to the task"}}
- {"action": "fail", "args": {"reason": "why the task cannot be completed"}}

When you have found the answer, use the "answer" action.
When the task requires a specific value, be precise.
Always think step by step before acting."""


def build_user_message(
    intent: str,
    observation: str,
    history: list[StepRecord],
) -> str:
    parts = [f"TASK: {intent}\n"]
    if history:
        parts.append("PREVIOUS ACTIONS:")
        for h in history[-5:]:
            parts.append(f"  Step {h.step}: {h.action}")
        parts.append("")
    parts.append(f"CURRENT PAGE:\n{observation}")
    return "\n".join(parts)


def parse_llm_action(raw: str) -> dict[str, Any]:
    """Extract JSON action from LLM response, tolerating markdown fences."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action": "fail", "args": {"reason": f"Could not parse: {raw[:200]}"}}


# ======================================================================
# Action execution
# ======================================================================

def execute_action(page: Page, action: dict[str, Any]) -> tuple[bool, str]:
    """Execute a browser action. Returns (success, description)."""
    act = action.get("action", "")
    args = action.get("args", {})

    try:
        if act == "click":
            text = args.get("text", "")
            loc = page.get_by_text(text, exact=False).first
            loc.click(timeout=ACTION_TIMEOUT_MS)
            page.wait_for_load_state("domcontentloaded", timeout=PAGE_LOAD_WAIT_MS)
            return True, f"click('{text}')"

        elif act == "type":
            field_label = args.get("field", "")
            text = args.get("text", "")
            loc = page.get_by_label(field_label, exact=False).first
            loc.fill(text)
            return True, f"type('{field_label}', '{text}')"

        elif act == "goto":
            url = args.get("url", "")
            page.goto(url, timeout=ACTION_TIMEOUT_MS)
            page.wait_for_load_state("domcontentloaded", timeout=PAGE_LOAD_WAIT_MS)
            return True, f"goto('{url}')"

        elif act == "select":
            field_label = args.get("field", "")
            text = args.get("text", "")
            loc = page.get_by_label(field_label, exact=False).first
            loc.select_option(label=text)
            return True, f"select('{field_label}', '{text}')"

        elif act == "scroll":
            direction = args.get("direction", "down")
            delta = -500 if direction == "up" else 500
            page.mouse.wheel(0, delta)
            page.wait_for_timeout(500)
            return True, f"scroll({direction})"

        elif act == "answer":
            return True, f"answer('{args.get('text', '')}')"

        elif act == "fail":
            return False, f"fail('{args.get('reason', '')}')"

        else:
            return False, f"unknown action: {act}"

    except Exception as e:
        return False, f"error executing {act}: {e}"


# ======================================================================
# WebAgent — the base agent
# ======================================================================

class WebAgent:
    """Baseline web browsing agent using LLM + Playwright."""

    def __init__(
        self,
        llm_model: str = "qwen3.5-flash",
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        max_steps: int = MAX_STEPS,
        headless: bool = True,
    ):
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.headless = headless
        self._client = OpenAI(
            api_key=llm_api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=llm_base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def _call_llm(self, intent: str, observation: str, history: list[StepRecord]) -> str:
        user_msg = build_user_message(intent, observation, history)
        resp = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    def run_task(
        self,
        intent: str,
        start_url: str,
        task_id: int = -1,
        page: Page | None = None,
    ) -> AgentResult:
        """Run a single WebArena task. If page is provided, reuse it."""
        result = AgentResult(task_id=task_id, intent=intent, response="")
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

            for step_num in range(self.max_steps):
                obs = extract_page_observation(page)

                step_t0 = time.perf_counter()
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
                    break

                if act_type == "fail":
                    result.response = ""
                    result.error = action.get("args", {}).get("reason", "unknown")
                    result.steps.append(StepRecord(
                        step=step_num, action=f"fail: {result.error}",
                        observation=obs[:200], llm_response=raw_llm,
                        elapsed_ms=elapsed, success=False,
                    ))
                    break

                success, desc = execute_action(page, action)
                result.steps.append(StepRecord(
                    step=step_num, action=desc,
                    observation=obs[:200], llm_response=raw_llm,
                    elapsed_ms=elapsed, success=success,
                ))

                if not success:
                    log.warning("Step %d failed: %s", step_num, desc)

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
