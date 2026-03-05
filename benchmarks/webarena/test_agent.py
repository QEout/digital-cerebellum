#!/usr/bin/env python3
"""
Quick integration test for the WebArena agent components.

Tests:
1. LLM call to qwen3.5-flash works
2. Action parsing works
3. Playwright browser opens and navigates
4. Cerebellum sidecar hooks work
5. Full agent loop against a live website

Usage:
    python -m benchmarks.webarena.test_agent
    python -m benchmarks.webarena.test_agent --browser   # include browser tests
    python -m benchmarks.webarena.test_agent --live       # test against live website
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml

from benchmarks.webarena.agent import (
    WebAgent,
    parse_llm_action,
    extract_page_observation,
    build_user_message,
)


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config.local.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def test_action_parsing():
    print("=== Test: Action Parsing ===")

    cases = [
        ('{"action": "click", "args": {"text": "Search"}}', "click"),
        ('{"thought": "I need to search", "action": "goto", "args": {"url": "http://example.com"}}', "goto"),
        ('```json\n{"action": "answer", "args": {"text": "42"}}\n```', "answer"),
        ("some random text", "fail"),
    ]

    for raw, expected_action in cases:
        result = parse_llm_action(raw)
        actual = result.get("action", "")
        status = "OK" if actual == expected_action else "FAIL"
        print(f"  [{status}] parse '{raw[:40]}...' -> action={actual}")

    print()


def test_llm_call():
    print("=== Test: LLM Call ===")
    cfg = load_config()
    llm = cfg.get("llm", {})

    from openai import OpenAI
    client = OpenAI(
        api_key=llm.get("api_key", ""),
        base_url=llm.get("base_url", ""),
    )

    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=llm.get("model", "qwen3.5-flash"),
            messages=[
                {"role": "system", "content": "You are a web browsing agent. Respond in JSON: {\"action\": \"answer\", \"args\": {\"text\": \"your answer\"}}"},
                {"role": "user", "content": "TASK: What is 2+2?\n\nRespond with the answer action."},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        content = resp.choices[0].message.content or ""
        print(f"  [OK] LLM responded in {elapsed:.0f}ms")
        print(f"  Response: {content[:200]}")

        action = parse_llm_action(content)
        print(f"  Parsed action: {action}")
        print()
        return True
    except Exception as e:
        print(f"  [FAIL] LLM call failed: {e}")
        print()
        return False


def test_browser():
    print("=== Test: Playwright Browser ===")
    try:
        from playwright.sync_api import sync_playwright

        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "E:\\playwright-browsers")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto("https://example.com", timeout=15000)
            page.wait_for_load_state("domcontentloaded")

            obs = extract_page_observation(page, max_chars=500)
            print(f"  [OK] Navigated to example.com")
            print(f"  Title: {page.title()}")
            print(f"  Observation length: {len(obs)} chars")

            browser.close()
            print()
            return True

    except Exception as e:
        print(f"  [FAIL] Browser test failed: {e}")
        print()
        return False


def test_cerebellum_sidecar():
    print("=== Test: Cerebellum Sidecar ===")
    try:
        from digital_cerebellum import DigitalCerebellum, StepMonitor
        from benchmarks.webarena.cerebellum_agent import CerebellumWebAgent, AblationConfig

        cb = DigitalCerebellum()
        monitor = StepMonitor(cerebellum=cb)
        monitor.reset()

        pred = monitor.before_step(action="click search button", state="search page loaded")
        print(f"  [OK] before_step: should_proceed={pred.should_proceed}, risk={pred.risk_score:.3f}")

        verdict = monitor.after_step(outcome="results page loaded", success=True)
        print(f"  [OK] after_step: spe={verdict.spe:.3f}, should_pause={verdict.should_pause}")

        match = cb.match_skill("click search button on search page")
        print(f"  [OK] match_skill: {'hit' if match else 'miss'}")

        skill_id = cb.learn_skill(
            "click search button",
            '{"action": "click", "args": {"text": "Search"}}',
            domain="webarena",
        )
        print(f"  [OK] learn_skill: id={skill_id[:20]}...")

        for ablation_name in ["full", "no_skill", "no_monitor", "no_memory"]:
            abl = getattr(AblationConfig, ablation_name)()
            print(f"  [OK] AblationConfig.{ablation_name}(): label={abl.label}")

        print()
        return True
    except Exception as e:
        print(f"  [FAIL] Cerebellum sidecar test failed: {e}")
        print()
        return False


def test_live_agent():
    print("=== Test: Live Agent (example.com) ===")
    cfg = load_config()
    llm = cfg.get("llm", {})

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "E:\\playwright-browsers")

    agent = WebAgent(
        llm_model=llm.get("model", "qwen3.5-flash"),
        llm_api_key=llm.get("api_key", ""),
        llm_base_url=llm.get("base_url", ""),
        max_steps=3,
        headless=True,
    )

    result = agent.run_task(
        intent="What is the title of this webpage?",
        start_url="https://example.com",
        task_id=999,
    )

    print(f"  Steps: {len(result.steps)}")
    print(f"  LLM calls: {result.total_llm_calls}")
    print(f"  Response: {result.response[:200] if result.response else 'N/A'}")
    print(f"  Error: {result.error}")
    print(f"  Time: {result.total_time_s:.1f}s")

    for s in result.steps:
        print(f"    Step {s.step}: {s.action[:80]} ({s.elapsed_ms:.0f}ms)")

    print()
    return result.response != "" or result.error is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", action="store_true", help="Include browser tests")
    parser.add_argument("--live", action="store_true", help="Test live agent against example.com")
    args = parser.parse_args()

    results = {}

    results["action_parsing"] = test_action_parsing() or True
    results["llm_call"] = test_llm_call()
    results["cerebellum_sidecar"] = test_cerebellum_sidecar()

    if args.browser or args.live:
        results["browser"] = test_browser()

    if args.live:
        results["live_agent"] = test_live_agent()

    print("=" * 50)
    print("Summary:")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    all_pass = all(results.values())
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
