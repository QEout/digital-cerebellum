"""
Demo: CerebellumAgent — GPT-5.4 Desktop Agent with learning.

This demo launches a Playwright browser, gives a task to the agent,
and shows the cerebellum learning in action:

    Round 1: GPT-5.4 reasons through the task (slow path, uses tokens)
    Round 2: Same task → SkillStore hits → instant replay (0 tokens)

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY=sk-...

    # Install dependencies
    pip install openai playwright
    playwright install chromium

    # Run
    python examples/agent_demo.py

    # With custom task
    python examples/agent_demo.py --task "Search for 'digital cerebellum' on Google"
"""

import argparse
import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("agent_demo")


async def main(task: str, url: str, rounds: int, headless: bool):
    from openai import OpenAI
    from playwright.async_api import async_playwright

    from digital_cerebellum.agent import CerebellumAgent
    from digital_cerebellum.agent.cua_loop import AgentConfig
    from digital_cerebellum.agent.playwright_computer import PlaywrightComputer

    client = OpenAI()
    config = AgentConfig(
        model="gpt-5.4",
        reasoning_effort="low",
        display_width=1440,
        display_height=900,
    )
    agent = CerebellumAgent(client, config)

    # Try to load previous learning
    try:
        agent.load()
        log.info("Loaded previous agent state")
    except Exception:
        log.info("No previous state found — starting fresh")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(
            viewport={"width": 1440, "height": 900}
        )
        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        computer = PlaywrightComputer(page)

        for round_num in range(1, rounds + 1):
            log.info("=" * 60)
            log.info("Round %d / %d", round_num, rounds)
            log.info("Task: %s", task)
            log.info("=" * 60)

            # Check proactive suggestions
            suggestions = agent.suggest()
            if suggestions:
                log.info("Proactive suggestions:")
                for s in suggestions:
                    log.info("  → %s (confidence=%.2f)",
                             s.get("action", "?"), s.get("confidence", 0))

            t0 = time.time()
            result = await agent.run(task, computer)
            elapsed = time.time() - t0

            log.info("─" * 60)
            log.info("Result:")
            log.info("  Success:     %s", result.success)
            log.info("  Skill hit:   %s", result.skill_hit)
            log.info("  Actions:     %d", result.actions_executed)
            log.info("  Tokens:      %d", result.tokens_used)
            log.info("  Time:        %.2fs", elapsed)
            if result.errors:
                log.info("  Errors:      %s", result.errors)
            if result.final_text:
                log.info("  Output:      %s", result.final_text[:200])
            log.info("─" * 60)

            if round_num < rounds:
                # Reset browser for next round
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(1)

        # Save learned state
        agent.save()
        log.info("Agent state saved. Skills persist for next session.")

        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CerebellumAgent Demo")
    parser.add_argument(
        "--task",
        default='Search for "digital cerebellum AI agent" and tell me the first result',
        help="Task for the agent to perform",
    )
    parser.add_argument(
        "--url",
        default="https://www.google.com",
        help="Starting URL",
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Number of rounds (round 2+ should hit skill cache)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser in headless mode",
    )
    args = parser.parse_args()
    asyncio.run(main(args.task, args.url, args.rounds, args.headless))
