"""
PlaywrightComputer — browser-based Computer implementation.

Wraps a Playwright Page into the Computer protocol expected
by CerebellumAgent. Handles screenshots, clicks, typing, and
all CUA actions through Playwright's automation API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class PlaywrightComputer:
    """
    Computer implementation backed by Playwright browser automation.

    Usage::

        from playwright.async_api import async_playwright
        from digital_cerebellum.agent.playwright_computer import PlaywrightComputer

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            await page.goto("https://example.com")

            computer = PlaywrightComputer(page)
            screenshot = await computer.screenshot()
    """

    def __init__(self, page: Any, *, delay_ms: int = 50):
        self._page = page
        self._delay_ms = delay_ms

    async def screenshot(self) -> bytes:
        return await self._page.screenshot(type="png", full_page=False)

    async def click(self, x: int, y: int, button: str = "left") -> None:
        await self._page.mouse.click(x, y, button=button)
        await self._settle()

    async def double_click(self, x: int, y: int) -> None:
        await self._page.mouse.dblclick(x, y)
        await self._settle()

    async def type(self, text: str) -> None:
        await self._page.keyboard.type(text, delay=self._delay_ms)

    async def keypress(self, keys: list[str]) -> None:
        for key in keys:
            await self._page.keyboard.press(key)
        await self._settle()

    async def scroll(
        self, x: int, y: int, scroll_x: int, scroll_y: int
    ) -> None:
        await self._page.mouse.move(x, y)
        await self._page.mouse.wheel(scroll_x, scroll_y)
        await self._settle()

    async def move(self, x: int, y: int) -> None:
        await self._page.mouse.move(x, y)

    async def drag(self, path: list[dict[str, int]]) -> None:
        if not path:
            return
        start = path[0]
        await self._page.mouse.move(start["x"], start["y"])
        await self._page.mouse.down()
        for point in path[1:]:
            await self._page.mouse.move(point["x"], point["y"])
        await self._page.mouse.up()
        await self._settle()

    async def wait(self) -> None:
        await asyncio.sleep(2.0)

    async def _settle(self) -> None:
        """Brief pause for UI to settle after an action."""
        await asyncio.sleep(0.1)
