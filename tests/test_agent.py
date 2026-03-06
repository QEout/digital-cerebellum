"""Tests for CerebellumAgent — GPT-5.4 CUA integration."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from digital_cerebellum.agent.cua_loop import (
    CerebellumAgent,
    AgentConfig,
    TaskResult,
    ActionRecord,
    _describe_actions,
    _extract_computer_call,
    _extract_text,
)


# ── Mock Computer ────────────────────────────────────────────────


class MockComputer:
    """In-memory computer for testing."""

    def __init__(self):
        self.actions_executed: list[dict] = []
        self.screenshot_count = 0
        self._screenshot_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    async def screenshot(self) -> bytes:
        self.screenshot_count += 1
        return self._screenshot_bytes

    async def click(self, x: int, y: int, button: str = "left") -> None:
        self.actions_executed.append({"type": "click", "x": x, "y": y, "button": button})

    async def double_click(self, x: int, y: int) -> None:
        self.actions_executed.append({"type": "double_click", "x": x, "y": y})

    async def type(self, text: str) -> None:
        self.actions_executed.append({"type": "type", "text": text})

    async def keypress(self, keys: list[str]) -> None:
        self.actions_executed.append({"type": "keypress", "keys": keys})

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.actions_executed.append({
            "type": "scroll", "x": x, "y": y,
            "scroll_x": scroll_x, "scroll_y": scroll_y,
        })

    async def move(self, x: int, y: int) -> None:
        self.actions_executed.append({"type": "move", "x": x, "y": y})

    async def drag(self, path: list[dict[str, int]]) -> None:
        self.actions_executed.append({"type": "drag", "path": path})

    async def wait(self) -> None:
        self.actions_executed.append({"type": "wait"})


# ── Mock OpenAI response objects ─────────────────────────────────


class MockAction:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockComputerCall:
    type = "computer_call"

    def __init__(self, call_id: str, actions: list):
        self.id = call_id
        self.actions = actions


class MockMessage:
    type = "message"

    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]


class MockUsage:
    def __init__(self, total_tokens: int = 100):
        self.total_tokens = total_tokens


class MockResponse:
    def __init__(self, output: list, response_id: str = "resp_1"):
        self.id = response_id
        self.output = output
        self.usage = MockUsage()


# ── Tests ────────────────────────────────────────────────────────


class TestHelpers:
    def test_describe_actions_click(self):
        actions = [{"type": "click", "x": 100, "y": 200}]
        assert "click(100,200)" in _describe_actions(actions)

    def test_describe_actions_type(self):
        actions = [{"type": "type", "text": "hello"}]
        assert 'type("hello")' in _describe_actions(actions)

    def test_describe_actions_chain(self):
        actions = [
            {"type": "click", "x": 10, "y": 20},
            {"type": "type", "text": "search"},
        ]
        desc = _describe_actions(actions)
        assert "→" in desc

    def test_describe_actions_empty(self):
        assert _describe_actions([]) == "screenshot"

    def test_extract_computer_call_present(self):
        call = MockComputerCall("call_1", [
            MockAction(type="click", x=10, y=20, button="left",
                       text=None, keys=None, scroll_x=None,
                       scroll_y=None, path=None),
        ])
        resp = MockResponse([call])
        result = _extract_computer_call(resp)
        assert result is not None
        assert result["call_id"] == "call_1"
        assert len(result["actions"]) == 1

    def test_extract_computer_call_absent(self):
        resp = MockResponse([MockMessage("done")])
        assert _extract_computer_call(resp) is None

    def test_extract_text(self):
        resp = MockResponse([MockMessage("The answer is 42")])
        assert "42" in _extract_text(resp)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == "gpt-5.4"
        assert cfg.max_turns == 50
        assert cfg.display_width == 1440

    def test_custom(self):
        cfg = AgentConfig(model="gpt-5.4-pro", max_turns=10)
        assert cfg.model == "gpt-5.4-pro"
        assert cfg.max_turns == 10


class TestActionRecord:
    def test_creation(self):
        r = ActionRecord(action_type="click", args={"x": 1, "y": 2})
        assert r.action_type == "click"
        assert r.args["x"] == 1
        assert r.timestamp > 0


class TestTaskResult:
    def test_creation(self):
        r = TaskResult(
            task="test", success=True, actions_executed=5,
            tokens_used=100, skill_hit=False, time_seconds=1.5,
            final_text="done",
        )
        assert r.success
        assert r.actions_executed == 5
        assert r.errors == []


class TestMockComputer:
    def test_click(self):
        c = MockComputer()
        asyncio.run(c.click(10, 20, "left"))
        assert len(c.actions_executed) == 1
        assert c.actions_executed[0]["type"] == "click"

    def test_screenshot(self):
        c = MockComputer()
        data = asyncio.run(c.screenshot())
        assert data.startswith(b"\x89PNG")
        assert c.screenshot_count == 1

    def test_type(self):
        c = MockComputer()
        asyncio.run(c.type("hello"))
        assert c.actions_executed[0]["text"] == "hello"


class TestCerebellumAgentInit:
    def test_init(self):
        client = MagicMock()
        agent = CerebellumAgent(client)
        assert agent.config.model == "gpt-5.4"
        assert agent.monitor is not None
        assert agent.habits is not None
        assert agent.rhythm is not None


class TestCerebellumAgentRun:
    """Test the CUA loop with mocked OpenAI responses."""

    def test_single_turn_completes(self):
        """Model returns one computer_call then a text message."""
        client = MagicMock()

        click_action = MockAction(
            type="click", x=100, y=200, button="left",
            text=None, keys=None, scroll_x=None,
            scroll_y=None, path=None,
        )
        first_resp = MockResponse(
            [MockComputerCall("call_1", [click_action])],
            response_id="resp_1",
        )
        second_resp = MockResponse(
            [MockMessage("Task completed successfully")],
            response_id="resp_2",
        )

        client.responses.create = MagicMock(side_effect=[first_resp, second_resp])

        agent = CerebellumAgent(client, AgentConfig(max_turns=5))
        computer = MockComputer()

        result = asyncio.run(agent.run("Click the button", computer))

        assert result.actions_executed >= 1
        assert result.tokens_used > 0
        assert not result.skill_hit
        assert "completed" in result.final_text.lower()

    def test_max_turns_limit(self):
        """Agent should stop after max_turns even if model keeps returning actions."""
        client = MagicMock()

        click = MockAction(
            type="click", x=50, y=50, button="left",
            text=None, keys=None, scroll_x=None,
            scroll_y=None, path=None,
        )
        resp = MockResponse(
            [MockComputerCall("call_loop", [click])],
            response_id="resp_loop",
        )
        client.responses.create = MagicMock(return_value=resp)

        agent = CerebellumAgent(client, AgentConfig(max_turns=3))
        computer = MockComputer()

        result = asyncio.run(agent.run("Loop task", computer))

        assert result.actions_executed <= 3 * 1

    def test_immediate_text_response(self):
        """Model returns text without any computer_call."""
        client = MagicMock()
        resp = MockResponse(
            [MockMessage("I cannot do that")],
            response_id="resp_text",
        )
        client.responses.create = MagicMock(return_value=resp)

        agent = CerebellumAgent(client)
        computer = MockComputer()

        result = asyncio.run(agent.run("Impossible task", computer))
        assert result.actions_executed == 0
        assert "cannot" in result.final_text.lower()


class TestCerebellumAgentExecuteAction:
    """Test individual action execution."""

    def test_execute_all_action_types(self):
        client = MagicMock()
        agent = CerebellumAgent(client)
        computer = MockComputer()

        actions = [
            {"type": "click", "x": 10, "y": 20, "button": "left"},
            {"type": "double_click", "x": 30, "y": 40},
            {"type": "type", "text": "hello world"},
            {"type": "keypress", "keys": ["Enter"]},
            {"type": "scroll", "x": 0, "y": 0, "scroll_x": 0, "scroll_y": -100},
            {"type": "move", "x": 50, "y": 60},
            {"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 100, "y": 100}]},
            {"type": "wait"},
        ]

        async def run_all():
            for action in actions:
                await agent._execute_action(computer, action)

        asyncio.run(run_all())

        assert len(computer.actions_executed) == 8
        assert len(agent._session_actions) == 8


class TestSaveLoad:
    def test_save_load_cycle(self, tmp_path):
        client = MagicMock()
        agent = CerebellumAgent(client)

        # Record a habit
        agent.habits.record("check email", domain="desktop")

        save_path = str(tmp_path / "agent-state")
        agent.save(save_path)

        agent2 = CerebellumAgent(client)
        agent2.load(save_path)

        assert len(agent2.habits._records) == 1
