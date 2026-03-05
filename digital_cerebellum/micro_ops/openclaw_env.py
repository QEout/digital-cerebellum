"""
OpenClawEnvironment — bridges OpenClaw's GUI tools with the cerebellar loop.

Architecture:
  OpenClaw provides the eyes (screenshot) and hands (mouse/keyboard).
  Digital Cerebellum provides the brain (prediction, correction, learning).

  Every cerebellar step:
    1. screenshot → ScreenStateEncoder → state vector
    2. GUIController computes action (cortex + cerebellar correction)
    3. GUIActionSpace decodes action → OpenClaw tool calls
    4. OpenClaw executes (mouse_move, left_click, type, etc.)
    5. Forward model compares prediction to reality → SPE → learning

This runs at the speed of OpenClaw's screenshot + execution round-trip,
typically 50-200ms. The cerebellar computation itself adds <1ms overhead.

Usage:
    from openclaw_sdk import OpenClawClient
    from digital_cerebellum.micro_ops.openclaw_env import OpenClawEnvironment

    async with OpenClawClient.connect() as client:
        env = OpenClawEnvironment(client, agent_id="my-agent")
        controller = GUIController(env.state_dim, env.action_dim)

        for episode in range(20):
            for step in range(100):
                result = await env.async_step(controller)
            controller.decay_noise()
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from digital_cerebellum.micro_ops.screen_state_encoder import (
    ScreenStateEncoder,
    ScreenStateConfig,
)
from digital_cerebellum.micro_ops.gui_action_space import (
    GUIActionSpace,
    GUIActionSpaceConfig,
    GUIAction,
    GUIActionType,
)

log = logging.getLogger(__name__)


@dataclass
class OpenClawEnvConfig:
    """Configuration for the OpenClaw GUI environment."""
    screen_w: int = 1024
    screen_h: int = 768
    state_strategy: str = "downsample"
    downsample_size: tuple[int, int] = (16, 16)
    move_scale: float = 100.0
    click_threshold: float = 0.3
    action_dim: int = 5


class OpenClawEnvironment:
    """
    Implements the MicroOpEngine Environment protocol using OpenClaw's
    real GUI control tools (screenshot, mouse_move, left_click, etc.).

    This is the bridge that gives the cerebellum real eyes and hands.
    """

    def __init__(
        self,
        openclaw_client: Any,
        agent_id: str = "default",
        cfg: OpenClawEnvConfig | None = None,
    ):
        self.client = openclaw_client
        self.agent_id = agent_id
        self.cfg = cfg or OpenClawEnvConfig()

        self.screen_encoder = ScreenStateEncoder(ScreenStateConfig(
            strategy=self.cfg.state_strategy,
            downsample_size=self.cfg.downsample_size,
            screen_w=self.cfg.screen_w,
            screen_h=self.cfg.screen_h,
            grayscale=True,
        ))

        self.action_space = GUIActionSpace(GUIActionSpaceConfig(
            screen_w=self.cfg.screen_w,
            screen_h=self.cfg.screen_h,
            move_scale=self.cfg.move_scale,
            click_threshold=self.cfg.click_threshold,
            action_dim=self.cfg.action_dim,
        ))

        self._last_screenshot: np.ndarray | None = None
        self._cursor_x: float = self.cfg.screen_w / 2
        self._cursor_y: float = self.cfg.screen_h / 2
        self._step_count = 0

    @property
    def state_dim(self) -> int:
        return self.screen_encoder.state_dim

    @property
    def action_dim(self) -> int:
        return self.action_space.action_dim

    # ── Synchronous interface (for MicroOpEngine compatibility) ──────

    def observe(self) -> np.ndarray:
        """
        Synchronous observe — runs the async screenshot in an event loop.
        For async code, use async_observe() directly.
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Use async_observe() in async context. "
                "observe() is for synchronous MicroOpEngine only."
            )
        except RuntimeError:
            return asyncio.run(self.async_observe())

    def execute(self, action: np.ndarray) -> float:
        """Synchronous execute — wraps async_execute."""
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError("Use async_execute() in async context.")
        except RuntimeError:
            return asyncio.run(self.async_execute(action))

    # ── Async interface (primary) ────────────────────────────────────

    async def async_observe(self) -> np.ndarray:
        """
        Take a screenshot via OpenClaw and encode it as a state vector.

        Calls OpenClaw's `screenshot` tool, decodes the base64 PNG,
        and feeds it through the ScreenStateEncoder.
        """
        agent = self.client.get_agent(self.agent_id)

        screenshot_result = await agent.execute(
            "Use the screenshot tool to capture the current screen"
        )

        image = self._decode_screenshot(screenshot_result)
        if image is not None:
            self._last_screenshot = image
            return self.screen_encoder.encode_image(image)

        if self._last_screenshot is not None:
            return self.screen_encoder.encode_image(self._last_screenshot)

        return np.zeros(self.state_dim, dtype=np.float32)

    async def async_execute(self, action: np.ndarray) -> float:
        """
        Decode action vector and execute via OpenClaw's GUI tools.

        Returns a reward signal based on execution success.
        """
        self._step_count += 1
        gui_action = self.action_space.decode(action)
        agent = self.client.get_agent(self.agent_id)

        try:
            await self._execute_gui_action(agent, gui_action)
            self._cursor_x += gui_action.dx
            self._cursor_y += gui_action.dy
            self._cursor_x = np.clip(self._cursor_x, 0, self.cfg.screen_w)
            self._cursor_y = np.clip(self._cursor_y, 0, self.cfg.screen_h)
            return 0.0
        except Exception as e:
            log.warning("OpenClaw execute failed: %s", e)
            return -1.0

    async def async_step(self, controller: Any) -> dict:
        """
        Run one full cerebellar step: observe → compute → execute.

        Parameters
        ----------
        controller : GUIController instance

        Returns
        -------
        Dict with step metrics (reward, spe, latency, action_type)
        """
        import time
        t0 = time.perf_counter()

        state = await self.async_observe()

        cortex = controller.cortex_signal(state)
        correction = controller.cerebellar_correction(state)
        final_action = np.clip(cortex + correction, -1.0, 1.0)

        gui_action = self.action_space.decode(final_action)

        reward = await self.async_execute(final_action)

        next_state = await self.async_observe()

        if controller._prev_state is not None and controller._prev_action is not None:
            controller.forward_model.learn(
                controller._prev_state, controller._prev_action, state,
            )

        prediction = controller.forward_model.predict(state, final_action)
        spe_vec = controller.forward_model.compute_spe(
            prediction.predicted_next_state, next_state,
        )
        spe = float(np.linalg.norm(spe_vec))

        controller._learn_correction(state, cortex, reward, spe)
        controller._prev_state = state.copy()
        controller._prev_action = final_action.copy()
        controller._step += 1

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "step": controller._step,
            "reward": reward,
            "spe": spe,
            "latency_ms": latency_ms,
            "action_type": gui_action.action_type.value,
            "cursor": (self._cursor_x, self._cursor_y),
        }

    # ── Internal helpers ─────────────────────────────────────────────

    async def _execute_gui_action(self, agent: Any, action: GUIAction):
        """Translate a GUIAction into OpenClaw tool calls."""
        if action.action_type == GUIActionType.NOOP:
            return

        if action.action_type in (GUIActionType.MOVE, GUIActionType.CLICK):
            target_x = self._cursor_x + action.dx
            target_y = self._cursor_y + action.dy
            await agent.execute(
                f"Use mouse_move to move cursor to coordinates "
                f"({int(target_x)}, {int(target_y)})"
            )

        if action.action_type == GUIActionType.CLICK:
            if action.right_click:
                await agent.execute("Use right_click at current position")
            else:
                await agent.execute("Use left_click at current position")

        elif action.action_type == GUIActionType.SCROLL:
            direction = "down" if action.scroll_amount > 0 else "up"
            clicks = abs(int(action.scroll_amount))
            await agent.execute(
                f"Use scroll to scroll {direction} {clicks} clicks"
            )

        elif action.action_type == GUIActionType.DRAG:
            target_x = self._cursor_x + action.dx
            target_y = self._cursor_y + action.dy
            await agent.execute(
                f"Use left_click_drag to drag from current position "
                f"to ({int(target_x)}, {int(target_y)})"
            )

    def _decode_screenshot(self, result: Any) -> np.ndarray | None:
        """Extract image array from OpenClaw's screenshot response."""
        try:
            content = result.content if hasattr(result, "content") else str(result)

            if "base64" in content.lower() or len(content) > 1000:
                b64_data = self._extract_base64(content)
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
                    try:
                        from PIL import Image
                        img = Image.open(io.BytesIO(img_bytes)).convert("L")
                        return np.array(img, dtype=np.uint8)
                    except ImportError:
                        raw = np.frombuffer(img_bytes, dtype=np.uint8)
                        side = int(np.sqrt(len(raw)))
                        if side * side == len(raw):
                            return raw.reshape(side, side)
                        return raw[:self.cfg.screen_w * self.cfg.screen_h].reshape(
                            self.cfg.screen_h, -1
                        )
        except Exception as e:
            log.debug("Screenshot decode failed: %s", e)

        return None

    @staticmethod
    def _extract_base64(text: str) -> str | None:
        """Extract base64-encoded data from text (handles data URIs)."""
        if "base64," in text:
            return text.split("base64,", 1)[1].split('"')[0].split("'")[0].strip()
        import re
        match = re.search(r'[A-Za-z0-9+/]{100,}={0,2}', text)
        return match.group(0) if match else None


# ── Convenience function for quick integration ────────────────────

async def run_openclaw_cerebellum(
    openclaw_client: Any,
    agent_id: str = "default",
    episodes: int = 10,
    steps_per_episode: int = 50,
    verbose: bool = True,
) -> list[dict]:
    """
    Ready-to-use function: connect OpenClaw's GUI to the cerebellum.

    Usage:
        from openclaw_sdk import OpenClawClient
        from digital_cerebellum.micro_ops.openclaw_env import run_openclaw_cerebellum

        async with OpenClawClient.connect() as client:
            results = await run_openclaw_cerebellum(client, "my-agent")
    """
    from digital_cerebellum.micro_ops.gui_controller import (
        GUIController,
        GUIControlConfig,
    )

    env = OpenClawEnvironment(openclaw_client, agent_id)
    controller = GUIController(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=GUIControlConfig(
            cortex_gain=1.0,
            cortex_noise=0.8,
            noise_decay=0.9,
        ),
    )

    all_results = []
    for ep in range(episodes):
        ep_data = {"episode": ep + 1, "steps": []}
        for step in range(steps_per_episode):
            step_result = await env.async_step(controller)
            ep_data["steps"].append(step_result)

        controller.decay_noise()

        mean_spe = np.mean([s["spe"] for s in ep_data["steps"]])
        mean_lat = np.mean([s["latency_ms"] for s in ep_data["steps"]])
        ep_data["mean_spe"] = float(mean_spe)
        ep_data["mean_latency_ms"] = float(mean_lat)
        all_results.append(ep_data)

        if verbose:
            print(
                f"  Episode {ep+1:3d}: SPE={mean_spe:.4f}, "
                f"latency={mean_lat:.0f}ms/step, "
                f"noise={controller._noise_scale:.3f}"
            )

    return all_results
