"""
GUIActionSpace — maps continuous action vectors to mouse/keyboard commands.

Biological basis:
  The deep cerebellar nuclei output continuous motor signals that drive
  muscles. These aren't discrete "press button X" commands — they're
  graded force vectors that produce smooth, coordinated movements.

  GUIActionSpace translates the cerebellum's continuous output into
  the discrete/continuous hybrid space of GUI interaction:
  - Mouse movement (continuous: dx, dy)
  - Click decision (continuous → threshold: click if > 0.5)
  - Scroll (continuous: scroll_amount)
  - Key press (discrete, selected by highest activation)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GUIActionType(Enum):
    MOVE = "move"
    CLICK = "click"
    DRAG = "drag"
    SCROLL = "scroll"
    KEY = "key"
    NOOP = "noop"


@dataclass
class GUIAction:
    """A decoded GUI action ready for execution."""
    action_type: GUIActionType
    dx: float = 0.0
    dy: float = 0.0
    click: bool = False
    right_click: bool = False
    scroll_amount: float = 0.0
    key: str | None = None
    confidence: float = 0.0


@dataclass
class GUIActionSpaceConfig:
    """Configuration for the GUI action space."""
    screen_w: int = 1920
    screen_h: int = 1080
    move_scale: float = 50.0
    click_threshold: float = 0.3
    scroll_scale: float = 3.0
    enable_keys: bool = False
    action_dim: int = 5


class GUIActionSpace:
    """
    Bidirectional codec between continuous vectors and GUI commands.

    Encodes: GUI events → [-1, 1] vectors (for learning from demonstrations)
    Decodes: [-1, 1] vectors → GUIAction (for execution)

    Action vector layout (default 5D):
      [0] dx — horizontal mouse movement (-1=left, +1=right)
      [1] dy — vertical mouse movement (-1=up, +1=down)
      [2] click — click signal (>threshold = click)
      [3] right_click — right-click signal
      [4] scroll — scroll amount
    """

    def __init__(self, cfg: GUIActionSpaceConfig | None = None):
        self.cfg = cfg or GUIActionSpaceConfig()

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    def decode(self, action_vec: np.ndarray) -> GUIAction:
        """
        Convert a continuous action vector into a GUIAction.

        Parameters
        ----------
        action_vec : (action_dim,) float array in [-1, 1]

        Returns
        -------
        GUIAction with concrete dx/dy/click/scroll values
        """
        vec = np.clip(action_vec, -1.0, 1.0)

        dx = float(vec[0]) * self.cfg.move_scale
        dy = float(vec[1]) * self.cfg.move_scale
        click = float(vec[2]) > self.cfg.click_threshold if len(vec) > 2 else False
        right_click = float(vec[3]) > self.cfg.click_threshold if len(vec) > 3 else False
        scroll = float(vec[4]) * self.cfg.scroll_scale if len(vec) > 4 else 0.0

        move_mag = abs(dx) + abs(dy)

        if click:
            action_type = GUIActionType.CLICK
        elif abs(scroll) > 0.5:
            action_type = GUIActionType.SCROLL
        elif move_mag > 1.0:
            action_type = GUIActionType.MOVE
        else:
            action_type = GUIActionType.NOOP

        return GUIAction(
            action_type=action_type,
            dx=dx,
            dy=dy,
            click=click,
            right_click=right_click,
            scroll_amount=scroll,
            confidence=min(move_mag / self.cfg.move_scale, 1.0),
        )

    def encode(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        click: bool = False,
        right_click: bool = False,
        scroll: float = 0.0,
    ) -> np.ndarray:
        """
        Encode a GUI event into a continuous vector (for imitation learning).

        Parameters
        ----------
        dx, dy : mouse movement in pixels
        click : left click
        right_click : right click
        scroll : scroll amount

        Returns
        -------
        (action_dim,) float32 array in [-1, 1]
        """
        vec = np.zeros(self.cfg.action_dim, dtype=np.float32)
        vec[0] = np.clip(dx / self.cfg.move_scale, -1.0, 1.0)
        vec[1] = np.clip(dy / self.cfg.move_scale, -1.0, 1.0)
        if self.cfg.action_dim > 2:
            vec[2] = 1.0 if click else -1.0
        if self.cfg.action_dim > 3:
            vec[3] = 1.0 if right_click else -1.0
        if self.cfg.action_dim > 4:
            vec[4] = np.clip(scroll / self.cfg.scroll_scale, -1.0, 1.0)
        return vec

    def encode_absolute_move(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        click: bool = False,
    ) -> np.ndarray:
        """Encode a move from one screen position to another."""
        return self.encode(
            dx=to_x - from_x,
            dy=to_y - from_y,
            click=click,
        )
