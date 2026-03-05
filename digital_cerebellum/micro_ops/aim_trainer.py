"""
AimTrainerEnv — simulated aim trainer for GUI control learning.

Models the core dynamics of an aim trainer / click target game:
  - Target appears at random position on screen
  - Agent moves cursor toward target and clicks
  - Reward: +1 for hit (cursor within radius), penalty for miss/time
  - Target respawns at new position after hit or timeout

This environment implements the MicroOpEngine's Environment protocol
and serves as the proof-of-concept for Phase 8b: continuous GUI control
learned through cerebellar prediction errors.

State vector: [cursor_x, cursor_y, target_x, target_y, dx, dy,
               distance, time_remaining, target_radius]
  - All positions normalised to [0, 1]
  - dx/dy = target - cursor (signed direction)
  - distance = euclidean distance (normalised)
  - time_remaining = fraction of timeout remaining

Action vector: [move_x, move_y, click_signal]
  - move_x/y in [-1, 1] → scaled to pixel movement
  - click_signal > 0 → attempt click
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AimTrainerConfig:
    """Configuration for the aim trainer environment."""
    screen_w: float = 800.0
    screen_h: float = 600.0
    target_radius: float = 30.0
    target_radius_min: float = 15.0
    target_shrink_rate: float = 0.1
    move_speed: float = 60.0
    timeout_steps: int = 120
    miss_penalty: float = -0.3
    hit_reward: float = 2.0
    noise: float = 0.3
    adaptive_difficulty: bool = True
    auto_click: bool = True


class AimTrainerEnv:
    """
    Simulated aim trainer implementing the Environment protocol.

    The cerebellum must learn:
      1. Motor mapping: action vector → cursor movement
      2. Targeting: move toward target (forward model predicts trajectory)
      3. Timing: click when close enough (prediction error → click signal)
      4. Adaptation: targets get smaller as performance improves

    This mirrors how the biological cerebellum learns reaching movements:
    initial errors are large, the forward model refines over hundreds of
    trials, and movements become fast and accurate.
    """

    def __init__(self, cfg: AimTrainerConfig | None = None):
        self.cfg = cfg or AimTrainerConfig()

        self._cursor = np.array([
            self.cfg.screen_w / 2,
            self.cfg.screen_h / 2,
        ], dtype=np.float32)

        self._target = np.zeros(2, dtype=np.float32)
        self._target_radius = self.cfg.target_radius
        self._steps_this_target = 0
        self._total_steps = 0
        self._hits = 0
        self._misses = 0
        self._attempts = 0
        self._targets_shown = 0
        self._episode_hits: list[bool] = []

        self._spawn_target()

    @property
    def state_dim(self) -> int:
        return 9

    @property
    def action_dim(self) -> int:
        return 3

    def observe(self) -> np.ndarray:
        sw, sh = self.cfg.screen_w, self.cfg.screen_h

        cx = self._cursor[0] / sw
        cy = self._cursor[1] / sh
        tx = self._target[0] / sw
        ty = self._target[1] / sh
        dx = tx - cx
        dy = ty - cy
        dist = np.sqrt(dx**2 + dy**2)
        time_left = max(0, 1.0 - self._steps_this_target / self.cfg.timeout_steps)
        radius_norm = self._target_radius / max(sw, sh)

        return np.array([
            cx, cy, tx, ty, dx, dy, dist, time_left, radius_norm,
        ], dtype=np.float32)

    def execute(self, action: np.ndarray) -> float:
        action = np.clip(action[:self.action_dim], -1.0, 1.0)
        self._total_steps += 1
        self._steps_this_target += 1
        self._last_hit = False

        old_distance = float(np.linalg.norm(self._cursor - self._target))

        move_x = float(action[0]) * self.cfg.move_speed
        move_y = float(action[1]) * self.cfg.move_speed
        click = (float(action[2]) > 0) if not self.cfg.auto_click else False

        self._cursor[0] += move_x + np.random.randn() * self.cfg.noise
        self._cursor[1] += move_y + np.random.randn() * self.cfg.noise
        self._cursor[0] = np.clip(self._cursor[0], 0, self.cfg.screen_w)
        self._cursor[1] = np.clip(self._cursor[1], 0, self.cfg.screen_h)

        new_distance = float(np.linalg.norm(self._cursor - self._target))
        max_dist = np.sqrt(self.cfg.screen_w**2 + self.cfg.screen_h**2)

        approach = (old_distance - new_distance) / max_dist
        reward = approach * 5.0

        proximity_bonus = max(0, 1.0 - new_distance / max_dist) * 0.1
        reward += proximity_bonus

        if self.cfg.auto_click and new_distance <= self._target_radius:
            click = True

        if click:
            self._attempts += 1
            if new_distance <= self._target_radius:
                speed_bonus = max(0, 1.0 - self._steps_this_target / self.cfg.timeout_steps)
                reward = self.cfg.hit_reward + speed_bonus
                self._last_hit = True
                self._hits += 1
                self._episode_hits.append(True)
                self._maybe_increase_difficulty()
                self._spawn_target()
            else:
                reward = self.cfg.miss_penalty
                self._misses += 1
                self._episode_hits.append(False)

        if self._steps_this_target >= self.cfg.timeout_steps:
            reward = self.cfg.miss_penalty
            self._episode_hits.append(False)
            self._spawn_target()

        return reward

    def _spawn_target(self):
        margin = self._target_radius * 2
        self._target[0] = np.random.uniform(margin, self.cfg.screen_w - margin)
        self._target[1] = np.random.uniform(margin, self.cfg.screen_h - margin)
        self._steps_this_target = 0
        self._targets_shown += 1

    def _maybe_increase_difficulty(self):
        if not self.cfg.adaptive_difficulty:
            return
        recent = self._episode_hits[-10:]
        if len(recent) >= 5 and sum(recent) / len(recent) > 0.7:
            self._target_radius = max(
                self.cfg.target_radius_min,
                self._target_radius * (1 - self.cfg.target_shrink_rate),
            )

    def reset(self):
        self._cursor = np.array([
            self.cfg.screen_w / 2,
            self.cfg.screen_h / 2,
        ], dtype=np.float32)
        self._steps_this_target = 0
        self._total_steps = 0
        self._hits = 0
        self._misses = 0
        self._attempts = 0
        self._targets_shown = 0
        self._episode_hits = []
        self._target_radius = self.cfg.target_radius
        self._spawn_target()

    @property
    def stats(self) -> dict:
        hit_rate = self._hits / max(1, self._attempts)
        return {
            "total_steps": self._total_steps,
            "targets_shown": self._targets_shown,
            "hits": self._hits,
            "misses": self._misses,
            "attempts": self._attempts,
            "hit_rate": round(hit_rate, 3),
            "target_radius": round(self._target_radius, 1),
            "cursor": [round(float(self._cursor[0]), 1), round(float(self._cursor[1]), 1)],
        }
