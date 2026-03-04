"""
Simulated environments for testing the Micro-Operation Engine.

These are simple physics-based environments that demonstrate
continuous control learning. They serve as proof-of-concept
before connecting to real environments (games, Figma, robots).
"""

from __future__ import annotations

import numpy as np


class TargetTracker:
    """
    Track a moving target in 2D space.

    The agent must move its position to match a target that moves
    in a smooth pattern.  Reward = -distance to target.

    This tests the cerebellum's core capability: learning a forward
    model of how actions affect state, then using predictions to
    generate corrective movements.

    State: [agent_x, agent_y, target_x, target_y, dx, dy]
    Action: [move_x, move_y]  in [-1, 1]
    """

    def __init__(
        self,
        speed: float = 0.1,
        target_speed: float = 0.02,
        noise: float = 0.01,
    ):
        self._speed = speed
        self._target_speed = target_speed
        self._noise = noise

        self._agent = np.array([0.0, 0.0], dtype=np.float32)
        self._target = np.array([0.5, 0.5], dtype=np.float32)
        self._t = 0.0

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 2

    def observe(self) -> np.ndarray:
        dx = self._target[0] - self._agent[0]
        dy = self._target[1] - self._agent[1]
        return np.array([
            self._agent[0], self._agent[1],
            self._target[0], self._target[1],
            dx, dy,
        ], dtype=np.float32)

    def execute(self, action: np.ndarray) -> float:
        action = np.clip(action[:2], -1.0, 1.0)

        self._agent += action * self._speed
        self._agent += np.random.randn(2).astype(np.float32) * self._noise
        self._agent = np.clip(self._agent, -2.0, 2.0)

        self._t += 1
        self._target[0] = 0.5 * np.sin(self._t * self._target_speed)
        self._target[1] = 0.5 * np.cos(self._t * self._target_speed * 0.7)

        distance = float(np.linalg.norm(self._target - self._agent))
        reward = -distance

        return reward

    def reset(self):
        self._agent = np.array([0.0, 0.0], dtype=np.float32)
        self._target = np.array([0.5, 0.5], dtype=np.float32)
        self._t = 0.0


class BalanceBeam:
    """
    Balance a pole on a beam (simplified CartPole).

    The agent applies force to keep a pole upright.
    Reward = 1.0 if pole is within ±15° of vertical, else 0.0.

    State: [position, velocity, angle, angular_velocity]
    Action: [force]  in [-1, 1]
    """

    def __init__(self, dt: float = 0.02):
        self._dt = dt
        self._gravity = 9.8
        self._pole_length = 1.0
        self._mass = 0.1

        self._pos = 0.0
        self._vel = 0.0
        self._angle = 0.05
        self._ang_vel = 0.0

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 1

    def observe(self) -> np.ndarray:
        return np.array([
            self._pos, self._vel, self._angle, self._ang_vel,
        ], dtype=np.float32)

    def execute(self, action: np.ndarray) -> float:
        force = float(np.clip(action[0], -1.0, 1.0)) * 10.0

        cos_a = np.cos(self._angle)
        sin_a = np.sin(self._angle)

        ang_acc = (
            self._gravity * sin_a - force * cos_a * 0.1
        ) / self._pole_length
        acc = force - self._mass * self._pole_length * ang_acc * cos_a

        self._vel += acc * self._dt
        self._pos += self._vel * self._dt
        self._ang_vel += ang_acc * self._dt
        self._angle += self._ang_vel * self._dt

        self._pos = np.clip(self._pos, -2.4, 2.4)

        upright = abs(self._angle) < (15 * np.pi / 180)
        in_bounds = abs(self._pos) < 2.4
        reward = 1.0 if (upright and in_bounds) else 0.0

        return reward

    def reset(self):
        self._pos = 0.0
        self._vel = 0.0
        self._angle = 0.05 * (np.random.randn() * 0.5 + 1.0)
        self._ang_vel = 0.0
