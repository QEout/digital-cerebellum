"""
TankBattleEnv — 2D tank battle for validating cortex-cerebellum collaboration.

The LLM (cortex) makes strategic decisions every few seconds:
  - Which enemy to target
  - Whether to be aggressive or defensive
  - Where to move

The cerebellum (GUIController) handles real-time micro-operations:
  - Precise turret aiming at the designated target
  - Dodging incoming bullets
  - Smooth movement toward strategic waypoints

Over time, the cerebellum learns these micro-operations from prediction
errors, and the LLM needs to intervene less frequently.

State vector (15D):
  [player_x, player_y, player_hp, turret_angle, body_angle,
   target_dx, target_dy, target_dist,
   nearest_bullet_dx, nearest_bullet_dy, nearest_bullet_dist,
   num_enemies_alive, ammo_frac, time_frac, strategy_code]

Action vector (4D):
  [move_x, move_y, turret_d_angle, shoot_trigger]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from digital_cerebellum.micro_ops.gui_controller import (
    GUIController,
    GUIControlConfig,
    CorrectionMicrozone,
)


@dataclass
class EnemyType:
    """Distinct enemy personality — makes LLM strategy meaningful."""
    name: str = "普通"
    name_en: str = "Normal"
    hp: float = 50.0
    speed: float = 0.8
    shoot_interval: int = 60
    bullet_speed: float = 5.0
    bullet_damage: float = 8.0
    accuracy: float = 0.15
    aggression: float = 0.5
    color: str = "#f44"

ENEMY_TYPES = {
    "heavy": EnemyType(
        name="重装", name_en="Heavy",
        hp=90.0, speed=0.4, shoot_interval=80,
        bullet_speed=4.0, bullet_damage=15.0,
        accuracy=0.25, aggression=0.3, color="#f80",
    ),
    "sniper": EnemyType(
        name="狙击", name_en="Sniper",
        hp=30.0, speed=0.6, shoot_interval=25,
        bullet_speed=8.0, bullet_damage=12.0,
        accuracy=0.06, aggression=0.6, color="#f0f",
    ),
    "guerrilla": EnemyType(
        name="游击", name_en="Guerrilla",
        hp=45.0, speed=1.8, shoot_interval=55,
        bullet_speed=5.0, bullet_damage=6.0,
        accuracy=0.20, aggression=0.9, color="#0f0",
    ),
}

DEFAULT_ENEMY_ROSTER = ["heavy", "sniper", "guerrilla"]

_MIN_SPAWN_DIST_FROM_PLAYER = 120.0
_MIN_SPAWN_DIST_BETWEEN_ENEMIES = 60.0


@dataclass
class TankConfig:
    arena_w: float = 600.0
    arena_h: float = 450.0
    player_speed: float = 2.5
    player_hp: float = 60.0
    bullet_speed: float = 8.0
    bullet_damage: float = 15.0
    shoot_cooldown: int = 15
    enemy_count: int = 3
    enemy_roster: list[str] = field(default_factory=lambda: list(DEFAULT_ENEMY_ROSTER))
    round_max_ticks: int = 1800
    turret_speed: float = 0.15
    noise: float = 0.1
    randomize_spawns: bool = True


@dataclass
class _Tank:
    x: float = 0.0
    y: float = 0.0
    hp: float = 100.0
    angle: float = 0.0
    turret_angle: float = 0.0
    alive: bool = True
    shoot_cd: int = 0
    label: str = ""
    etype: EnemyType | None = None


@dataclass
class _Bullet:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    damage: float = 15.0
    owner: str = "player"
    alive: bool = True


_GRADE_THRESHOLDS = [
    (600, "S"), (400, "A"), (250, "B"), (120, "C"),
]


def _grade(score: float) -> str:
    for thresh, g in _GRADE_THRESHOLDS:
        if score >= thresh:
            return g
    return "D"


class TankBattleEnv:
    """
    2D tank battle implementing the Environment protocol.

    Player tank (blue) vs enemy tanks (red).
    Controlled by cortex (strategy) + cerebellum (micro-ops).
    """

    def __init__(self, cfg: TankConfig | None = None):
        self.cfg = cfg or TankConfig()
        self._tick = 0
        self._kills = 0
        self._shots_fired = 0
        self._shots_hit = 0
        self._dodges = 0
        self._times_hit = 0
        self._llm_calls = 0
        self._done = False

        self._strategy_target: int = 0
        self._strategy_code: float = 0.0
        self._move_toward: np.ndarray = np.array([400.0, 300.0])

        self._player = _Tank()
        self._enemies: list[_Tank] = []
        self._bullets: list[_Bullet] = []
        self._explosions: list[dict] = []

        self._near_miss_cooldown = 0

        self._damage_from: dict[str, float] = {}
        self._hits_on: dict[str, int] = {}
        self._deaths_by: dict[str, int] = {}
        self._round_history: list[dict] = []

        self.reset()

    @property
    def state_dim(self) -> int:
        return 15

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self):
        c = self.cfg

        if self._tick > 0:
            self._round_history.append({
                "damage_from": dict(self._damage_from),
                "deaths_by": dict(self._deaths_by),
                "hits_on": dict(self._hits_on),
                "killed": [e.label for e in self._enemies if not e.alive],
                "survived": self._player.hp > 0,
            })

        self._tick = 0
        self._kills = 0
        self._shots_fired = 0
        self._shots_hit = 0
        self._dodges = 0
        self._times_hit = 0
        self._llm_calls = 0
        self._done = False
        self._bullets.clear()
        self._explosions.clear()
        self._near_miss_cooldown = 0
        self._damage_from = {}
        self._hits_on = {}
        self._deaths_by = {}

        px = c.arena_w * (0.3 + np.random.random() * 0.4) if c.randomize_spawns else c.arena_w / 2
        py = c.arena_h * (0.3 + np.random.random() * 0.4) if c.randomize_spawns else c.arena_h / 2
        self._player = _Tank(
            x=px, y=py,
            hp=c.player_hp, angle=0.0, turret_angle=0.0,
            alive=True, label="player",
        )

        self._enemies = []
        fixed_positions = [
            (c.arena_w * 0.15, c.arena_h * 0.15),
            (c.arena_w * 0.85, c.arena_h * 0.15),
            (c.arena_w * 0.5, c.arena_h * 0.85),
        ]
        placed: list[tuple[float, float]] = []
        for i in range(c.enemy_count):
            type_key = c.enemy_roster[i % len(c.enemy_roster)]
            et = ENEMY_TYPES.get(type_key, EnemyType())
            label = chr(65 + i)
            if c.randomize_spawns:
                ex, ey = self._random_enemy_spawn(px, py, placed, c)
            else:
                ex, ey = fixed_positions[i % len(fixed_positions)]
            placed.append((ex, ey))
            self._enemies.append(_Tank(
                x=ex, y=ey, hp=et.hp,
                angle=np.random.uniform(0, math.tau),
                turret_angle=np.random.uniform(0, math.tau),
                alive=True, label=label,
                shoot_cd=np.random.randint(0, et.shoot_interval),
                etype=et,
            ))

        self._strategy_target = 0
        self._strategy_code = 0.0
        self._move_toward = np.array([c.arena_w / 2, c.arena_h / 2])

    @property
    def done(self) -> bool:
        return self._done

    def observe(self) -> np.ndarray:
        c = self.cfg
        p = self._player
        sw, sh = c.arena_w, c.arena_h

        target = self._get_target_enemy()
        if target and target.alive:
            tdx = (target.x - p.x) / sw
            tdy = (target.y - p.y) / sh
            td = math.sqrt(tdx**2 + tdy**2)
        else:
            tdx, tdy, td = 0.0, 0.0, 0.0

        bdx, bdy, bd = 0.0, 0.0, 1.0
        min_d = float("inf")
        for b in self._bullets:
            if not b.alive or b.owner == "player":
                continue
            dx = (b.x - p.x) / sw
            dy = (b.y - p.y) / sh
            d = math.sqrt(dx**2 + dy**2)
            if d < min_d:
                min_d = d
                bdx, bdy, bd = dx, dy, d

        alive_count = sum(1 for e in self._enemies if e.alive)
        time_frac = min(1.0, self._tick / c.round_max_ticks)

        return np.array([
            p.x / sw, p.y / sh,
            p.hp / c.player_hp,
            p.turret_angle / math.tau,
            p.angle / math.tau,
            tdx, tdy, td,
            bdx, bdy, bd,
            alive_count / max(1, c.enemy_count),
            1.0 - p.shoot_cd / max(1, c.shoot_cooldown),
            time_frac,
            self._strategy_code,
        ], dtype=np.float32)

    def execute(self, action: np.ndarray) -> float:
        action = np.clip(action[:self.action_dim], -1.0, 1.0)
        c = self.cfg
        self._tick += 1
        reward = 0.0

        target = self._get_target_enemy()
        old_aim_error = math.pi
        old_target_dist = 1.0
        if target and target.alive:
            dx = target.x - self._player.x
            dy = target.y - self._player.y
            desired = math.atan2(dy, dx) % math.tau
            old_aim_error = abs(self._angle_diff(self._player.turret_angle, desired))
            old_target_dist = math.hypot(dx, dy)

        mx = float(action[0]) * c.player_speed
        my = float(action[1]) * c.player_speed
        self._player.x = np.clip(self._player.x + mx, 20, c.arena_w - 20)
        self._player.y = np.clip(self._player.y + my, 20, c.arena_h - 20)

        if abs(mx) > 0.1 or abs(my) > 0.1:
            self._player.angle = math.atan2(my, mx)

        d_turret = float(action[2]) * c.turret_speed
        self._player.turret_angle = (self._player.turret_angle + d_turret) % math.tau

        shoot = float(action[3]) > 0.3
        if shoot and self._player.shoot_cd <= 0:
            self._fire_bullet(self._player, "player", c.bullet_speed, c.bullet_damage)
            self._player.shoot_cd = c.shoot_cooldown
            self._shots_fired += 1

        if self._player.shoot_cd > 0:
            self._player.shoot_cd -= 1

        self._tick_enemies()
        hit_reward = self._tick_bullets()
        reward += hit_reward

        if target and target.alive:
            dx = target.x - self._player.x
            dy = target.y - self._player.y
            desired = math.atan2(dy, dx) % math.tau
            new_aim_error = abs(self._angle_diff(self._player.turret_angle, desired))
            new_target_dist = math.hypot(dx, dy)

            aim_improvement = old_aim_error - new_aim_error
            reward += aim_improvement * 0.8

            if new_aim_error < 0.2:
                reward += 0.15
            elif new_aim_error < 0.5:
                reward += 0.05

            max_dist = math.hypot(c.arena_w, c.arena_h)
            approach = (old_target_dist - new_target_dist) / max_dist
            reward += approach * 2.0

        alive = sum(1 for e in self._enemies if e.alive)
        if alive == 0 or self._player.hp <= 0 or self._tick >= c.round_max_ticks:
            self._done = True

        return reward

    def set_strategy(self, target_idx: int, strategy: str, move_toward: list[float]):
        target_idx = max(0, min(target_idx, len(self._enemies) - 1))
        if not self._enemies[target_idx].alive:
            for i, e in enumerate(self._enemies):
                if e.alive:
                    target_idx = i
                    break
        self._strategy_target = target_idx
        strat_map = {"aggressive": 0.8, "defensive": 0.2, "neutral": 0.5, "retreat": 0.0}
        self._strategy_code = strat_map.get(strategy, 0.5)
        self._move_toward = np.array(move_toward[:2], dtype=np.float32)
        self._llm_calls += 1

    def get_threat_assessment(self) -> dict[str, dict]:
        """
        Cerebellar threat assessment — the "gut feeling" about each enemy.

        Aggregates across current round and round history to identify
        which enemy is truly dangerous (not what LLM assumes).
        This is the cerebellum's predictive model feeding back to cortex.
        """
        threats: dict[str, dict] = {}
        for e in self._enemies:
            label = e.label
            et = e.etype or EnemyType()

            cur_dmg = self._damage_from.get(label, 0)
            cur_deaths = self._deaths_by.get(label, 0)

            hist_dmg = sum(r["damage_from"].get(label, 0) for r in self._round_history)
            hist_deaths = sum(r["deaths_by"].get(label, 0) for r in self._round_history)
            hist_rounds = len(self._round_history) or 1

            avg_dmg_per_round = (hist_dmg + cur_dmg) / (hist_rounds + 1)
            death_rate = (hist_deaths + cur_deaths) / (hist_rounds + 1)

            threat_score = avg_dmg_per_round * 2.0 + death_rate * 50.0

            if not e.alive:
                threat_level = "已消灭"
            elif threat_score > 30:
                threat_level = "极高危"
            elif threat_score > 15:
                threat_level = "高危"
            elif threat_score > 5:
                threat_level = "中等"
            else:
                threat_level = "低"

            threats[label] = {
                "type": et.name,
                "type_en": et.name_en,
                "alive": e.alive,
                "hp": round(e.hp, 1),
                "threat_score": round(threat_score, 1),
                "threat_level": threat_level,
                "avg_dmg_per_round": round(avg_dmg_per_round, 1),
                "death_rate": round(death_rate, 2),
                "color": et.color,
            }
        return threats

    def get_state_summary(self) -> str:
        """State summary for LLM, now includes cerebellar threat assessment."""
        p = self._player
        threats = self.get_threat_assessment()

        lines = [f"你在({p.x:.0f},{p.y:.0f})，血量{p.hp:.0f}。"]
        for i, e in enumerate(self._enemies):
            t = threats.get(e.label, {})
            if e.alive:
                lines.append(
                    f"敌人{e.label}(索引{i})[{t.get('type','?')}]"
                    f"在({e.x:.0f},{e.y:.0f})"
                    f"血量{e.hp:.0f}，小脑威胁评估:{t.get('threat_level','?')}"
                    f"(场均伤害{t.get('avg_dmg_per_round',0):.0f}，"
                    f"致死率{t.get('death_rate',0):.0%})。"
                )
            else:
                lines.append(f"敌人{e.label}(索引{i})已被击毁。")

        incoming = sum(1 for b in self._bullets if b.alive and b.owner != "player")
        if incoming > 0:
            lines.append(f"{incoming}发子弹朝你飞来。")

        max_threat = max(threats.values(), key=lambda t: t["threat_score"] if t["alive"] else -1, default=None)
        if max_threat and max_threat["alive"] and max_threat["threat_score"] > 15:
            lines.append(f"小脑直觉警告：{max_threat['type']}型敌人威胁极高，"
                         f"场均造成{max_threat['avg_dmg_per_round']:.0f}伤害，"
                         f"建议优先处理！")

        lines.append(f"已击杀{self._kills}辆，剩余时间{100 - self._tick * 100 // self.cfg.round_max_ticks}%。")
        return " ".join(lines)

    def get_round_score(self) -> dict:
        tick = max(1, self._tick)
        time_sec = tick / 60.0
        max_time = self.cfg.round_max_ticks / 60.0
        hit_rate = self._shots_hit / max(1, self._shots_fired)
        all_killed = self._kills == self.cfg.enemy_count

        if all_killed:
            speed_ratio = 1.0 - (time_sec / max_time)
            speed_score = max(0, speed_ratio) * 400
        else:
            speed_score = 0.0

        kill_score = self._kills * 80
        accuracy_score = hit_rate * 60
        dodge_score = self._dodges * 8
        hp_bonus = max(0, self._player.hp) * 0.5
        llm_penalty = self._llm_calls * 3

        total = speed_score + kill_score + accuracy_score + dodge_score + hp_bonus - llm_penalty
        total = max(0, total)

        return {
            "kills": self._kills,
            "shots_fired": self._shots_fired,
            "shots_hit": self._shots_hit,
            "hit_rate": round(hit_rate, 3),
            "survive_time": round(time_sec, 1),
            "time_ticks": tick,
            "dodges": self._dodges,
            "times_hit": self._times_hit,
            "llm_calls": self._llm_calls,
            "speed_score": round(speed_score, 1),
            "kill_score": round(kill_score, 1),
            "accuracy_score": round(accuracy_score, 1),
            "dodge_score": round(dodge_score, 1),
            "hp_bonus": round(hp_bonus, 1),
            "llm_penalty": round(llm_penalty, 1),
            "total_score": round(total, 1),
            "grade": _grade(total),
            "player_hp": round(self._player.hp, 1),
        }

    def get_game_state(self) -> dict:
        """Full state for frontend rendering."""
        p = self._player
        threats = self.get_threat_assessment()
        return {
            "tick": self._tick,
            "player": {"x": p.x, "y": p.y, "hp": p.hp, "angle": p.angle,
                        "turret": p.turret_angle, "alive": p.alive},
            "enemies": [
                {"x": e.x, "y": e.y, "hp": e.hp, "angle": e.angle,
                 "turret": e.turret_angle, "alive": e.alive, "label": e.label,
                 "type": (e.etype.name if e.etype else "?"),
                 "type_en": (e.etype.name_en if e.etype else "Normal"),
                 "color": (e.etype.color if e.etype else "#f44"),
                 "threat": threats.get(e.label, {}).get("threat_level", "?"),
                 "threat_score": threats.get(e.label, {}).get("threat_score", 0)}
                for e in self._enemies
            ],
            "bullets": [
                {"x": b.x, "y": b.y, "owner": b.owner, "alive": b.alive}
                for b in self._bullets if b.alive
            ],
            "explosions": list(self._explosions),
            "kills": self._kills,
            "shots_fired": self._shots_fired,
            "shots_hit": self._shots_hit,
            "dodges": self._dodges,
            "llm_calls": self._llm_calls,
            "done": self._done,
            "w": self.cfg.arena_w,
            "h": self.cfg.arena_h,
            "llm_target": self._strategy_target,
        }

    @property
    def stats(self) -> dict:
        return self.get_round_score()

    def _get_target_enemy(self) -> _Tank | None:
        idx = self._strategy_target
        if 0 <= idx < len(self._enemies) and self._enemies[idx].alive:
            return self._enemies[idx]
        for e in self._enemies:
            if e.alive:
                return e
        return None

    @staticmethod
    def _random_enemy_spawn(
        player_x: float, player_y: float,
        placed: list[tuple[float, float]],
        cfg: TankConfig,
    ) -> tuple[float, float]:
        """Pick a random spawn position far enough from player and others."""
        margin = 30.0
        for _ in range(200):
            ex = np.random.uniform(margin, cfg.arena_w - margin)
            ey = np.random.uniform(margin, cfg.arena_h - margin)
            if math.hypot(ex - player_x, ey - player_y) < _MIN_SPAWN_DIST_FROM_PLAYER:
                continue
            if any(
                math.hypot(ex - px, ey - py) < _MIN_SPAWN_DIST_BETWEEN_ENEMIES
                for px, py in placed
            ):
                continue
            return ex, ey
        return cfg.arena_w * 0.8, cfg.arena_h * 0.2

    def _fire_bullet(self, tank: _Tank, owner: str, speed: float, damage: float):
        a = tank.turret_angle
        self._bullets.append(_Bullet(
            x=tank.x + math.cos(a) * 18,
            y=tank.y + math.sin(a) * 18,
            vx=math.cos(a) * speed,
            vy=math.sin(a) * speed,
            damage=damage, owner=owner,
        ))

    def _tick_enemies(self):
        c = self.cfg
        p = self._player
        for e in self._enemies:
            if not e.alive:
                continue
            et = e.etype or EnemyType()
            dx = p.x - e.x
            dy = p.y - e.y
            dist = math.sqrt(dx**2 + dy**2) + 1e-6
            nx, ny = dx / dist, dy / dist

            if et.name_en == "Heavy":
                move_x = nx * et.speed * 0.3
                move_y = ny * et.speed * 0.3
            elif et.name_en == "Guerrilla":
                orbit_angle = math.atan2(dy, dx) + math.pi / 2
                preferred_dist = 120.0
                if dist < preferred_dist * 0.7:
                    radial = -0.6
                elif dist > preferred_dist * 1.3:
                    radial = 0.6
                else:
                    radial = 0.0
                tangent_x = math.cos(orbit_angle)
                tangent_y = math.sin(orbit_angle)
                move_x = (nx * radial + tangent_x * 0.8) * et.speed
                move_y = (ny * radial + tangent_y * 0.8) * et.speed
                move_x += np.random.randn() * et.speed * 0.4
                move_y += np.random.randn() * et.speed * 0.4
            elif et.name_en == "Sniper":
                preferred_dist = 180.0
                if dist < preferred_dist:
                    move_x = -nx * et.speed * 0.5
                    move_y = -ny * et.speed * 0.5
                else:
                    move_x = nx * et.speed * 0.15
                    move_y = ny * et.speed * 0.15
                move_x += np.random.randn() * 0.3
                move_y += np.random.randn() * 0.3
            else:
                move_x = nx * et.speed * et.aggression
                move_y = ny * et.speed * et.aggression

            e.x = np.clip(e.x + move_x, 20, c.arena_w - 20)
            e.y = np.clip(e.y + move_y, 20, c.arena_h - 20)

            e.turret_angle = math.atan2(dy, dx) + np.random.randn() * et.accuracy
            e.angle = math.atan2(ny, nx)

            e.shoot_cd -= 1
            if e.shoot_cd <= 0:
                self._fire_bullet(e, f"enemy_{e.label}", et.bullet_speed, et.bullet_damage)
                e.shoot_cd = et.shoot_interval + np.random.randint(-10, 10)

    def _tick_bullets(self) -> float:
        c = self.cfg
        reward = 0.0
        p = self._player
        new_explosions: list[dict] = []

        self._near_miss_cooldown = max(0, self._near_miss_cooldown - 1)

        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.vx
            b.y += b.vy

            if b.x < 0 or b.x > c.arena_w or b.y < 0 or b.y > c.arena_h:
                b.alive = False
                continue

            if b.owner == "player":
                for e in self._enemies:
                    if not e.alive:
                        continue
                    if math.hypot(b.x - e.x, b.y - e.y) < 18:
                        e.hp -= b.damage
                        b.alive = False
                        self._shots_hit += 1
                        self._hits_on[e.label] = self._hits_on.get(e.label, 0) + 1
                        reward += 2.0
                        new_explosions.append({"x": b.x, "y": b.y, "r": 12, "life": 1.0})
                        if e.hp <= 0:
                            e.alive = False
                            self._kills += 1
                            reward += 5.0
                            new_explosions.append({"x": e.x, "y": e.y, "r": 30, "life": 1.0})
                        break
            else:
                dist_to_player = math.hypot(b.x - p.x, b.y - p.y)
                if dist_to_player < 16:
                    p.hp -= b.damage
                    b.alive = False
                    self._times_hit += 1
                    src = b.owner.replace("enemy_", "")
                    self._damage_from[src] = self._damage_from.get(src, 0) + b.damage
                    if p.hp <= 0:
                        self._deaths_by[src] = self._deaths_by.get(src, 0) + 1
                    reward -= 1.0
                    new_explosions.append({"x": p.x, "y": p.y, "r": 10, "life": 1.0})
                elif dist_to_player < 40 and self._near_miss_cooldown <= 0:
                    self._dodges += 1
                    reward += 0.5
                    self._near_miss_cooldown = 10

        self._explosions = [
            {**ex, "life": ex["life"] - 0.05}
            for ex in [*self._explosions, *new_explosions]
            if ex["life"] > 0.05
        ]

        self._bullets = [b for b in self._bullets if b.alive]
        return reward

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = (b - a) % math.tau
        if d > math.pi:
            d -= math.tau
        return d


class TankController(GUIController):
    """
    Tank-specific cortex signals + multi-microzone cerebellar controller.

    Three parallel microzones mirror biological specialization:
      aim   — turret angle error  (climbing fiber: retinal slip)
      dodge — bullet proximity    (climbing fiber: nociceptive/threat)
      move  — target approach     (climbing fiber: proprioceptive drift)

    Each microzone has its own correction net and learns independently.
    Their outputs are summed at the "deep cerebellar nuclei" (action vector).

    Task-specific overrides:
      - _build_microzones()       — 3 parallel correction circuits
      - cortex_error_signals()    — per-microzone climbing fibers
      - cortex_signal()           — crude motor intention for tank combat
    """

    def _build_microzones(self) -> list[CorrectionMicrozone]:
        cfg = self.cfg
        return [
            CorrectionMicrozone(
                "aim", self.state_dim, self.action_dim,
                hidden_dim=cfg.correction_hidden,
                lr=cfg.correction_lr,
                scale=cfg.correction_scale * 0.8,
            ),
            CorrectionMicrozone(
                "dodge", self.state_dim, self.action_dim,
                hidden_dim=cfg.correction_hidden,
                lr=cfg.correction_lr * 2.0,
                scale=cfg.correction_scale * 1.2,
            ),
            CorrectionMicrozone(
                "move", self.state_dim, self.action_dim,
                hidden_dim=cfg.correction_hidden,
                lr=cfg.correction_lr,
                scale=cfg.correction_scale * 0.5,
            ),
        ]

    def cortex_error_signals(
        self, state_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "aim": self._aim_error(state_t),
            "dodge": self._dodge_error(state_t),
            "move": self._move_error(state_t),
        }

    @staticmethod
    def _aim_error(state_t: torch.Tensor) -> torch.Tensor:
        """Turret aiming error — the "retinal slip" for gaze control.

        sin(angle_error) via cross product is differentiable everywhere.
        """
        turret_frac = state_t[..., 3:4]
        turret_angle = turret_frac * (2 * math.pi)
        turret_cx = torch.cos(turret_angle)
        turret_cy = torch.sin(turret_angle)

        target_dx = state_t[..., 5:6]
        target_dy = state_t[..., 6:7]
        target_norm = torch.sqrt(target_dx ** 2 + target_dy ** 2 + 1e-8)
        target_nx = target_dx / target_norm
        target_ny = target_dy / target_norm

        aim_sin = turret_cx * target_ny - turret_cy * target_nx
        aim_cos_gap = 1.0 - (turret_cx * target_nx + turret_cy * target_ny)

        return torch.cat([aim_sin * 3.0, aim_cos_gap * 2.0], dim=-1)

    @staticmethod
    def _dodge_error(state_t: torch.Tensor) -> torch.Tensor:
        """Bullet proximity error — the nociceptive climbing fiber.

        High error when nearest bullet is close → correction learns to
        add evasive movement.  Error ~0 when no threat → no interference.
        """
        bullet_dx = state_t[..., 8:9]
        bullet_dy = state_t[..., 9:10]
        bullet_dist = state_t[..., 10:11]

        proximity = 1.0 / (bullet_dist + 0.15)

        return torch.cat([
            bullet_dx * proximity * 2.0,
            bullet_dy * proximity * 2.0,
        ], dim=-1)

    @staticmethod
    def _move_error(state_t: torch.Tensor) -> torch.Tensor:
        """Target approach error — proprioceptive drift from waypoint.

        Modulated by strategy_code: aggressive → want to close distance;
        defensive → smaller drive to approach.
        """
        target_dx = state_t[..., 5:6]
        target_dy = state_t[..., 6:7]
        strategy = state_t[..., 14:15].clamp(0.1, 1.0)

        return torch.cat([
            target_dx * strategy * 1.5,
            target_dy * strategy * 1.5,
        ], dim=-1)

    def cortex_signal(self, state: np.ndarray) -> np.ndarray:
        """
        Crude motor intention for tank combat — like a novice player.

        Knows the general idea (aim at enemy, move toward, try to shoot)
        but executes imprecisely. The cerebellum will refine via SPE.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)

        target_dx = state[5] if len(state) > 5 else 0.0
        target_dy = state[6] if len(state) > 6 else 0.0
        target_dist = state[7] if len(state) > 7 else 1.0
        turret_angle = state[3] * math.tau if len(state) > 3 else 0.0
        strategy = state[14] if len(state) > 14 else 0.5

        action[0] = np.clip(target_dx * self.cfg.cortex_gain * 1.2 * strategy, -0.9, 0.9)
        action[1] = np.clip(target_dy * self.cfg.cortex_gain * 1.2 * strategy, -0.9, 0.9)

        bdx = state[8] if len(state) > 8 else 0
        bdy = state[9] if len(state) > 9 else 0
        bdist = state[10] if len(state) > 10 else 1.0
        if bdist < 0.15:
            action[0] -= bdx * 0.4
            action[1] -= bdy * 0.4

        noisy_dx = target_dx + np.random.randn() * 0.08
        noisy_dy = target_dy + np.random.randn() * 0.08
        desired_angle = math.atan2(noisy_dy, noisy_dx + 1e-8)
        angle_err = desired_angle - turret_angle
        while angle_err > math.pi:
            angle_err -= math.tau
        while angle_err < -math.pi:
            angle_err += math.tau
        action[2] = np.clip(angle_err * self.cfg.cortex_gain * 0.8, -0.9, 0.9)

        ammo_ready = state[12] if len(state) > 12 else 0
        if target_dist < 0.8 and abs(angle_err) < 0.5 and ammo_ready > 0.5:
            action[3] = 0.8
        elif target_dist < 0.5 and abs(angle_err) < 1.0:
            action[3] = 0.3
        else:
            action[3] = -0.5

        action += np.random.randn(self.action_dim).astype(np.float32) * self._noise_scale

        return np.clip(action, -1.0, 1.0)
