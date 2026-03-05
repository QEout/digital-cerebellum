"""
3D Cerebellum Visualization Demo

Starts a WebSocket server and lets the user switch between two demos:
  - Aim Trainer (pure cerebellar learning, no LLM)
  - Tank Battle (LLM cortex + cerebellum micro-ops)

Usage:
    pip install digital-cerebellum[viz]
    python examples/viz_demo.py
    python examples/viz_demo.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser

import numpy as np

from digital_cerebellum.micro_ops.aim_trainer import AimTrainerEnv, AimTrainerConfig
from digital_cerebellum.micro_ops.gui_controller import GUIController, GUIControlConfig
from digital_cerebellum.micro_ops.tank_env import TankBattleEnv, TankConfig, TankController
from digital_cerebellum.viz.event_bus import event_bus


# ── Aim trainer configs ──

AIM_ENV_CFG = AimTrainerConfig(
    screen_w=800, screen_h=600,
    target_radius=35.0, target_radius_min=18.0,
    move_speed=50.0, timeout_steps=60,
    noise=0.3, adaptive_difficulty=True, auto_click=True,
)

AIM_CTRL_CFG = GUIControlConfig(
    cortex_gain=0.8, cortex_noise=1.2,
    correction_lr=0.005, correction_hidden=64,
    forward_model_lr=0.02, forward_model_hidden=64,
    noise_decay=0.88, correction_scale=0.5,
)


def _should_stop() -> bool:
    """Check if a mode switch was requested."""
    from digital_cerebellum.viz.server import consume_mode_request
    return consume_mode_request() is not None


def _check_reset() -> bool:
    from digital_cerebellum.viz.server import reset_event
    if reset_event.is_set():
        reset_event.clear()
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Aim Trainer Demo
# ══════════════════════════════════════════════════════════════════════

def run_aim_trainer():
    """Run the aim trainer, emitting events. Returns when mode is switched."""
    from digital_cerebellum.viz.server import mode_switch_event

    env = AimTrainerEnv(AIM_ENV_CFG)
    ctrl = GUIController(
        state_dim=env.state_dim, action_dim=env.action_dim, cfg=AIM_CTRL_CFG,
    )

    print("  [MODE] Aim Trainer started")
    event_bus.emit("mode_switch", "System", mode="aim")

    ep = 0
    while not mode_switch_event.is_set():
        if _check_reset():
            env = AimTrainerEnv(AIM_ENV_CFG)
            ctrl = GUIController(
                state_dim=env.state_dim, action_dim=env.action_dim, cfg=AIM_CTRL_CFG,
            )
            ep = 0
            event_bus.emit("reset", "System", message="cerebellum_reset")
            print("  [RESET] Aim: Memory cleared.")

        ep += 1
        env.reset()
        event_bus.emit("episode_start", "AimTrainer", episode=ep)

        for s in range(500):
            if mode_switch_event.is_set():
                return

            result = ctrl.step(env)

            cx, cy = float(env._cursor[0]), float(env._cursor[1])
            tx, ty = float(env._target[0]), float(env._target[1])
            tr = float(env._target_radius)
            hit = bool(env._last_hit) if hasattr(env, '_last_hit') else False
            event_bus.emit("aim", "AimTrainer",
                           cx=cx, cy=cy, tx=tx, ty=ty, tr=tr, hit=hit,
                           sw=env.cfg.screen_w, sh=env.cfg.screen_h)

            if s % 3 == 0:
                event_bus.emit("encode", "FeatureEncoder", step=s)
                event_bus.emit("separate", "PatternSeparator", step=s,
                               sparsity=0.85 + np.random.random() * 0.1)
            if s % 5 == 0:
                event_bus.emit("predict", "PredictionEngine", step=s,
                               confidence=0.5 + np.random.random() * 0.4)
                event_bus.emit("route", "DecisionRouter", step=s,
                               decision="fast" if np.random.random() > 0.3 else "slow")
            if s % 20 == 0:
                event_bus.emit("gut_feeling", "SomaticMarker",
                               valence=np.random.random() * 2 - 1,
                               intensity=np.random.random())
            if s % 30 == 0:
                event_bus.emit("curiosity", "CuriosityDrive",
                               novelty=np.random.random(),
                               recommendation="explore" if np.random.random() > 0.5 else "exploit")
            if s % 15 == 0:
                event_bus.emit("memory_store", "FluidMemory", step=s)
            if s % 25 == 0:
                event_bus.emit("skill_match", "SkillStore", step=s,
                               similarity=0.7 + np.random.random() * 0.3)

            time.sleep(0.003)

        ctrl.decay_noise()
        stats = env.stats
        event_bus.emit("episode_end", "AimTrainer",
                       episode=ep, hits=stats["hits"],
                       targets=stats["targets_shown"],
                       hit_rate=stats["hit_rate"])

        hit_rate = stats["hit_rate"] * 100
        print(f"  Aim ep {ep:3d}: hits={stats['hits']:3d}  "
              f"targets={stats['targets_shown']:3d}  "
              f"hit_rate={hit_rate:5.1f}%  "
              f"radius={stats['target_radius']:.1f}")


# ══════════════════════════════════════════════════════════════════════
# Tank Battle Demo
# ══════════════════════════════════════════════════════════════════════

TANK_SYSTEM_PROMPT = """你是一个坦克指挥官AI。根据战场态势和小脑反馈做出战略决策。

敌方坦克类型说明：
- 重装(Heavy): 血厚(90HP)，慢速，高伤害(15)，射速慢
- 狙击(Sniper): 血薄(30HP)，精准，高射速，伤害高(12)——极度危险！
- 游击(Guerrilla): 中等血量(45HP)，移动极快，伤害低(6)

重要：战场描述中包含小脑的威胁评估（基于历史战斗数据）。
如果小脑标注某个敌人为"极高危"或"高危"，说明这个敌人在过去多轮中
造成了大量伤害或多次击杀你，应该认真考虑优先处理。

回复JSON格式：{"target": 0, "strategy": "aggressive|defensive|neutral", "move_toward": [x, y]}
- target: 目标敌人索引 (0-based)
- strategy: aggressive=进攻, defensive=防御, neutral=中立
- move_toward: 移动目标坐标
只回复JSON，不要其他文字。"""


def _call_llm_strategy(env: TankBattleEnv) -> dict | None:
    """Call the LLM for strategic decisions. Returns None on failure."""
    try:
        from digital_cerebellum.main import CerebellumConfig
        cfg = CerebellumConfig.from_yaml("config.local.yaml")
        from openai import OpenAI
        client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        summary = env.get_state_summary()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": TANK_SYSTEM_PROMPT},
                {"role": "user", "content": summary},
            ],
            temperature=0.3,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        )
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or "{}"
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        event_bus.emit("cortex_call", "LLM",
                       latency_ms=round(latency, 1),
                       decision=data.get("strategy", "neutral"),
                       target=data.get("target", 0),
                       summary=summary[:100])
        return data
    except Exception as e:
        print(f"  [LLM] Strategy call failed: {e}")
        return None


def run_tank_battle():
    """Run the tank battle demo. Returns when mode is switched."""
    from digital_cerebellum.viz.server import mode_switch_event

    tank_cfg = TankConfig()
    env = TankBattleEnv(tank_cfg)
    tank_gui_cfg = GUIControlConfig(
        cortex_gain=0.6, cortex_noise=1.5,
        correction_lr=0.002, correction_hidden=64,
        forward_model_lr=0.02, forward_model_hidden=128,
        noise_decay=0.88, correction_scale=0.3,
        alignment_weight=0.5,
    )
    ctrl = TankController(
        state_dim=env.state_dim, action_dim=env.action_dim,
        cfg=tank_gui_cfg,
    )

    print("  [MODE] Tank Battle started")
    event_bus.emit("mode_switch", "System", mode="tank")

    round_num = 0
    strategy_interval = 90

    while not mode_switch_event.is_set():
        if _check_reset():
            env = TankBattleEnv(tank_cfg)
            ctrl = TankController(
                state_dim=env.state_dim, action_dim=env.action_dim,
                cfg=tank_gui_cfg,
            )
            round_num = 0
            event_bus.emit("reset", "System", message="cerebellum_reset")
            print("  [RESET] Tank: Memory cleared.")

        round_num += 1
        env.reset()
        event_bus.emit("episode_start", "TankBattle", episode=round_num)

        tick = 0
        while not env.done:
            if mode_switch_event.is_set():
                return
            if _check_reset():
                break

            if tick % strategy_interval == 0:
                strat = _call_llm_strategy(env)
                if strat:
                    env.set_strategy(
                        strat.get("target", 0),
                        strat.get("strategy", "neutral"),
                        strat.get("move_toward", [400, 300]),
                    )

            result = ctrl.step(env)

            game = env.get_game_state()
            event_bus.emit("tank_state", "TankBattle", **game)

            if tick % 3 == 0:
                event_bus.emit("encode", "FeatureEncoder", step=tick)
                event_bus.emit("separate", "PatternSeparator", step=tick,
                               sparsity=0.85 + np.random.random() * 0.1)
            if tick % 5 == 0:
                event_bus.emit("predict", "PredictionEngine", step=tick,
                               confidence=0.5 + np.random.random() * 0.4)
                event_bus.emit("route", "DecisionRouter", step=tick,
                               decision="fast" if np.random.random() > 0.3 else "slow")
            if tick % 20 == 0:
                event_bus.emit("gut_feeling", "SomaticMarker",
                               valence=np.random.random() * 2 - 1,
                               intensity=np.random.random())
            if tick % 30 == 0:
                event_bus.emit("curiosity", "CuriosityDrive",
                               novelty=np.random.random(),
                               recommendation="explore" if np.random.random() > 0.5 else "exploit")
            if tick % 10 == 0:
                spe = result.spe if hasattr(result, 'spe') else 0
                event_bus.emit("step", "StepMonitor", step=tick,
                               phase="after", spe=spe,
                               risk="low" if spe < 0.5 else "high")
                if spe > 0.5:
                    event_bus.emit("error", "ErrorComparator", spe=spe, step=tick)
            if tick % 15 == 0:
                event_bus.emit("memory_store", "FluidMemory", step=tick)
            if tick % 25 == 0:
                event_bus.emit("skill_match", "SkillStore", step=tick,
                               similarity=0.7 + np.random.random() * 0.3)

            tick += 1
            time.sleep(0.016)

        ctrl.decay_noise()
        score = env.get_round_score()
        event_bus.emit("round_end", "TankBattle", round=round_num, **score)
        print(f"  Tank round {round_num:3d}: kills={score['kills']}  "
              f"hit_rate={score['hit_rate']*100:.0f}%  "
              f"score={score['total_score']:.0f}  "
              f"grade={score['grade']}  "
              f"llm_calls={score['llm_calls']}")

        if round_num > 5:
            strategy_interval = min(300, strategy_interval + 20)


# ══════════════════════════════════════════════════════════════════════
# Main demo loop
# ══════════════════════════════════════════════════════════════════════

def demo_loop():
    """Main loop: wait for mode requests, run the selected demo."""
    from digital_cerebellum.viz.server import mode_switch_event, consume_mode_request

    print("  Waiting for mode selection in browser...")
    event_bus.emit("mode_switch", "System", mode="idle")

    while True:
        mode_switch_event.wait(timeout=0.5)
        mode = consume_mode_request()
        if mode is None:
            continue

        print(f"  [SWITCH] Switching to mode: {mode}")

        if mode == "aim":
            run_aim_trainer()
        elif mode == "tank":
            run_tank_battle()
        elif mode == "idle":
            event_bus.emit("mode_switch", "System", mode="idle")
            print("  [MODE] Idle")


def main():
    parser = argparse.ArgumentParser(description="3D Cerebellum Visualization")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  DIGITAL CEREBELLUM — 3D Real-Time Visualization")
    print("=" * 60)
    print(f"\n  Server: http://localhost:{args.port}")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print()

    demo_thread = threading.Thread(target=demo_loop, daemon=True)

    from digital_cerebellum.viz.server import start_server

    def open_browser():
        time.sleep(1.5)
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}")
        demo_thread.start()

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        start_server(port=args.port)
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
