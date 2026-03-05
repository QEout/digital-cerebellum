#!/usr/bin/env python3
"""
WebArena Benchmark Runner.

Runs baseline and cerebellum-enhanced agents on WebArena-Verified tasks,
collects results, and supports ablation experiments.

Usage:
    # Quick test (5 tasks)
    python -m benchmarks.webarena.runner --quick

    # Full hard subset (258 tasks)
    python -m benchmarks.webarena.runner --hard

    # Full ablation (5 configs x 258 tasks)
    python -m benchmarks.webarena.runner --hard --ablation

    # Specific tasks
    python -m benchmarks.webarena.runner --tasks 0,5,10

    # With live WebArena (provide base URLs)
    python -m benchmarks.webarena.runner --quick \\
        --shopping-url http://localhost:7770 \\
        --reddit-url http://localhost:9999 \\
        --gitlab-url http://localhost:8023 \\
        --cms-url http://localhost:7780

    # Mock mode (no browser, test infrastructure)
    python -m benchmarks.webarena.runner --mock --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# WebArena site URL placeholders → actual URLs
DEFAULT_URLS = {
    "__SHOPPING__": "http://localhost:7770",
    "__SHOPPING_ADMIN__": "http://localhost:7770/admin",
    "__REDDIT__": "http://localhost:9999",
    "__GITLAB__": "http://localhost:8023",
    "__WIKIPEDIA__": "http://localhost:8888",
    "__MAP__": "http://localhost:443",
    "__HOMEPAGE__": "http://localhost:4399",
}

# The 258-task "hard" subset from WebArena-Verified paper
HARD_TASK_IDS: list[int] | None = None  # loaded lazily


def _load_hard_task_ids() -> list[int]:
    """Load the hard subset task IDs from webarena-verified."""
    global HARD_TASK_IDS
    if HARD_TASK_IDS is not None:
        return HARD_TASK_IDS
    try:
        from webarena_verified import WebArenaVerified
        wav = WebArenaVerified()
        all_tasks = wav.get_tasks()
        HARD_TASK_IDS = [t.task_id for t in all_tasks]
        return HARD_TASK_IDS
    except Exception:
        HARD_TASK_IDS = list(range(258))
        return HARD_TASK_IDS


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class TaskResult:
    task_id: int
    intent: str
    agent_response: str
    score: float
    steps: int
    llm_calls: int
    time_s: float
    error: str | None = None
    skill_hits: int = 0
    cascades_caught: int = 0
    steps_blocked: int = 0


@dataclass
class RunConfig:
    mode: str  # "baseline" | "full" | "no_skill" | "no_monitor" | "no_memory"
    task_ids: list[int] = field(default_factory=list)
    mock: bool = False
    site_urls: dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkRunResult:
    config_label: str
    task_results: list[TaskResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.score > 0) / len(self.task_results)

    @property
    def avg_steps(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.steps for r in self.task_results) / len(self.task_results)

    @property
    def avg_llm_calls(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.llm_calls for r in self.task_results) / len(self.task_results)

    @property
    def total_skill_hits(self) -> int:
        return sum(r.skill_hits for r in self.task_results)

    @property
    def total_cascades_caught(self) -> int:
        return sum(r.cascades_caught for r in self.task_results)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_label,
            "tasks": len(self.task_results),
            "pass_rate": round(self.pass_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_llm_calls": round(self.avg_llm_calls, 2),
            "total_skill_hits": self.total_skill_hits,
            "total_cascades_caught": self.total_cascades_caught,
            "total_time_s": round(self.total_time_s, 1),
        }


# ======================================================================
# Mock agent — for testing infrastructure without a browser
# ======================================================================

class MockAgent:
    """Simulates agent runs without a browser or LLM for infra testing."""

    def __init__(self, mode: str = "baseline"):
        self.mode = mode
        self._cerebellum_stats = None

    def run_task(self, intent: str, start_url: str, task_id: int = -1, page=None):
        from benchmarks.webarena.agent import AgentResult, StepRecord
        import random

        time.sleep(0.01)  # simulate minimal latency
        result = AgentResult(task_id=task_id, intent=intent, response="")

        n_steps = random.randint(1, 5)
        for i in range(n_steps):
            result.steps.append(StepRecord(
                step=i,
                action=f"mock_action_{i}",
                observation="mock page content",
                llm_response='{"action": "click", "args": {"text": "button"}}',
                elapsed_ms=random.uniform(50, 200),
            ))
            result.total_llm_calls += 1

        result.response = f"mock answer for task {task_id}"
        result.total_time_s = n_steps * 0.1

        if self.mode != "baseline":
            from benchmarks.webarena.cerebellum_agent import CerebellumStats
            self._cerebellum_stats = CerebellumStats(
                skill_hits=random.randint(0, 2) if "no_skill" not in self.mode else 0,
                cascades_caught=random.randint(0, 1) if "no_monitor" not in self.mode else 0,
            )

        return result

    @property
    def cerebellum_stats(self):
        return self._cerebellum_stats


# ======================================================================
# Runner
# ======================================================================

def resolve_start_url(url_template: str, site_urls: dict[str, str]) -> str:
    """Replace WebArena URL placeholders with actual URLs."""
    urls = {**DEFAULT_URLS, **site_urls}
    for placeholder, real_url in urls.items():
        url_template = url_template.replace(placeholder, real_url)
    return url_template


def evaluate_task(task_id: int, agent_response: str) -> float:
    """Evaluate agent response using WebArena-Verified scorer."""
    try:
        from webarena_verified import WebArenaVerified
        from webarena_verified.types.agent_response import FinalAgentResponse

        wav = WebArenaVerified()
        task = wav.get_task(task_id)

        final_response = FinalAgentResponse(
            task_type=task.eval[0].expected.task_type,
            status="SUCCESS" if agent_response else "FAILURE",
            retrieved_data=[agent_response] if agent_response else [],
            error_details=None if agent_response else "No response",
        )

        result = wav.evaluate_task(
            task_id=task_id,
            agent_response=final_response,
            network_trace=[],
        )
        return result.score if hasattr(result, "score") else (1.0 if result else 0.0)
    except Exception as e:
        log.warning("Evaluation error for task %d: %s", task_id, e)
        return 0.0


def run_benchmark(config: RunConfig) -> BenchmarkRunResult:
    """Run a single benchmark configuration."""
    from webarena_verified import WebArenaVerified
    wav = WebArenaVerified()
    all_tasks = {t.task_id: t for t in wav.get_tasks()}

    if config.mock:
        agent = MockAgent(mode=config.mode)
    elif config.mode == "baseline":
        from benchmarks.webarena.agent import WebAgent
        llm_cfg = _load_llm_config()
        agent = WebAgent(**llm_cfg)
    else:
        from benchmarks.webarena.cerebellum_agent import CerebellumWebAgent, AblationConfig
        llm_cfg = _load_llm_config()
        ablation_map = {
            "full": AblationConfig.full,
            "no_skill": AblationConfig.no_skill,
            "no_monitor": AblationConfig.no_monitor,
            "no_memory": AblationConfig.no_memory,
        }
        ablation = ablation_map.get(config.mode, AblationConfig.full)()
        agent = CerebellumWebAgent(ablation=ablation, **llm_cfg)

    run_result = BenchmarkRunResult(config_label=config.mode)
    t0 = time.perf_counter()

    for i, tid in enumerate(config.task_ids):
        task = all_tasks.get(tid)
        if task is None:
            log.warning("Task %d not found, skipping", tid)
            continue

        start_url = resolve_start_url(task.start_urls[0], config.site_urls)
        intent = task.intent

        log.info(
            "[%s] Task %d/%d (id=%d): %s",
            config.mode, i + 1, len(config.task_ids), tid, intent[:60],
        )

        agent_result = agent.run_task(
            intent=intent,
            start_url=start_url,
            task_id=tid,
        )

        if config.mock:
            score = 1.0 if "mock" in agent_result.response else 0.0
        else:
            score = evaluate_task(tid, agent_result.response)

        cb_stats_obj = getattr(agent, "cerebellum_stats", None) if config.mode != "baseline" else None

        task_result = TaskResult(
            task_id=tid,
            intent=intent,
            agent_response=agent_result.response,
            score=score,
            steps=len(agent_result.steps),
            llm_calls=agent_result.total_llm_calls,
            time_s=agent_result.total_time_s,
            error=agent_result.error,
            skill_hits=cb_stats_obj.skill_hits if cb_stats_obj else 0,
            cascades_caught=cb_stats_obj.cascades_caught if cb_stats_obj else 0,
            steps_blocked=cb_stats_obj.steps_blocked if cb_stats_obj else 0,
        )
        run_result.task_results.append(task_result)

        log.info(
            "  -> score=%.2f, steps=%d, llm=%d, time=%.1fs",
            score, task_result.steps, task_result.llm_calls, task_result.time_s,
        )

    run_result.total_time_s = time.perf_counter() - t0
    return run_result


def _load_llm_config() -> dict[str, str]:
    """Load LLM config from config.local.yaml."""
    cfg_path = Path(__file__).parent.parent.parent / "config.local.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        llm = cfg.get("llm", {})
        return {
            "llm_model": llm.get("model", "qwen3.5-flash"),
            "llm_api_key": llm.get("api_key", ""),
            "llm_base_url": llm.get("base_url", ""),
        }
    return {}


# ======================================================================
# CLI
# ======================================================================

def save_results(results: list[BenchmarkRunResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = output_dir / f"webarena_{r.config_label}.json"
        data = {
            "summary": r.summary_dict(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "intent": t.intent,
                    "response": t.agent_response,
                    "score": t.score,
                    "steps": t.steps,
                    "llm_calls": t.llm_calls,
                    "time_s": round(t.time_s, 2),
                    "error": t.error,
                    "skill_hits": t.skill_hits,
                    "cascades_caught": t.cascades_caught,
                }
                for t in r.task_results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Saved %s", path)


def print_summary(results: list[BenchmarkRunResult]) -> None:
    print("\n" + "=" * 70)
    print("WebArena-Verified Benchmark Results")
    print("=" * 70)

    header = f"{'Config':<15} {'Tasks':>6} {'Pass%':>7} {'AvgSteps':>9} {'AvgLLM':>7} {'SkillHit':>9} {'Cascade':>8} {'Time':>8}"
    print(header)
    print("-" * 70)

    for r in results:
        s = r.summary_dict()
        print(
            f"{s['config']:<15} {s['tasks']:>6} {s['pass_rate']*100:>6.1f}% "
            f"{s['avg_steps']:>9.1f} {s['avg_llm_calls']:>7.1f} "
            f"{s['total_skill_hits']:>9} {s['total_cascades_caught']:>8} "
            f"{s['total_time_s']:>7.1f}s"
        )

    if len(results) >= 2:
        baseline = next((r for r in results if r.config_label == "baseline"), results[0])
        best = max(results, key=lambda r: r.pass_rate)
        if best.config_label != baseline.config_label:
            delta = best.pass_rate - baseline.pass_rate
            print(f"\nBest improvement: {best.config_label} ({delta*100:+.1f}% over baseline)")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebArena Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Run 5 tasks only")
    parser.add_argument("--hard", action="store_true", help="Run 258 hard subset tasks")
    parser.add_argument("--tasks", type=str, help="Comma-separated task IDs")
    parser.add_argument("--ablation", action="store_true", help="Run all ablation configs")
    parser.add_argument("--mock", action="store_true", help="Mock mode (no browser)")
    parser.add_argument("--output", type=str, default="benchmarks/results",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true")

    for site in ["shopping", "reddit", "gitlab", "cms", "wikipedia", "map"]:
        parser.add_argument(f"--{site}-url", type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine task IDs
    if args.tasks:
        task_ids = [int(x.strip()) for x in args.tasks.split(",")]
    elif args.quick:
        all_ids = _load_hard_task_ids()
        task_ids = all_ids[:5]
    elif args.hard:
        task_ids = _load_hard_task_ids()
    else:
        task_ids = _load_hard_task_ids()[:20]

    # Site URLs
    site_urls: dict[str, str] = {}
    for site, placeholder in [
        ("shopping", "__SHOPPING__"),
        ("reddit", "__REDDIT__"),
        ("gitlab", "__GITLAB__"),
        ("cms", "__HOMEPAGE__"),
        ("wikipedia", "__WIKIPEDIA__"),
        ("map", "__MAP__"),
    ]:
        url = getattr(args, f"{site}_url", None)
        if url:
            site_urls[placeholder] = url
            if site == "shopping":
                site_urls["__SHOPPING_ADMIN__"] = url + "/admin"

    # Determine which configs to run
    modes = ["baseline", "full"]
    if args.ablation:
        modes += ["no_skill", "no_monitor", "no_memory"]

    results: list[BenchmarkRunResult] = []
    for mode in modes:
        config = RunConfig(
            mode=mode,
            task_ids=task_ids,
            mock=args.mock,
            site_urls=site_urls,
        )
        log.info("Starting run: %s (%d tasks)", mode, len(task_ids))
        result = run_benchmark(config)
        results.append(result)

    print_summary(results)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
