# WebArena Benchmark for Digital Cerebellum

Standard benchmark evaluation on [WebArena-Verified](https://github.com/ServiceNow/webarena-verified) (ICLR 2024).

## Quick Start

```bash
# 1. Mock mode — test infrastructure without WebArena servers
python -m benchmarks.webarena.runner --mock --quick

# 2. Full mock with ablation
python -m benchmarks.webarena.runner --mock --hard --ablation

# 3. Analyze results
python -m benchmarks.webarena.analysis --results-dir benchmarks/results
```

## Running Against Real WebArena

WebArena requires 4 web services (Shopping, Reddit, GitLab, CMS). Each Docker image is ~40GB, so a cloud VM is recommended.

### Option A: AWS EC2 (Recommended, ~$1-2 total)

```bash
pip install boto3
aws configure

# Launch + run
python -m benchmarks.webarena.deploy_aws --launch --key-name YOUR_KEY --mode quick

# Or connect to existing instance
python -m benchmarks.webarena.deploy_aws \
    --hostname ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com \
    --run --mode ablation

# Clean up
python -m benchmarks.webarena.deploy_aws --terminate --instance-id i-xxxxx
```

### Option B: Manual Server

Point the runner at any WebArena-compatible server:

```bash
python -m benchmarks.webarena.runner --hard --ablation \
    --shopping-url http://YOUR_SERVER:7770 \
    --reddit-url http://YOUR_SERVER:9999 \
    --gitlab-url http://YOUR_SERVER:8023
```

## Architecture

```
benchmarks/webarena/
├── agent.py              # Base LLM browsing agent (Playwright + qwen3.5-flash)
├── cerebellum_agent.py   # Cerebellum-wrapped agent (StepMonitor + SkillStore)
├── runner.py             # Benchmark runner with ablation support
├── analysis.py           # Results analysis (text/LaTeX/CSV/JSON export)
├── deploy_aws.py         # AWS EC2 deployment helper
├── setup_docker.py       # Local Docker setup (if you have 50GB+ free)
└── test_agent.py         # Component integration tests
```

## Ablation Configs

| Config | StepMonitor | SkillStore | FailureMemory |
|--------|------------|------------|---------------|
| baseline | - | - | - |
| full | Y | Y | Y |
| no_skill | Y | - | Y |
| no_monitor | - | Y | - |
| no_memory | Y | Y | - |

## Metrics

- **Pass rate**: % of tasks solved correctly (WebArena-Verified deterministic scoring)
- **Avg steps**: Mean browser actions per task
- **LLM calls**: Total LLM API invocations (fewer = more skill cache hits)
- **Skill hits**: Actions served from SkillStore cache
- **Cascades caught**: Error cascades detected and stopped early
- **Latency overhead**: Additional ms per step from cerebellum sidecar
