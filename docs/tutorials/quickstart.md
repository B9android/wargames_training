# Quick-Start Guide

Welcome to the Wargames Training research platform.  This guide walks you
through installation, running your first training experiment, and submitting
results to WargamesBench.

## Installation

### From PyPI

```bash
pip install wargames-training
```

### From Source

```bash
git clone https://github.com/B9android/wargames_training.git
cd wargames_training
pip install -e ".[dev]"
```

### Optional extras

| Extra | What it adds |
|-------|-------------|
| `dev` | pytest, black, ruff, jupyter |
| `distributed` | ray[rllib] for multi-GPU league training |
| `export` | ONNX policy export |
| `docs` | MkDocs documentation builder |
| `all` | distributed + export |

## Your First Training Run

The simplest experiment — train a scripted baseline on `BattalionEnv`:

```python
import wandb
from envs import BattalionEnv
from training.train import train

wandb.init(project="wargames_training", config={"n_steps": 10_000})
env = BattalionEnv()
train(env=env, total_timesteps=10_000)
```

Or use the CLI:

```bash
python -m training.train --config configs/experiment_1.yaml
```

## Running WargamesBench

Evaluate any policy against the 20 canonical scenarios:

```bash
# Quick dry-run (CI-friendly)
wargames-bench --episodes 5 --scenarios 4

# Full canonical benchmark (100 episodes × 20 scenarios)
wargames-bench --episodes 100 --label my_policy_v1
```

Programmatic usage:

```python
from benchmarks import WargamesBench, BenchConfig

cfg = BenchConfig(n_eval_episodes=10, n_scenarios=4)
bench = WargamesBench(cfg)
summary = bench.run(policy=None)   # None → scripted baseline
print(summary)
summary.write_markdown()           # → docs/wargames_bench_leaderboard.md
```

## What's Next?

- **[Training Guide](training.md)** — PPO, self-play, and league training
- **[Benchmark Guide](benchmark.md)** — WargamesBench deep-dive
- **[API Reference](../api/index.md)** — full API documentation
- **[Contributing](../../CONTRIBUTING.md)** — how to contribute
