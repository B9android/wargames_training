# Wargames Training

**Reinforcement learning research platform for Napoleonic-era battalion simulation.**

[![PyPI](https://img.shields.io/pypi/v/wargames-training)](https://pypi.org/project/wargames-training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Wargames Training?

Wargames Training is an open-source RL research platform built around a
continuous 2D simulation of Napoleonic-era military battalions.  The platform
provides:

- **Gymnasium environments** — from single battalions to corps-level operations
- **PPO + self-play + league training** — complete training pipelines
- **WargamesBench** — 20 standardised scenarios for reproducible comparison
- **GIS terrain** — real battlefield data (Waterloo, Austerlitz, Borodino, Salamanca)

## Quick Install

```bash
pip install wargames-training
```

## First Steps

```python
from envs import BattalionEnv

env = BattalionEnv()
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## WargamesBench

Run the canonical benchmark against your policy:

```bash
wargames-bench --episodes 100 --label "my_policy_v1"
```

Results are reproducible within ± 2 % win rate across seeds.

## Community

- **GitHub Discussions** — research questions, benchmark results, ideas
- **Issues** — bug reports and feature requests
- **Contributing** — see [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Code of Conduct** — see [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)

## Citation

If you use Wargames Training or WargamesBench in your research, please cite:

```bibtex
@software{wargames_training,
  title   = {Wargames Training: An Open RL Research Platform for Military Simulation},
  author  = {Wargames Training Contributors},
  year    = {2026},
  url     = {https://github.com/B9android/wargames_training},
}
```
