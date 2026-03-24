# Wargames Training

**Reinforcement learning research platform for Napoleonic-era battalion simulation.**

[![PyPI](https://img.shields.io/pypi/v/wargames-training)](https://pypi.org/project/wargames-training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/B9android/wargames_training/actions/workflows/ray_smoke_test.yml/badge.svg)](https://github.com/B9android/wargames_training/actions)

---

## Navigate

- [Quickstart](tutorials/quickstart.md)
- [Training Guide](tutorials/training.md)
- [WargamesBench Guide](tutorials/benchmark.md)
- [API Reference](api/index.md)
- [Leaderboard](wargames_bench_leaderboard.md)
- [Roadmap](ROADMAP.md)

---

<div class="grid cards" markdown>

-   :material-map-legend: __Gymnasium Environments__

    ---

    Fully-featured Gymnasium and PettingZoo environments spanning single battalions through corps-level operations.  Observations normalized, angles as (cos θ, sin θ), positions normalized by map size.

    [:octicons-arrow-right-24: Environment Spec](ENVIRONMENT_SPEC.md)

-   :material-robot-outline: __PPO · Self-Play · League__

    ---

    Complete training pipelines: PPO baseline, MAPPO multi-agent, self-play loop, and AlphaStar-style league training with PFSP matchmaking and Elo tracking.

    [:octicons-arrow-right-24: Training Guide](TRAINING_GUIDE.md)

-   :material-trophy-outline: __WargamesBench__

    ---

    20 canonical scenarios for reproducible comparison.  Results guaranteed within ± 2 % win rate across identical seeds.  Open leaderboard included.

    [:octicons-arrow-right-24: Benchmark Leaderboard](wargames_bench_leaderboard.md)

-   :material-terrain: __GIS Terrain__

    ---

    Real geographic battlefield data for Waterloo, Austerlitz, Borodino, and Salamanca.  Elevation grids drive line-of-sight, movement cost, and terrain bonuses.

    [:octicons-arrow-right-24: Historical Scenarios](historical_scenarios.md)

</div>

---

## Quick Start

=== "pip install"

    ```bash
    pip install wargames-training
    ```

=== "from source"

    ```bash
    git clone https://github.com/B9android/wargames_training.git
    cd wargames_training
    pip install -e ".[dev]"
    ```

=== "first training run"

    ```python
    from envs import BattalionEnv
    from training.train import train
    import wandb

    wandb.init(project="wargames_training", config={"total_timesteps": 100_000})

    env = BattalionEnv()
    train(env=env, total_timesteps=100_000)
    ```

    Or via CLI:

    ```bash
    python -m training.train --config configs/experiment_1.yaml
    ```

=== "benchmark"

    ```bash
    wargames-bench --episodes 100 --label "my_policy_v1"
    ```

    Results are written to `docs/wargames_bench_leaderboard.md`.  Submit a PR to add your row to the public leaderboard.

---

## Project Status

| Version | Theme | Status |
|---------|-------|--------|
| **v1** | Foundation — 1v1 battalion | ✅ Complete |
| **v2** | Multi-Agent — MARL 2v2+ | ✅ Complete |
| **v3** | Hierarchy — Brigade / Division HRL | ✅ Complete |
| **v4** | League — AlphaStar-style training | ✅ Complete |
| **v5** | Real-World Interface & Analysis | ✅ Complete |
| **v6** | Physics-Accurate Simulation | 🔲 Planned |
| **v7** | Operational Scale (Corps / Army) | 🔲 Planned |
| **v8** | Transformer Policy & Architecture | 🔲 Planned |
| **v9** | Human-in-the-Loop & Decision Support | 🔲 Planned |
| **v10** | Multi-Domain & Joint Operations | 🔲 Planned |
| **v11** | Real-World Data & Transfer | 🔲 Planned |
| **v12** | Foundation Model & Open Platform | 🔲 Planned |

See the full [:octicons-arrow-right-24: Roadmap](ROADMAP.md) for milestone detail and epic breakdown.

---

## Community

- **GitHub Discussions** — research questions, benchmark results, ideas
- **Issues** — bug reports and feature requests
- **Contributing** — see [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct** — see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

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
