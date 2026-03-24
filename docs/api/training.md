# Training API

The `training` package exposes the full training workflow as a clean
programmatic Python API.  All stable symbols are importable from the
top-level `training` namespace.

## Quick-start

```python
from training import train, TrainingConfig

# Minimal run with defaults
model = train(total_timesteps=500_000, n_envs=4, enable_wandb=False)

# Full config object
config = TrainingConfig(
    total_timesteps=1_000_000,
    n_envs=8,
    curriculum_level=3,
    enable_self_play=True,
    wandb_project="my_project",
)
model = train(config)
```

## Training runners

::: training.train.TrainingConfig
::: training.train.train

## Callbacks

::: training.train.WandbCallback
::: training.train.RewardBreakdownCallback
::: training.train.EloEvalCallback
::: training.train.ManifestCheckpointCallback
::: training.train.ManifestEvalCallback

## Evaluation

::: training.evaluate
::: training.evaluate.EvaluationResult
::: training.evaluate.evaluate
::: training.evaluate.evaluate_detailed
::: training.evaluate.run_episodes_with_model

## Self-play

::: training.self_play
::: training.self_play.OpponentPool
::: training.self_play.SelfPlayCallback
::: training.self_play.WinRateVsPoolCallback
::: training.self_play.evaluate_vs_pool
::: training.self_play.TeamOpponentPool
::: training.self_play.evaluate_team_vs_pool
::: training.self_play.nash_exploitability_proxy

## Curriculum

::: training.curriculum_scheduler.CurriculumScheduler
::: training.curriculum_scheduler.CurriculumStage
::: training.curriculum_scheduler.load_v1_weights_into_mappo

## Policy Registry

::: training.policy_registry.PolicyRegistry
::: training.policy_registry.Echelon
::: training.policy_registry.PolicyEntry

## Elo Ratings

::: training.elo.EloRegistry
::: training.elo.TeamEloRegistry

## Artifacts

::: training.artifacts.CheckpointManifest
::: training.artifacts.checkpoint_name_prefix
::: training.artifacts.checkpoint_final_stem
::: training.artifacts.checkpoint_best_filename
::: training.artifacts.parse_step_from_checkpoint_name

## Benchmarks

::: training.wfm1_benchmark.WFM1Benchmark
::: training.wfm1_benchmark.WFM1BenchmarkConfig
::: training.wfm1_benchmark.WFM1BenchmarkSummary
::: training.transfer_benchmark.TransferBenchmark
::: training.transfer_benchmark.TransferEvalConfig
::: training.transfer_benchmark.TransferSummary
::: training.historical_benchmark.HistoricalBenchmark
