"""Wargames Training — training public API.

Stable interfaces for training runners, evaluation utilities, callbacks,
self-play, curriculum management, policy registry, artifacts, Elo ratings,
and benchmarks.  Import from this module to remain insulated from internal
restructuring.

Training runners
----------------
:func:`train` — train a single-agent policy with PPO.
:class:`TrainingConfig` — configuration dataclass for :func:`train`.

Evaluation
----------
:func:`evaluate` — quick win-rate evaluation.
:func:`evaluate_detailed` — detailed win/draw/loss statistics.
:func:`run_episodes_with_model` — low-level episode runner.
:class:`EvaluationResult` — structured evaluation result.

Callbacks
---------
:class:`WandbCallback` — log rollout statistics to W&B.
:class:`RewardBreakdownCallback` — per-component reward logging.
:class:`EloEvalCallback` — Elo evaluation callback.
:class:`ManifestCheckpointCallback` — manifest-aware checkpoint saving.
:class:`ManifestEvalCallback` — manifest-aware best-model saving.

Self-play
---------
:class:`OpponentPool` — pool of frozen single-agent policy snapshots.
:class:`SelfPlayCallback` — snapshot the current policy into the pool.
:class:`WinRateVsPoolCallback` — evaluate win-rate vs. the pool.
:func:`evaluate_vs_pool` — standalone pool win-rate helper.
:class:`TeamOpponentPool` — pool for MAPPO team self-play.
:func:`evaluate_team_vs_pool` — evaluate a MAPPO policy vs. pool.
:func:`nash_exploitability_proxy` — exploitability proxy metric.

Curriculum
----------
:class:`CurriculumScheduler` — win-rate-based curriculum progression.
:class:`CurriculumStage` — curriculum stage enum.
:func:`load_v1_weights_into_mappo` — warm-start MAPPO from a v1 checkpoint.

Policy registry
---------------
:class:`PolicyRegistry` — versioned multi-echelon checkpoint registry.
:class:`Echelon` — echelon enum (battalion / brigade / division).
:class:`PolicyEntry` — registry entry named-tuple.

Artifacts
---------
:class:`CheckpointManifest` — append-only JSONL checkpoint index.
:func:`checkpoint_name_prefix` — canonical periodic-checkpoint prefix.
:func:`checkpoint_final_stem` — canonical final-checkpoint stem.
:func:`checkpoint_best_filename` — canonical best-checkpoint filename.
:func:`parse_step_from_checkpoint_name` — extract step number from filename.

Elo
---
:class:`EloRegistry` — Elo rating registry.
:class:`TeamEloRegistry` — multi-agent team Elo registry.
:data:`DEFAULT_RATING` — default rating for unseen agents.
:data:`BASELINE_RATINGS` — fixed ratings for scripted baseline opponents.

Benchmarks
----------
:class:`WFM1Benchmark` — WFM-1 zero-shot evaluation.
:class:`WFM1BenchmarkConfig` — WFM-1 benchmark configuration.
:class:`WFM1BenchmarkResult` — WFM-1 per-scenario result.
:class:`WFM1BenchmarkSummary` — WFM-1 aggregate results.
:class:`TransferBenchmark` — GIS terrain transfer benchmark.
:class:`TransferEvalConfig` — transfer benchmark configuration.
:class:`TransferResult` — transfer benchmark per-condition result.
:class:`TransferSummary` — transfer benchmark aggregate results.
:class:`HistoricalBenchmark` — historical battle fidelity benchmark.
:class:`BenchmarkEntry` — per-battle benchmark entry.
:class:`BenchmarkSummary` — historical benchmark aggregate summary.
"""

from __future__ import annotations

# ── Training runners ──────────────────────────────────────────────────────
from training.train import (
    TrainingConfig,
    train,
    WandbCallback,
    RewardBreakdownCallback,
    EloEvalCallback,
    ManifestCheckpointCallback,
    ManifestEvalCallback,
)

# ── Evaluation ────────────────────────────────────────────────────────────
from training.evaluate import (
    EvaluationResult,
    evaluate,
    evaluate_detailed,
    run_episodes_with_model,
)

# ── Self-play ─────────────────────────────────────────────────────────────
from training.self_play import (
    OpponentPool,
    SelfPlayCallback,
    WinRateVsPoolCallback,
    evaluate_vs_pool,
    TeamOpponentPool,
    evaluate_team_vs_pool,
    nash_exploitability_proxy,
)

# ── Curriculum ────────────────────────────────────────────────────────────
from training.curriculum_scheduler import (
    CurriculumScheduler,
    CurriculumStage,
    load_v1_weights_into_mappo,
)

# ── Policy registry ───────────────────────────────────────────────────────
from training.policy_registry import (
    PolicyRegistry,
    Echelon,
    PolicyEntry,
)

# ── Artifacts ─────────────────────────────────────────────────────────────
from training.artifacts import (
    CheckpointManifest,
    checkpoint_name_prefix,
    checkpoint_final_stem,
    checkpoint_best_filename,
    parse_step_from_checkpoint_name,
)

# ── Elo ───────────────────────────────────────────────────────────────────
from training.elo import (
    EloRegistry,
    TeamEloRegistry,
    DEFAULT_RATING,
    BASELINE_RATINGS,
)

# ── Benchmarks ────────────────────────────────────────────────────────────
from training.wfm1_benchmark import (
    WFM1Benchmark,
    WFM1BenchmarkConfig,
    WFM1BenchmarkResult,
    WFM1BenchmarkSummary,
)
from training.transfer_benchmark import (
    TransferBenchmark,
    TransferEvalConfig,
    TransferResult,
    TransferSummary,
)
from training.historical_benchmark import (
    HistoricalBenchmark,
    BenchmarkEntry,
    BenchmarkSummary,
)

__all__ = [
    # Training runners
    "TrainingConfig",
    "train",
    "WandbCallback",
    "RewardBreakdownCallback",
    "EloEvalCallback",
    "ManifestCheckpointCallback",
    "ManifestEvalCallback",
    # Evaluation
    "EvaluationResult",
    "evaluate",
    "evaluate_detailed",
    "run_episodes_with_model",
    # Self-play
    "OpponentPool",
    "SelfPlayCallback",
    "WinRateVsPoolCallback",
    "evaluate_vs_pool",
    "TeamOpponentPool",
    "evaluate_team_vs_pool",
    "nash_exploitability_proxy",
    # Curriculum
    "CurriculumScheduler",
    "CurriculumStage",
    "load_v1_weights_into_mappo",
    # Policy registry
    "PolicyRegistry",
    "Echelon",
    "PolicyEntry",
    # Artifacts
    "CheckpointManifest",
    "checkpoint_name_prefix",
    "checkpoint_final_stem",
    "checkpoint_best_filename",
    "parse_step_from_checkpoint_name",
    # Elo
    "EloRegistry",
    "TeamEloRegistry",
    "DEFAULT_RATING",
    "BASELINE_RATINGS",
    # Benchmarks
    "WFM1Benchmark",
    "WFM1BenchmarkConfig",
    "WFM1BenchmarkResult",
    "WFM1BenchmarkSummary",
    "TransferBenchmark",
    "TransferEvalConfig",
    "TransferResult",
    "TransferSummary",
    "HistoricalBenchmark",
    "BenchmarkEntry",
    "BenchmarkSummary",
]
