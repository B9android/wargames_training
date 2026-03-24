# training/train.py
"""Main PPO training entry point.

Loads configuration via Hydra, initializes a W&B run, creates vectorized
:class:`~envs.battalion_env.BattalionEnv` environments, builds a
Stable-Baselines3 :class:`~stable_baselines3.PPO` model with
:class:`~models.mlp_policy.BattalionMlpPolicy`, attaches checkpoint and
evaluation callbacks, runs the training loop, and saves the final model.

Usage::

    # Default config (configs/default.yaml)
    python training/train.py

    # CLI overrides via Hydra
    python training/train.py training.learning_rate=1e-4 env.num_envs=4

    # Use an experiment override file
    python training/train.py --config-name experiment_1
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

# Ensure project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

import wandb
from envs.battalion_env import BattalionEnv
from envs.reward import RewardWeights
from models.mlp_policy import BattalionMlpPolicy
from training.artifacts import (
    CheckpointManifest,
    checkpoint_best_filename,
    checkpoint_final_stem,
    checkpoint_name_prefix,
    parse_step_from_checkpoint_name,
)
from training.elo import EloRegistry
from training.evaluate import run_episodes_with_model
from training.self_play import OpponentPool, SelfPlayCallback, WinRateVsPoolCallback

log = logging.getLogger(__name__)


def _stable_config_hash(cfg: DictConfig) -> str:
    """Build a deterministic hash for the fully-resolved run config."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_resume_checkpoint(
    cfg: DictConfig,
    checkpoint_dir: Path,
    manifest: Optional["CheckpointManifest"],
    periodic_prefix: str,
    current_hash: str,
) -> Optional[Path]:
    """Return a checkpoint path to resume from, or None.

    Resolution order:
    1. Explicit ``resume.checkpoint`` path from config.
    2. Manifest ``latest_periodic()`` (when manifest is present and populated).
    3. Filesystem glob scan for ``{prefix}_*_steps.zip``.
    """
    # 1. Explicit override.
    explicit = OmegaConf.select(cfg, "resume.checkpoint", default=None)
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"resume.checkpoint not found: {p}")
        _warn_hash_mismatch(
            p,
            current_hash,
            manifest,
            expected_seed=OmegaConf.select(cfg, "training.seed", default=None),
            expected_curriculum_level=OmegaConf.select(cfg, "env.curriculum_level", default=None),
        )
        return p

    auto = bool(OmegaConf.select(cfg, "resume.auto", default=False))
    if not auto:
        return None

    # 2. Manifest lookup.
    if manifest is not None:
        candidate = manifest.latest_periodic(checkpoint_dir, periodic_prefix)
        if candidate is not None:
            _warn_hash_mismatch(
                candidate,
                current_hash,
                manifest,
                expected_seed=OmegaConf.select(cfg, "training.seed", default=None),
                expected_curriculum_level=OmegaConf.select(cfg, "env.curriculum_level", default=None),
            )
            return candidate

    # 3. Filesystem fallback.
    matches = sorted(
        checkpoint_dir.glob(f"{periodic_prefix}_*_steps.zip"),
        key=lambda p: parse_step_from_checkpoint_name(p) or -1,
    )
    if matches:
        candidate = matches[-1]
        log.warning(
            "resume.auto: manifest unavailable — found checkpoint via filesystem scan: %s",
            candidate,
        )
        return candidate

    log.info("resume.auto set but no existing checkpoint found; starting fresh.")
    return None


def _warn_hash_mismatch(
    checkpoint_path: Path,
    current_hash: str,
    manifest: Optional["CheckpointManifest"],
    *,
    expected_seed: Optional[int] = None,
    expected_curriculum_level: Optional[int] = None,
) -> None:
    """Emit warnings when manifest metadata does not match the current run."""
    if manifest is None:
        return
    row = manifest.latest_entry_for_path(checkpoint_path)
    if row is None:
        return

    recorded = str(row.get("config_hash", "") or "")
    if recorded and recorded != current_hash:
        log.warning(
            "Config hash mismatch for checkpoint %s: recorded=%s current=%s. "
            "Hyperparameters may differ from original run.",
            checkpoint_path.name,
            recorded[:12],
            current_hash[:12],
        )

    if expected_seed is not None and row.get("seed") != expected_seed:
        log.warning(
            "Seed mismatch for checkpoint %s: recorded=%s current=%s. "
            "Resume may continue from a different run lineage.",
            checkpoint_path.name,
            row.get("seed"),
            expected_seed,
        )

    if (
        expected_curriculum_level is not None
        and row.get("curriculum_level") != expected_curriculum_level
    ):
        log.warning(
            "Curriculum mismatch for checkpoint %s: recorded=%s current=%s. "
            "Opponent difficulty may differ from the original run.",
            checkpoint_path.name,
            row.get("curriculum_level"),
            expected_curriculum_level,
        )


class ManifestCheckpointCallback(CheckpointCallback):
    """Checkpoint callback that appends periodic saves to the manifest immediately."""

    def __init__(
        self,
        *,
        manifest: Optional[CheckpointManifest],
        seed: int,
        curriculum_level: int,
        run_id: Optional[str],
        config_hash: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._manifest = manifest
        self._seed = int(seed)
        self._curriculum_level = int(curriculum_level)
        self._run_id = run_id
        self._config_hash = str(config_hash)

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self._manifest is None:
            return result
        if self.n_calls % self.save_freq != 0:
            return result

        checkpoint_path = Path(self._checkpoint_path(extension="zip"))
        if checkpoint_path.exists():
            self._manifest.register(
                checkpoint_path,
                artifact_type="periodic",
                seed=self._seed,
                curriculum_level=self._curriculum_level,
                run_id=self._run_id,
                config_hash=self._config_hash,
                step=parse_step_from_checkpoint_name(checkpoint_path),
            )
        return result


class ManifestEvalCallback(EvalCallback):
    """Eval callback that materializes best-model metadata at creation time."""

    def __init__(
        self,
        eval_env,
        *,
        manifest: Optional[CheckpointManifest],
        seed: int,
        curriculum_level: int,
        run_id: Optional[str],
        config_hash: str,
        enable_naming_v2: bool,
        **kwargs,
    ) -> None:
        super().__init__(eval_env, **kwargs)
        self._manifest = manifest
        self._seed = int(seed)
        self._curriculum_level = int(curriculum_level)
        self._run_id = run_id
        self._config_hash = str(config_hash)
        self._enable_naming_v2 = bool(enable_naming_v2)

    def _on_step(self) -> bool:
        best_before = self.best_mean_reward
        result = super()._on_step()
        if self._manifest is None:
            return result
        if self.best_mean_reward <= best_before:
            return result

        best_alias_zip = Path(self.best_model_save_path) / "best_model.zip"
        if not best_alias_zip.exists():
            return result

        self._manifest.register(
            best_alias_zip,
            artifact_type="best_alias",
            seed=self._seed,
            curriculum_level=self._curriculum_level,
            run_id=self._run_id,
            config_hash=self._config_hash,
            step=int(self.num_timesteps),
        )

        best_canonical_zip = Path(self.best_model_save_path) / checkpoint_best_filename(
            seed=self._seed,
            curriculum_level=self._curriculum_level,
            enable_v2=self._enable_naming_v2,
        )
        if best_canonical_zip != best_alias_zip:
            best_canonical_zip.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_alias_zip, best_canonical_zip)
        if best_canonical_zip.exists():
            self._manifest.register(
                best_canonical_zip,
                artifact_type="best",
                seed=self._seed,
                curriculum_level=self._curriculum_level,
                run_id=self._run_id,
                config_hash=self._config_hash,
                step=int(self.num_timesteps),
            )
        return result

# ---------------------------------------------------------------------------
# W&B callback
# ---------------------------------------------------------------------------


class WandbCallback(BaseCallback):
    """Logs SB3 training metrics to an active W&B run.

    Emits episode-level rollout statistics (mean reward and episode length)
    every ``log_freq`` environment steps, and policy-update losses (if
    available from the SB3 logger) at the end of each rollout.

    Parameters
    ----------
    log_freq:
        How often (in environment steps) to log rollout statistics.
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0 and len(self.model.ep_info_buffer) > 0:
            ep_infos = list(self.model.ep_info_buffer)
            mean_reward = float(np.mean([ep["r"] for ep in ep_infos]))
            mean_length = float(np.mean([ep["l"] for ep in ep_infos]))
            wandb.log(
                {
                    "rollout/ep_rew_mean": mean_reward,
                    "rollout/ep_len_mean": mean_length,
                    "time/total_timesteps": self.num_timesteps,
                },
                step=self.num_timesteps,
            )
        return True

    def _on_rollout_end(self) -> None:
        """Log policy-update losses after each PPO update."""
        logger_kvs: dict = self.model.logger.name_to_value  # type: ignore[attr-defined]
        if logger_kvs:
            wandb.log(
                {f"train/{k}": v for k, v in logger_kvs.items()},
                step=self.num_timesteps,
            )


class RewardBreakdownCallback(BaseCallback):
    """Logs per-component reward breakdown to W&B at episode boundaries.

    Accumulates reward components from ``info`` dicts (populated by
    :class:`~envs.battalion_env.BattalionEnv`) across all parallel
    environments every step and rolls them into per-episode totals when
    an episode ends.  The episode means are logged to W&B every
    ``log_freq`` timesteps.  Any remaining episodes at the end of
    training are flushed in ``_on_training_end()``.

    Parameters
    ----------
    log_freq:
        How often (in environment steps) to flush accumulated episode
        means to W&B.
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    _COMPONENT_KEYS: tuple[str, ...] = (
        "reward/delta_enemy_strength",
        "reward/delta_own_strength",
        "reward/survival_bonus",
        "reward/win_bonus",
        "reward/loss_penalty",
        "reward/time_penalty",
        "reward/total",
    )

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        # Per-env step accumulators, indexed by env index.  Initialised in
        # _on_training_start() once the number of parallel envs is known.
        self._step_sums: list[dict[str, float]] = []
        # Completed-episode accumulators (sum across episodes and episode count).
        self._ep_sums: dict[str, float] = {k: 0.0 for k in self._COMPONENT_KEYS}
        self._ep_count: int = 0

    def _on_training_start(self) -> None:
        """Initialise per-env step accumulators once the env count is known."""
        n_envs = self.training_env.num_envs  # type: ignore[union-attr]
        self._step_sums = [
            {k: 0.0 for k in self._COMPONENT_KEYS} for _ in range(n_envs)
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            # Accumulate each component value for this step.
            for key in self._COMPONENT_KEYS:
                self._step_sums[env_idx][key] += float(info.get(key, 0.0))

            if done:
                # Episode complete — transfer step sums to episode accumulators.
                for key in self._COMPONENT_KEYS:
                    self._ep_sums[key] += self._step_sums[env_idx][key]
                    self._step_sums[env_idx][key] = 0.0
                self._ep_count += 1

        if self.num_timesteps % self.log_freq == 0 and self._ep_count > 0:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        """Flush any remaining accumulated episodes at the end of training."""
        if self._ep_count > 0:
            self._flush()

    def _flush(self) -> None:
        """Log episode means to W&B and reset accumulators."""
        means = {
            f"reward_breakdown/{k.split('/')[-1]}": v / self._ep_count
            for k, v in self._ep_sums.items()
        }
        means["time/total_timesteps"] = self.num_timesteps
        wandb.log(means, step=self.num_timesteps)
        self._ep_sums = {k: 0.0 for k in self._COMPONENT_KEYS}
        self._ep_count = 0


# ---------------------------------------------------------------------------
# Elo evaluation callback
# ---------------------------------------------------------------------------


class EloEvalCallback(BaseCallback):
    """Evaluate the current policy vs scripted opponents and log Elo to W&B.

    Every ``eval_freq`` environment steps the callback runs *n_eval_episodes*
    episodes against each opponent in *opponents* using the live model,
    updates the :class:`~training.elo.EloRegistry`, persists it to disk, and
    logs per-opponent Elo ratings and win rates to W&B.

    Parameters
    ----------
    opponents:
        List of opponent identifiers (e.g. ``["scripted_l1", "scripted_l3",
        "scripted_l5"]``).  Each must be a valid argument to
        :func:`~training.evaluate.run_episodes_with_model`.
    n_eval_episodes:
        Number of episodes to run per opponent per evaluation.
    registry:
        :class:`~training.elo.EloRegistry` instance used for ratings.
    agent_name:
        Key used to identify this training run in the registry.
    eval_freq:
        How often (in environment steps) to trigger evaluation.
    env_kwargs:
        Keyword arguments forwarded to :class:`~envs.battalion_env.BattalionEnv`
        when creating evaluation environments.  This ensures Elo evaluation
        uses the same map size, terrain settings, and reward weights as the
        training run.
    seed:
        Base random seed for evaluation episodes.
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        opponents: list[str],
        n_eval_episodes: int,
        registry: EloRegistry,
        agent_name: str,
        eval_freq: int,
        env_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.opponents = opponents
        self.n_eval_episodes = n_eval_episodes
        self.registry = registry
        self.agent_name = agent_name
        self.eval_freq = eval_freq
        self.env_kwargs = env_kwargs or {}
        self.seed = seed
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
        if (
            self.num_timesteps - self._last_eval_step >= self.eval_freq
            and self.num_timesteps > 0
        ):
            self._run_elo_eval()
            self._last_eval_step = self.num_timesteps
        return True

    def _run_elo_eval(self) -> None:
        """Evaluate vs all opponents and update the Elo registry."""
        log_dict: dict = {"time/total_timesteps": self.num_timesteps}
        for opponent in self.opponents:
            result = run_episodes_with_model(
                self.model,
                opponent=opponent,
                n_episodes=self.n_eval_episodes,
                deterministic=True,
                seed=self.seed,
                env_kwargs=self.env_kwargs,
            )
            outcome = (result.wins + 0.5 * result.draws) / result.n_episodes
            delta = self.registry.update(
                agent=self.agent_name,
                opponent=opponent,
                outcome=outcome,
                n_games=result.n_episodes,
            )
            elo_rating = self.registry.get_rating(self.agent_name)
            log_dict[f"elo/rating_vs_{opponent}"] = elo_rating
            log_dict[f"elo/win_rate_vs_{opponent}"] = result.win_rate
            log_dict[f"elo/delta_vs_{opponent}"] = delta
            if self.verbose >= 1:
                log.info(
                    "EloEval [%d steps] vs %s — win %.1f%% Elo %.1f (Δ%+.1f)",
                    self.num_timesteps,
                    opponent,
                    result.win_rate * 100,
                    elo_rating,
                    delta,
                )
        # Persist only when the registry has a backing file.
        if self.registry.can_save:
            self.registry.save()
        wandb.log(log_dict, step=self.num_timesteps)


# ---------------------------------------------------------------------------
# TrainingConfig — programmatic training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for a single PPO training run on :class:`~envs.battalion_env.BattalionEnv`.

    All fields are optional; the defaults match ``configs/default.yaml``.
    Instances can be passed directly to :func:`train` or individual fields
    can be overridden via ``**kwargs`` at the call site.

    Examples
    --------
    Minimal run with defaults::

        from training import train, TrainingConfig
        model = train(TrainingConfig(total_timesteps=500_000))

    Override specific fields at call time without constructing a config::

        model = train(total_timesteps=200_000, n_envs=4, enable_wandb=False)
    """

    # ── PPO hyperparameters ───────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    learning_rate: float = 3.0e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "auto"

    # ── Environment ───────────────────────────────────────────────────────
    n_envs: int = 8
    curriculum_level: int = 5
    map_width: float = 1000.0
    map_height: float = 1000.0
    max_steps: int = 500
    randomize_terrain: bool = True
    hill_speed_factor: float = 0.5

    # ── Reward weights ────────────────────────────────────────────────────
    reward_delta_enemy_strength: float = 5.0
    reward_delta_own_strength: float = 5.0
    reward_survival_bonus: float = 0.0
    reward_win_bonus: float = 10.0
    reward_loss_penalty: float = -10.0
    reward_time_penalty: float = -0.01

    # ── Checkpointing and evaluation ──────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 100_000
    eval_freq: int = 50_000
    n_eval_episodes: int = 20
    eval_deterministic: bool = True

    # ── Artifact management ───────────────────────────────────────────────
    write_manifest: bool = True
    manifest_path: str = "checkpoints/manifest.jsonl"
    enable_naming_v2: bool = True
    keep_legacy_aliases: bool = True

    # ── W&B experiment tracking ───────────────────────────────────────────
    enable_wandb: bool = True
    wandb_project: str = "wargames_training"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["v1", "ppo"])
    wandb_log_freq: int = 1000

    # ── Self-play (disabled by default) ───────────────────────────────────
    enable_self_play: bool = False
    self_play_pool_dir: str = "checkpoints/pool"
    self_play_pool_max_size: int = 10
    self_play_snapshot_freq: int = 50_000
    self_play_eval_freq: int = 50_000
    self_play_n_eval_episodes: int = 20
    self_play_use_latest_for_eval: bool = False

    # ── Elo evaluation (disabled by default) ─────────────────────────────
    elo_opponents: List[str] = field(default_factory=list)
    elo_registry_path: str = "checkpoints/elo_registry.json"
    elo_eval_freq: int = 50_000
    elo_n_eval_episodes: int = 20

    # ── Retention / pruning ───────────────────────────────────────────────
    keep_periodic: int = 5
    keep_self_play_snapshots: int = 10
    prune_on_run_end: bool = True

    # ── Logging ───────────────────────────────────────────────────────────
    verbose: int = 1
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# train() — programmatic training entry point
# ---------------------------------------------------------------------------


def train(
    config: Optional[TrainingConfig] = None,
    *,
    extra_callbacks: Optional[List[BaseCallback]] = None,
    resume: Optional[Union[str, Path]] = None,
    **override_kwargs: Any,
) -> PPO:
    """Train a PPO policy on :class:`~envs.battalion_env.BattalionEnv`.

    This is the programmatic entry-point for training, fully decoupled from
    Hydra/YAML so it can be called from any Python script or notebook.

    Parameters
    ----------
    config:
        :class:`TrainingConfig` instance.  When ``None`` a default config is
        used.  Individual fields can be overridden via ``**override_kwargs``.
    extra_callbacks:
        Additional SB3 :class:`~stable_baselines3.common.callbacks.BaseCallback`
        instances appended to the built-in callback list.
    resume:
        Path to an existing ``.zip`` checkpoint to resume training from.
        When provided, the model weights and optimizer state are loaded
        before training begins (the ``.zip`` extension may be omitted).
    **override_kwargs:
        Keyword arguments that override individual :class:`TrainingConfig`
        fields.  Any unrecognised key raises :exc:`ValueError`.

    Returns
    -------
    stable_baselines3.PPO
        The trained model.  Periodic checkpoints, best-model, and a manifest
        are written to *config.checkpoint_dir* during training.

    Raises
    ------
    ValueError
        If any configuration value is invalid or an unrecognised
        ``**override_kwarg`` key is passed.
    FileNotFoundError
        If *resume* points to a path that does not exist on disk.

    Examples
    --------
    Quickstart with defaults::

        from training import train
        model = train(total_timesteps=200_000, n_envs=4, enable_wandb=False)

    Full config::

        from training import train, TrainingConfig
        config = TrainingConfig(
            total_timesteps=1_000_000,
            n_envs=8,
            curriculum_level=3,
            enable_self_play=True,
        )
        model = train(config)
    """
    if config is None:
        config = TrainingConfig()

    # Apply per-call overrides.
    if override_kwargs:
        valid_fields = {f.name for f in dataclasses.fields(config)}
        unknown = set(override_kwargs) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown TrainingConfig fields: {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(valid_fields))}."
            )
        config = dataclasses.replace(config, **override_kwargs)

    # Validate critical parameters.
    if config.total_timesteps < 1:
        raise ValueError(
            f"total_timesteps must be >= 1, got {config.total_timesteps}."
        )
    if config.n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {config.n_envs}.")
    if config.checkpoint_freq <= 0:
        raise ValueError(
            f"checkpoint_freq must be > 0, got {config.checkpoint_freq}."
        )
    if config.eval_freq <= 0:
        raise ValueError(f"eval_freq must be > 0, got {config.eval_freq}.")

    # Resolve paths relative to the current working directory.
    checkpoint_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint manifest.
    manifest_path_resolved = Path(config.manifest_path)
    manifest: Optional[CheckpointManifest] = (
        CheckpointManifest(manifest_path_resolved) if config.write_manifest else None
    )

    # Deterministic config hash for traceability.
    config_dict = dataclasses.asdict(config)
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    periodic_prefix = checkpoint_name_prefix(
        seed=config.seed,
        curriculum_level=config.curriculum_level,
        enable_v2=config.enable_naming_v2,
    )

    # W&B initialisation (optional; failures are non-fatal).
    run: Any = None
    run_id: Optional[str] = None
    if config.enable_wandb:
        try:
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config_dict,
                tags=list(config.wandb_tags),
                sync_tensorboard=False,
                reinit=True,
            )
            run_id = (
                run.id
                if run is not None and hasattr(run, "id") and run.id
                else None
            )
            log.info("W&B run: %s", getattr(run, "url", "offline"))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "W&B initialisation failed (%s) — continuing without W&B.", exc
            )

    # Reward weights.
    reward_weights = RewardWeights(
        delta_enemy_strength=config.reward_delta_enemy_strength,
        delta_own_strength=config.reward_delta_own_strength,
        survival_bonus=config.reward_survival_bonus,
        win_bonus=config.reward_win_bonus,
        loss_penalty=config.reward_loss_penalty,
        time_penalty=config.reward_time_penalty,
    )

    env_kwargs: dict = dict(
        map_width=config.map_width,
        map_height=config.map_height,
        max_steps=config.max_steps,
        randomize_terrain=config.randomize_terrain,
        hill_speed_factor=config.hill_speed_factor,
        curriculum_level=config.curriculum_level,
        reward_weights=reward_weights,
    )

    vec_env = make_vec_env(
        BattalionEnv,
        n_envs=config.n_envs,
        seed=config.seed,
        env_kwargs=env_kwargs,
    )
    eval_env = make_vec_env(
        BattalionEnv,
        n_envs=1,
        seed=config.seed + 1000,
        env_kwargs=env_kwargs,
    )

    # Built-in callbacks.
    _checkpoint_cb = ManifestCheckpointCallback(
        save_freq=max(1, config.checkpoint_freq // config.n_envs),
        save_path=str(checkpoint_dir),
        name_prefix=periodic_prefix,
        manifest=manifest,
        seed=config.seed,
        curriculum_level=config.curriculum_level,
        run_id=run_id,
        config_hash=config_hash,
        verbose=config.verbose,
    )
    _eval_cb = ManifestEvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(1, config.eval_freq // config.n_envs),
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.eval_deterministic,
        manifest=manifest,
        seed=config.seed,
        curriculum_level=config.curriculum_level,
        run_id=run_id,
        config_hash=config_hash,
        enable_naming_v2=config.enable_naming_v2,
        verbose=config.verbose,
    )
    all_callbacks: list = [_checkpoint_cb, _eval_cb]

    if config.enable_wandb:
        all_callbacks.append(WandbCallback(log_freq=config.wandb_log_freq))
        all_callbacks.append(RewardBreakdownCallback(log_freq=config.wandb_log_freq))

    # Self-play callbacks (optional).
    if config.enable_self_play:
        _pool = OpponentPool(
            pool_dir=Path(config.self_play_pool_dir),
            max_size=config.self_play_pool_max_size,
        )
        _sp_cb = SelfPlayCallback(
            pool=_pool,
            snapshot_freq=config.self_play_snapshot_freq,
            vec_env=vec_env,
            verbose=config.verbose,
            manifest=manifest,
            seed=config.seed,
            curriculum_level=config.curriculum_level,
            run_id=run_id,
            config_hash=config_hash,
        )
        _wr_cb = WinRateVsPoolCallback(
            pool=_pool,
            eval_freq=config.self_play_eval_freq,
            n_eval_episodes=config.self_play_n_eval_episodes,
            deterministic=True,
            use_latest=config.self_play_use_latest_for_eval,
            verbose=config.verbose,
        )
        all_callbacks.extend([_sp_cb, _wr_cb])
        log.info(
            "Self-play enabled: pool_dir=%s, max_size=%d",
            config.self_play_pool_dir,
            config.self_play_pool_max_size,
        )

    # Elo callbacks (optional).
    if config.elo_opponents:
        _elo_registry = EloRegistry(path=Path(config.elo_registry_path))
        _elo_run_id = run_id or f"run_seed{config.seed}"
        _elo_cb = EloEvalCallback(
            opponents=list(config.elo_opponents),
            n_eval_episodes=config.elo_n_eval_episodes,
            registry=_elo_registry,
            agent_name=_elo_run_id,
            eval_freq=config.elo_eval_freq,
            env_kwargs=dict(env_kwargs),
            seed=config.seed,
            verbose=config.verbose,
        )
        all_callbacks.append(_elo_cb)

    # Merge extra caller-supplied callbacks.
    all_callbacks.extend(extra_callbacks or [])

    # Resolve resume checkpoint.
    resume_path: Optional[Path] = None
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.exists():
            zip_path = Path(str(resume_path) + ".zip")
            if zip_path.exists():
                resume_path = zip_path
            else:
                raise FileNotFoundError(
                    f"Resume checkpoint not found: '{resume}'. "
                    "Provide an existing .zip path or omit the extension."
                )

    # Build or reload PPO model.
    if resume_path is not None:
        log.info("Resuming from checkpoint: %s", resume_path)
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            device=config.device,
            custom_objects={
                "learning_rate": config.learning_rate,
                "clip_range": config.clip_range,
            },
        )
    else:
        model = PPO(
            BattalionMlpPolicy,
            vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
        )
    log.info("PPO model ready. Training for %d timesteps.", config.total_timesteps)

    # Training loop.
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(all_callbacks),
        progress_bar=False,
        reset_num_timesteps=resume_path is None,
    )

    # Save final checkpoint.
    final_stem = checkpoint_final_stem(
        seed=config.seed,
        curriculum_level=config.curriculum_level,
        enable_v2=config.enable_naming_v2,
    )
    final_path = checkpoint_dir / final_stem
    model.save(str(final_path))
    log.info("Saved final model to %s.zip", final_path)

    legacy_alias = checkpoint_dir / "ppo_battalion_final"
    if config.keep_legacy_aliases and final_path != legacy_alias:
        model.save(str(legacy_alias))

    # Register artifacts in the manifest.
    if manifest is not None:
        for periodic_zip in checkpoint_dir.glob(f"{periodic_prefix}_*_steps.zip"):
            manifest.register(
                periodic_zip,
                artifact_type="periodic",
                seed=config.seed,
                curriculum_level=config.curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=parse_step_from_checkpoint_name(periodic_zip),
            )
        final_zip = final_path.with_suffix(".zip")
        if final_zip.exists():
            manifest.register(
                final_zip,
                artifact_type="final",
                seed=config.seed,
                curriculum_level=config.curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=int(getattr(model, "num_timesteps", 0) or 0),
            )
        # Prune old checkpoints if requested.
        if config.prune_on_run_end and config.keep_periodic > 0:
            pruned = manifest.prune_periodic(
                checkpoint_dir, periodic_prefix, keep_last=config.keep_periodic
            )
            if pruned:
                log.info("Pruned %d old periodic checkpoint(s).", len(pruned))
        if config.enable_self_play and config.keep_self_play_snapshots > 0:
            pruned_sp = manifest.prune_self_play_snapshots(
                Path(config.self_play_pool_dir),
                keep_last=config.keep_self_play_snapshots,
            )
            if pruned_sp:
                log.info("Pruned %d old self-play snapshot(s).", len(pruned_sp))

    # Upload final artifact to W&B and close the run.
    if run is not None:
        artifact = wandb.Artifact(name="ppo_battalion_final", type="model")
        zip_str = str(final_path) + ".zip"
        if Path(zip_str).exists():
            artifact.add_file(zip_str)
            run.log_artifact(artifact)
        run.finish()

    vec_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Hydra training entry point (CLI / YAML config)
# ---------------------------------------------------------------------------


@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run a PPO training session from a Hydra configuration.

    Parameters
    ----------
    cfg:
        Hydra-merged configuration dict (see ``configs/default.yaml``).
    """
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    # ------------------------------------------------------------------
    # Resolve paths relative to project root (Hydra changes cwd).
    # ------------------------------------------------------------------
    checkpoint_dir = _PROJECT_ROOT / cfg.eval.checkpoint_dir
    log_dir = _PROJECT_ROOT / cfg.logging.log_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.training.seed)
    curriculum_level = int(OmegaConf.select(cfg, "env.curriculum_level", default=5))
    enable_naming_v2 = bool(OmegaConf.select(cfg, "artifacts.enable_naming_v2", default=False))
    keep_legacy_aliases = bool(
        OmegaConf.select(cfg, "artifacts.keep_legacy_aliases", default=True)
    )
    write_manifest = bool(OmegaConf.select(cfg, "artifacts.write_manifest", default=True))
    manifest_rel = str(
        OmegaConf.select(cfg, "artifacts.manifest_path", default="checkpoints/manifest.jsonl")
    )
    manifest_path = _PROJECT_ROOT / manifest_rel
    manifest = CheckpointManifest(manifest_path) if write_manifest else None
    config_hash = _stable_config_hash(cfg)
    periodic_prefix = checkpoint_name_prefix(
        seed=seed,
        curriculum_level=curriculum_level,
        enable_v2=enable_naming_v2,
    )

    # ------------------------------------------------------------------
    # W&B initialisation
    # ------------------------------------------------------------------
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        sync_tensorboard=False,
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")
    run_id = run.id if run is not None and hasattr(run, "id") and run.id else None

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    _default_w = RewardWeights()
    reward_weights = RewardWeights(
        delta_enemy_strength=float(
            OmegaConf.select(cfg, "reward.delta_enemy_strength", default=_default_w.delta_enemy_strength)
        ),
        delta_own_strength=float(
            OmegaConf.select(cfg, "reward.delta_own_strength", default=_default_w.delta_own_strength)
        ),
        survival_bonus=float(
            OmegaConf.select(cfg, "reward.survival_bonus", default=_default_w.survival_bonus)
        ),
        win_bonus=float(
            OmegaConf.select(cfg, "reward.win_bonus", default=_default_w.win_bonus)
        ),
        loss_penalty=float(
            OmegaConf.select(cfg, "reward.loss_penalty", default=_default_w.loss_penalty)
        ),
        time_penalty=float(
            OmegaConf.select(cfg, "reward.time_penalty", default=_default_w.time_penalty)
        ),
    )

    env_kwargs = dict(
        map_width=cfg.env.map_width,
        map_height=cfg.env.map_height,
        max_steps=cfg.env.max_steps,
        randomize_terrain=OmegaConf.select(cfg, "env.randomize_terrain", default=True),
        hill_speed_factor=OmegaConf.select(cfg, "env.hill_speed_factor", default=0.5),
        curriculum_level=curriculum_level,
        reward_weights=reward_weights,
    )

    # Basic config validation to avoid invalid vectorized envs or callback frequencies.
    if cfg.env.num_envs < 1:
        raise ValueError(f"cfg.env.num_envs must be >= 1, got {cfg.env.num_envs}.")
    if cfg.eval.checkpoint_freq <= 0:
        raise ValueError(
            f"cfg.eval.checkpoint_freq must be a positive integer, got {cfg.eval.checkpoint_freq}."
        )
    if cfg.eval.eval_freq <= 0:
        raise ValueError(
            f"cfg.eval.eval_freq must be a positive integer, got {cfg.eval.eval_freq}."
        )

    vec_env = make_vec_env(
        BattalionEnv,
        n_envs=cfg.env.num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
    eval_env = make_vec_env(
        BattalionEnv,
        n_envs=1,
        seed=seed + 1000,
        env_kwargs=env_kwargs,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ManifestCheckpointCallback(
        save_freq=max(1, cfg.eval.checkpoint_freq // cfg.env.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix=periodic_prefix,
        manifest=manifest,
        seed=seed,
        curriculum_level=curriculum_level,
        run_id=run_id,
        config_hash=config_hash,
        verbose=1,
    )

    eval_cb = ManifestEvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(1, cfg.eval.eval_freq // cfg.env.num_envs),
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True,
        manifest=manifest,
        seed=seed,
        curriculum_level=curriculum_level,
        run_id=run_id,
        config_hash=config_hash,
        enable_naming_v2=enable_naming_v2,
        verbose=1,
    )

    wandb_cb = WandbCallback(log_freq=cfg.wandb.log_freq)
    reward_breakdown_cb = RewardBreakdownCallback(log_freq=cfg.wandb.log_freq)

    # ------------------------------------------------------------------
    # Self-play callbacks (optional — enabled via cfg.self_play.enabled)
    # ------------------------------------------------------------------
    extra_callbacks: list = []
    if OmegaConf.select(cfg, "self_play.enabled", default=False):
        pool_dir = _PROJECT_ROOT / cfg.self_play.pool_dir
        pool = OpponentPool(
            pool_dir=pool_dir,
            max_size=int(cfg.self_play.pool_max_size),
        )
        sp_snapshot_freq = int(cfg.self_play.snapshot_freq)
        sp_eval_freq = int(cfg.self_play.eval_freq)
        sp_n_eval = int(cfg.self_play.n_eval_episodes)
        self_play_cb = SelfPlayCallback(
            pool=pool,
            snapshot_freq=sp_snapshot_freq,
            vec_env=vec_env,
            verbose=1,
            manifest=manifest,
            seed=seed,
            curriculum_level=curriculum_level,
            run_id=run_id,
            config_hash=config_hash,
        )
        win_rate_cb = WinRateVsPoolCallback(
            pool=pool,
            eval_freq=sp_eval_freq,
            n_eval_episodes=sp_n_eval,
            deterministic=True,
            use_latest=bool(cfg.self_play.use_latest_for_eval),
            verbose=1,
        )
        extra_callbacks.extend([self_play_cb, win_rate_cb])
        log.info(
            "Self-play enabled: pool_dir=%s, pool_max_size=%d, snapshot_freq=%d, eval_freq=%d",
            pool_dir,
            pool.max_size,
            sp_snapshot_freq,
            sp_eval_freq,
        )

    # ------------------------------------------------------------------
    # Elo evaluation callback (optional — enabled via cfg.eval.elo_opponents)
    # ------------------------------------------------------------------
    elo_opponents = list(OmegaConf.select(cfg, "eval.elo_opponents", default=[]))
    if elo_opponents:
        elo_registry_path = _PROJECT_ROOT / OmegaConf.select(
            cfg, "eval.elo_registry", default="checkpoints/elo_registry.json"
        )
        elo_eval_freq = int(
            OmegaConf.select(cfg, "eval.elo_eval_freq", default=cfg.eval.eval_freq)
        )
        elo_n_eval = int(
            OmegaConf.select(cfg, "eval.elo_n_eval_episodes", default=cfg.eval.n_eval_episodes)
        )
        elo_run_id = run_id or f"run_seed{seed}_{os.getpid()}"
        elo_registry = EloRegistry(path=elo_registry_path)
        elo_cb = EloEvalCallback(
            opponents=list(elo_opponents),
            n_eval_episodes=elo_n_eval,
            registry=elo_registry,
            agent_name=elo_run_id,
            eval_freq=elo_eval_freq,
            env_kwargs=dict(env_kwargs),
            seed=seed,
            verbose=1,
        )
        extra_callbacks.append(elo_cb)
        log.info(
            "Elo eval enabled: opponents=%s, eval_freq=%d, registry=%s",
            elo_opponents,
            elo_eval_freq,
            elo_registry_path,
        )

    # ------------------------------------------------------------------
    # Resolve resume checkpoint
    # ------------------------------------------------------------------
    resume_path = _resolve_resume_checkpoint(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        periodic_prefix=periodic_prefix,
        current_hash=config_hash,
    )

    # ------------------------------------------------------------------
    # PPO model
    # ------------------------------------------------------------------
    if resume_path is not None:
        log.info("Resuming from checkpoint: %s", resume_path)
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            device="auto",
            custom_objects={
                "learning_rate": cfg.training.learning_rate,
                "clip_range": cfg.training.clip_range,
            },
        )
    else:
        model = PPO(
            BattalionMlpPolicy,
            vec_env,
            learning_rate=cfg.training.learning_rate,
            n_steps=cfg.training.n_steps,
            batch_size=cfg.training.batch_size,
            n_epochs=cfg.training.n_epochs,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
            clip_range=cfg.training.clip_range,
            ent_coef=cfg.training.ent_coef,
            vf_coef=cfg.training.vf_coef,
            max_grad_norm=cfg.training.max_grad_norm,
            seed=seed,
            verbose=1,
        )
    log.info("PPO model ready. Training for %d timesteps.", cfg.training.total_timesteps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb, wandb_cb, reward_breakdown_cb, *extra_callbacks]),
        progress_bar=False,
        reset_num_timesteps=True,
    )

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    final_stem = checkpoint_final_stem(
        seed=seed,
        curriculum_level=curriculum_level,
        enable_v2=enable_naming_v2,
    )
    final_path = checkpoint_dir / final_stem
    model.save(str(final_path))
    log.info("Saved final model to %s.zip", final_path)

    legacy_final_alias_path = checkpoint_dir / "ppo_battalion_final"
    if keep_legacy_aliases and final_path != legacy_final_alias_path:
        model.save(str(legacy_final_alias_path))

    best_alias_zip = checkpoint_dir / "best" / "best_model.zip"
    best_canonical_zip = checkpoint_dir / "best" / checkpoint_best_filename(
        seed=seed,
        curriculum_level=curriculum_level,
        enable_v2=enable_naming_v2,
    )
    if best_alias_zip.exists() and best_alias_zip != best_canonical_zip:
        best_canonical_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_alias_zip, best_canonical_zip)

    if manifest is not None:
        for periodic_zip in checkpoint_dir.glob(f"{periodic_prefix}_*_steps.zip"):
            manifest.register(
                periodic_zip,
                artifact_type="periodic",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=parse_step_from_checkpoint_name(periodic_zip),
            )

        final_zip = final_path.with_suffix(".zip")
        if final_zip.exists():
            manifest.register(
                final_zip,
                artifact_type="final",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=int(getattr(model, "num_timesteps", 0) or 0),
            )

        legacy_final_zip = legacy_final_alias_path.with_suffix(".zip")
        if keep_legacy_aliases and legacy_final_zip.exists():
            manifest.register(
                legacy_final_zip,
                artifact_type="final_alias",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=int(getattr(model, "num_timesteps", 0) or 0),
            )

        if best_alias_zip.exists():
            manifest.register(
                best_alias_zip,
                artifact_type="best_alias",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=int(getattr(model, "num_timesteps", 0) or 0),
            )
        if best_canonical_zip.exists():
            manifest.register(
                best_canonical_zip,
                artifact_type="best",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id=run_id,
                config_hash=config_hash,
                step=int(getattr(model, "num_timesteps", 0) or 0),
            )

    # ── Retention / pruning ──────────────────────────────────────────────
    retention_cfg = cfg.get("retention", {})
    if retention_cfg.get("prune_on_run_end", True) and write_manifest:
        keep_periodic = int(retention_cfg.get("keep_periodic", 5))
        keep_snapshots = int(retention_cfg.get("keep_self_play_snapshots", 10))
        prefix = checkpoint_name_prefix(
            seed=seed,
            curriculum_level=curriculum_level,
            enable_v2=enable_naming_v2,
        )
        if keep_periodic > 0:
            pruned = manifest.prune_periodic(checkpoint_dir, prefix, keep_last=keep_periodic)
            if pruned:
                log.info("Pruned %d old periodic checkpoint(s)", len(pruned))
        sp_cfg = cfg.get("self_play", {})
        if sp_cfg.get("enabled", False) and keep_snapshots > 0:
            pool_dir = Path(sp_cfg.get("pool_dir", "checkpoints/pool"))
            pruned_sp = manifest.prune_self_play_snapshots(
                pool_dir, keep_last=keep_snapshots
            )
            if pruned_sp:
                log.info("Pruned %d old self-play snapshot(s)", len(pruned_sp))

    if run is not None:
        artifact = wandb.Artifact(name="ppo_battalion_final", type="model")
        zip_path = str(final_path) + ".zip"
        if Path(zip_path).exists():
            artifact.add_file(zip_path)
            run.log_artifact(artifact)
        run.finish()

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
