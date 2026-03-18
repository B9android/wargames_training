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

import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
from training.elo import EloRegistry
from training.evaluate import run_episodes_with_model
from training.self_play import OpponentPool, SelfPlayCallback, WinRateVsPoolCallback

log = logging.getLogger(__name__)

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
# Training entry point
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
        curriculum_level=int(OmegaConf.select(cfg, "env.curriculum_level", default=5)),
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
        seed=cfg.training.seed,
        env_kwargs=env_kwargs,
    )
    eval_env = make_vec_env(
        BattalionEnv,
        n_envs=1,
        seed=cfg.training.seed + 1000,
        env_kwargs=env_kwargs,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, cfg.eval.checkpoint_freq // cfg.env.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_battalion",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(1, cfg.eval.eval_freq // cfg.env.num_envs),
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True,
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
        run_id = (
            run.id
            if run is not None and hasattr(run, "id") and run.id
            else f"run_seed{cfg.training.seed}_{os.getpid()}"
        )
        elo_registry = EloRegistry(path=elo_registry_path)
        elo_cb = EloEvalCallback(
            opponents=list(elo_opponents),
            n_eval_episodes=elo_n_eval,
            registry=elo_registry,
            agent_name=run_id,
            eval_freq=elo_eval_freq,
            env_kwargs=dict(env_kwargs),
            seed=cfg.training.seed,
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
    # PPO model
    # ------------------------------------------------------------------
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
        seed=cfg.training.seed,
        verbose=1,
    )
    log.info("PPO model created. Training for %d timesteps.", cfg.training.total_timesteps)

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
    final_path = checkpoint_dir / "ppo_battalion_final"
    model.save(str(final_path))
    log.info("Saved final model to %s.zip", final_path)

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
