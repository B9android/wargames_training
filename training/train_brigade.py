"""Brigade Commander training — PPO on the :class:`~envs.brigade_env.BrigadeEnv`.

Trains a brigade-level RL agent that issues macro-commands (options) to
frozen battalion-level policies.  The brigade commander observes aggregate
sector states and selects which option to execute for each battalion.

The underlying :class:`~envs.multi_battalion_env.MultiBattalionEnv` battalion
policies are kept **frozen** — no gradient updates flow through them during
brigade training.

Usage::

    # Default config
    python training/train_brigade.py

    # Hydra CLI overrides
    python training/train_brigade.py training.total_timesteps=100000

    # With a pre-trained v2 MAPPO checkpoint for Red
    python training/train_brigade.py env.battalion_checkpoint=checkpoints/mappo_2v2/mappo_policy_final.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from envs.brigade_env import BrigadeEnv
from models.mappo_policy import MAPPOPolicy

log = logging.getLogger(__name__)

__all__ = ["BrigadeWinRateCallback", "load_frozen_battalion_policy", "train_brigade"]


# ---------------------------------------------------------------------------
# Frozen policy loader
# ---------------------------------------------------------------------------


def load_frozen_battalion_policy(
    checkpoint_path: Path,
    obs_dim: int,
    action_dim: int,
    state_dim: int,
    n_agents: int = 2,
    device: str = "cpu",
) -> MAPPOPolicy:
    """Load a v2 MAPPO checkpoint and return a **frozen** policy.

    All parameters of the returned policy have ``requires_grad=False`` so
    no gradients flow through it during brigade training.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``mappo_policy*.pt`` checkpoint saved by
        :mod:`training.train_mappo`.
    obs_dim:
        Per-agent observation dimensionality (must match the checkpoint).
    action_dim:
        Action dimensionality (must match).
    state_dim:
        Global state dimensionality (must match).
    n_agents:
        Number of agents the policy was trained with.
    device:
        PyTorch device string.

    Returns
    -------
    MAPPOPolicy
        Policy loaded from the checkpoint with all gradients frozen.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=n_agents,
        share_parameters=True,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])

    # Freeze all parameters
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()
    policy = policy.to(device)

    log.info(
        "Loaded frozen battalion policy from %s (step=%d)",
        checkpoint_path,
        checkpoint.get("total_steps", -1),
    )
    return policy


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------


class BrigadeWinRateCallback(BaseCallback):
    """Evaluate brigade win rate periodically and log to W&B.

    A *win* is defined as the episode ending with at least one Blue battalion
    alive and no Red battalions alive.

    Parameters
    ----------
    eval_env:
        A single :class:`~envs.brigade_env.BrigadeEnv` used for evaluation.
    eval_freq:
        Number of environment steps between evaluations.
    n_eval_episodes:
        Episodes per evaluation.
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        eval_env: BrigadeEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            win_rate = self._evaluate()
            if self.verbose >= 1:
                log.info(
                    "[brigade] step=%d win_rate=%.3f",
                    self.num_timesteps,
                    win_rate,
                )
            if wandb.run is not None:
                wandb.log(
                    {"brigade/win_rate": win_rate},
                    step=self.num_timesteps,
                )
            self._last_eval_step = self.num_timesteps
        return True

    def _evaluate(self) -> float:
        """Run evaluation episodes and return win rate."""
        wins = 0
        model = self.model
        for _ in range(self._n_eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            last_info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if done:
                    last_info = info
            # Use the winner signal set by BrigadeEnv when the episode ends
            winner = last_info.get("winner")
            if winner == "blue":
                wins += 1
        return wins / self._n_eval_episodes


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_brigade(
    n_blue: int = 2,
    n_red: int = 2,
    total_timesteps: int = 500_000,
    n_steps: int = 512,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
    checkpoint_dir: Optional[Path] = None,
    checkpoint_freq: int = 50_000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
    battalion_checkpoint: Optional[Path] = None,
    red_random: bool = False,
    max_steps: int = 500,
    map_width: float = 1000.0,
    map_height: float = 1000.0,
    randomize_terrain: bool = True,
    visibility_radius: float = 600.0,
    log_interval: int = 4,
    temporal_ratio: int = 10,
) -> PPO:
    """Train a brigade PPO agent.

    Parameters
    ----------
    n_blue:
        Number of Blue battalions.
    n_red:
        Number of Red opponent battalions.
    total_timesteps:
        Total macro-steps for training.
    n_steps:
        SB3 PPO rollout buffer size.
    n_epochs:
        Number of PPO update epochs per rollout.
    batch_size:
        Minibatch size for PPO updates.
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    gae_lambda:
        GAE smoothing parameter.
    clip_range:
        PPO ε clipping.
    ent_coef:
        Entropy coefficient.
    vf_coef:
        Value function loss coefficient.
    max_grad_norm:
        Gradient clipping norm.
    seed:
        Random seed.
    device:
        PyTorch device string.
    checkpoint_dir:
        Directory for saving PPO checkpoints.
    checkpoint_freq:
        Checkpoint save frequency in macro-steps.
    eval_freq:
        Win-rate evaluation frequency in macro-steps.
    n_eval_episodes:
        Episodes per win-rate evaluation.
    battalion_checkpoint:
        Optional path to a frozen v2 MAPPO checkpoint for driving Red.
    red_random:
        Use random Red actions (ignored when battalion_checkpoint is set).
    max_steps:
        Inner environment maximum primitive steps per episode.
    map_width:
        Map width in metres.
    map_height:
        Map height in metres.
    randomize_terrain:
        Randomize terrain each episode.
    visibility_radius:
        Fog-of-war visibility radius in metres.
    log_interval:
        SB3 logging interval (rollouts).
    temporal_ratio:
        Number of primitive battalion steps per brigade macro-step (option
        duration cap).  Passed to :class:`~envs.brigade_env.BrigadeEnv` and
        logged to W&B run config.  Use the sweep config
        ``configs/sweeps/temporal_ratio_sweep.yaml`` to search over values
        ``[5, 10, 20, 50]``.

    Returns
    -------
    PPO
        The trained SB3 PPO model.
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frozen battalion policy ──────────────────────────────────────
    frozen_policy: Optional[MAPPOPolicy] = None
    if battalion_checkpoint is not None:
        battalion_checkpoint = Path(battalion_checkpoint)
        if battalion_checkpoint.exists():
            # Determine dimensions from a temporary env
            _tmp_env = BrigadeEnv(
                n_blue=n_blue,
                n_red=n_red,
                max_steps=max_steps,
                map_width=map_width,
                map_height=map_height,
            )
            inner_obs_dim = _tmp_env._inner._obs_dim
            inner_act_dim = _tmp_env._inner._act_space.shape[0]
            inner_state_dim = _tmp_env._inner._state_dim
            _tmp_env.close()

            frozen_policy = load_frozen_battalion_policy(
                checkpoint_path=battalion_checkpoint,
                obs_dim=inner_obs_dim,
                action_dim=inner_act_dim,
                state_dim=inner_state_dim,
                n_agents=n_blue,
                device=device,
            )
        else:
            log.warning(
                "battalion_checkpoint not found: %s — Red will use fallback actions.",
                battalion_checkpoint,
            )

    # ── Verify frozen policy has no trainable parameters ─────────────────
    if frozen_policy is not None:
        trainable = sum(
            p.numel() for p in frozen_policy.parameters() if p.requires_grad
        )
        assert trainable == 0, (
            f"Expected 0 trainable params in frozen battalion policy, got {trainable}"
        )
        log.info("Battalion policy frozen: 0 trainable parameters confirmed.")

    # ── Environment factory ───────────────────────────────────────────────
    def _make_env():
        env = BrigadeEnv(
            n_blue=n_blue,
            n_red=n_red,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            battalion_policy=frozen_policy,
            red_random=red_random and frozen_policy is None,
            temporal_ratio=temporal_ratio,
        )
        return env

    # ── Training environment ──────────────────────────────────────────────
    train_env = DummyVecEnv([_make_env])

    # ── Evaluation environment ────────────────────────────────────────────
    eval_env = _make_env()

    # ── SB3 PPO model ─────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        device=device,
        verbose=1,
    )

    log.info(
        "BrigadeEnv: n_blue=%d n_red=%d obs_dim=%d n_options=%d temporal_ratio=%d",
        n_blue,
        n_red,
        eval_env._obs_dim,
        eval_env.n_options,
        temporal_ratio,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks: list[BaseCallback] = [
        BrigadeWinRateCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=1,
        ),
    ]

    if checkpoint_dir is not None:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="ppo_brigade",
                verbose=1,
            )
        )

    # ── Training ──────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        log_interval=log_interval,
        progress_bar=False,
    )

    # ── Save final model ──────────────────────────────────────────────────
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "ppo_brigade_final.zip"
        model.save(str(final_path))
        log.info("Brigade model saved → %s", final_path)

    eval_env.close()
    train_env.close()
    return model


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="experiment_brigade",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run brigade PPO training from a Hydra configuration."""
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    checkpoint_dir = _PROJECT_ROOT / cfg.eval.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")

    # ── Battalion checkpoint ──────────────────────────────────────────────
    battalion_ckpt_str = OmegaConf.select(cfg, "env.battalion_checkpoint", default=None)
    battalion_ckpt: Optional[Path] = None
    if battalion_ckpt_str:
        p = Path(battalion_ckpt_str)
        battalion_ckpt = p if p.is_absolute() else _PROJECT_ROOT / p

    # ── Train ─────────────────────────────────────────────────────────────
    temporal_ratio = int(OmegaConf.select(cfg, "env.temporal_ratio", default=10))
    train_brigade(
        n_blue=int(cfg.env.n_blue),
        n_red=int(cfg.env.n_red),
        total_timesteps=int(cfg.training.total_timesteps),
        n_steps=int(cfg.training.n_steps),
        n_epochs=int(cfg.training.n_epochs),
        batch_size=int(cfg.training.batch_size),
        lr=float(cfg.training.lr),
        gamma=float(cfg.training.gamma),
        gae_lambda=float(cfg.training.gae_lambda),
        clip_range=float(cfg.training.clip_range),
        ent_coef=float(cfg.training.ent_coef),
        vf_coef=float(cfg.training.vf_coef),
        max_grad_norm=float(cfg.training.max_grad_norm),
        seed=int(cfg.training.seed),
        device=str(OmegaConf.select(cfg, "training.device", default="cpu")),
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=int(cfg.eval.checkpoint_freq),
        eval_freq=int(OmegaConf.select(cfg, "eval.eval_freq", default=10_000)),
        n_eval_episodes=int(OmegaConf.select(cfg, "eval.n_eval_episodes", default=20)),
        battalion_checkpoint=battalion_ckpt,
        red_random=bool(OmegaConf.select(cfg, "env.red_random", default=False)),
        max_steps=int(cfg.env.max_steps),
        map_width=float(cfg.env.map_width),
        map_height=float(cfg.env.map_height),
        randomize_terrain=bool(cfg.env.randomize_terrain),
        visibility_radius=float(cfg.env.visibility_radius),
        log_interval=int(cfg.wandb.log_freq),
        temporal_ratio=temporal_ratio,
    )

    if run:
        run.finish()


if __name__ == "__main__":
    main()
