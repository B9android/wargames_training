"""Division Commander training — PPO on the :class:`~envs.division_env.DivisionEnv`.

Trains a division-level RL agent that issues operational commands to
frozen brigade-level policies.  The division commander observes aggregate
theatre-sector states and selects which operational command to execute for
each brigade.

The underlying :class:`~envs.brigade_env.BrigadeEnv` is held fixed during
division training — no gradient updates flow through it.  Optionally a
pre-trained brigade PPO checkpoint can be loaded to drive Red brigades as a
challenging, frozen adversary.

Usage::

    # Default config
    python training/train_division.py

    # Hydra CLI overrides
    python training/train_division.py training.total_timesteps=1000000

    # With a frozen brigade checkpoint for Red
    python training/train_division.py \\
        env.brigade_checkpoint=checkpoints/brigade/ppo_brigade_final.zip
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from envs.division_env import DivisionEnv

log = logging.getLogger(__name__)

__all__ = [
    "DivisionWinRateCallback",
    "load_frozen_brigade_policy",
    "train_division",
]


# ---------------------------------------------------------------------------
# Frozen brigade policy loader
# ---------------------------------------------------------------------------


def load_frozen_brigade_policy(checkpoint_path: Path, device: str = "cpu") -> PPO:
    """Load an SB3 PPO brigade checkpoint and return a **frozen** policy.

    The loaded policy is intended to drive Red battalions as a challenging
    opponent inside :class:`~envs.division_env.DivisionEnv`.  It must have
    been trained on a :class:`~envs.brigade_env.BrigadeEnv` whose Red-side
    battalion count matches ``n_red`` of the target ``DivisionEnv``.

    At each division step the policy receives a
    :class:`~envs.brigade_env.BrigadeEnv`-compatible observation for the
    Red side of shape ``(3 + 7 * n_red + 1,)`` (treating Red battalions as
    the "blue" side) and returns a per-battalion action of shape ``(n_red,)``
    with option indices in ``[0, n_options)``.

    The PPO model's internal policy network parameters are frozen with
    ``requires_grad=False`` so no gradients flow through the policy during
    division training.

    Parameters
    ----------
    checkpoint_path:
        Path to an SB3 PPO ``.zip`` checkpoint saved by
        :mod:`training.train_brigade`.  The checkpoint must have been
        trained with the same ``n_blue`` (= Red-side ``n_red`` here).
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    PPO
        Loaded SB3 PPO model with frozen policy parameters.
    """
    model = PPO.load(str(checkpoint_path), device=device)

    # Freeze all policy network parameters
    for param in model.policy.parameters():
        param.requires_grad_(False)
    model.policy.eval()

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    assert trainable == 0, (
        f"Expected 0 trainable params in frozen brigade policy, got {trainable}"
    )
    log.info("Loaded frozen brigade policy from %s (all params frozen).", checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------


class DivisionWinRateCallback(BaseCallback):
    """Evaluate division win rate periodically and log to W&B.

    A *win* is defined as the episode ending with at least one Blue battalion
    alive and no Red battalions alive (``info["winner"] == "blue"``).

    Parameters
    ----------
    eval_env:
        A single :class:`~envs.division_env.DivisionEnv` used for evaluation.
    eval_freq:
        Number of environment steps between evaluations.
    n_eval_episodes:
        Episodes per evaluation run.
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        eval_env: DivisionEnv,
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
                    "[division] step=%d win_rate=%.3f",
                    self.num_timesteps,
                    win_rate,
                )
            if wandb.run is not None:
                wandb.log(
                    {"division/win_rate": win_rate},
                    step=self.num_timesteps,
                )
            self._last_eval_step = self.num_timesteps
        return True

    def _evaluate(self) -> float:
        """Run evaluation episodes and return the win rate."""
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
            if last_info.get("winner") == "blue":
                wins += 1
        return wins / self._n_eval_episodes


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_division(
    n_brigades: int = 2,
    n_blue_per_brigade: int = 2,
    n_red_brigades: int = 2,
    n_red_per_brigade: int = 2,
    total_timesteps: int = 1_000_000,
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
    checkpoint_freq: int = 100_000,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 20,
    brigade_checkpoint: Optional[Path] = None,
    red_random: bool = False,
    max_steps: int = 500,
    map_width: float = 1000.0,
    map_height: float = 1000.0,
    randomize_terrain: bool = True,
    visibility_radius: float = 600.0,
    log_interval: int = 4,
) -> PPO:
    """Train a division PPO agent.

    Parameters
    ----------
    n_brigades:
        Number of Blue brigades (each containing *n_blue_per_brigade* battalions).
    n_blue_per_brigade:
        Blue battalions per brigade.
    n_red_brigades:
        Number of Red brigades.
    n_red_per_brigade:
        Red battalions per Red brigade.
    total_timesteps:
        Total division macro-steps for training.
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
        Checkpoint save frequency in division macro-steps.
    eval_freq:
        Win-rate evaluation frequency in macro-steps.
    n_eval_episodes:
        Episodes per win-rate evaluation.
    brigade_checkpoint:
        Optional path to a frozen SB3 PPO brigade ``.zip`` checkpoint for
        driving Red brigades.  When ``None``, Red battalions are stationary
        (or random when *red_random* is ``True``).
    red_random:
        Use random Red battalion actions when no *brigade_checkpoint* is set.
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

    Returns
    -------
    PPO
        The trained SB3 PPO model.
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frozen brigade policy for Red ────────────────────────────
    frozen_brigade: Optional[PPO] = None
    if brigade_checkpoint is not None:
        brigade_checkpoint = Path(brigade_checkpoint)
        if brigade_checkpoint.exists():
            frozen_brigade = load_frozen_brigade_policy(brigade_checkpoint, device=device)
        else:
            log.warning(
                "brigade_checkpoint not found: %s — Red will use fallback actions.",
                brigade_checkpoint,
            )

    # ── Environment factory ───────────────────────────────────────────
    def _make_env():
        env = DivisionEnv(
            n_brigades=n_brigades,
            n_blue_per_brigade=n_blue_per_brigade,
            n_red_brigades=n_red_brigades,
            n_red_per_brigade=n_red_per_brigade,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            brigade_policy=frozen_brigade,
            # red_random is only effective when no frozen brigade policy is set;
            # when frozen_brigade is provided it drives Red, so red_random is ignored.
            red_random=red_random and frozen_brigade is None,
        )
        return env

    # ── Training and evaluation environments ─────────────────────────
    train_env = DummyVecEnv([_make_env])
    eval_env = _make_env()

    # ── SB3 PPO model ─────────────────────────────────────────────────
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
        "DivisionEnv: n_brigades=%d n_blue_per_brigade=%d n_red_brigades=%d "
        "n_red_per_brigade=%d obs_dim=%d n_div_options=%d",
        n_brigades,
        n_blue_per_brigade,
        n_red_brigades,
        n_red_per_brigade,
        eval_env._obs_dim,
        eval_env.n_div_options,
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks: list[BaseCallback] = [
        DivisionWinRateCallback(
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
                name_prefix="ppo_division",
                verbose=1,
            )
        )

    # ── Training ──────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        log_interval=log_interval,
        progress_bar=False,
    )

    # ── Save final model ──────────────────────────────────────────────
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "ppo_division_final.zip"
        model.save(str(final_path))
        log.info("Division model saved → %s", final_path)

    eval_env.close()
    train_env.close()
    return model


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="experiment_division",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run division PPO training from a Hydra configuration."""
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    checkpoint_dir = _PROJECT_ROOT / cfg.eval.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")

    # ── Brigade checkpoint ────────────────────────────────────────────
    brigade_ckpt_str = OmegaConf.select(cfg, "env.brigade_checkpoint", default=None)
    brigade_ckpt: Optional[Path] = None
    if brigade_ckpt_str:
        p = Path(brigade_ckpt_str)
        brigade_ckpt = p if p.is_absolute() else _PROJECT_ROOT / p

    # ── Train ─────────────────────────────────────────────────────────
    train_division(
        n_brigades=int(cfg.env.n_brigades),
        n_blue_per_brigade=int(cfg.env.n_blue_per_brigade),
        n_red_brigades=int(cfg.env.n_red_brigades),
        n_red_per_brigade=int(cfg.env.n_red_per_brigade),
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
        eval_freq=int(OmegaConf.select(cfg, "eval.eval_freq", default=20_000)),
        n_eval_episodes=int(OmegaConf.select(cfg, "eval.n_eval_episodes", default=20)),
        brigade_checkpoint=brigade_ckpt,
        red_random=bool(OmegaConf.select(cfg, "env.red_random", default=False)),
        max_steps=int(cfg.env.max_steps),
        map_width=float(cfg.env.map_width),
        map_height=float(cfg.env.map_height),
        randomize_terrain=bool(cfg.env.randomize_terrain),
        visibility_radius=float(cfg.env.visibility_radius),
        log_interval=int(cfg.wandb.log_freq),
    )

    if run:
        run.finish()


if __name__ == "__main__":
    main()
