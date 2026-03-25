# SPDX-License-Identifier: MIT
# training/wfm1_multitask.py
"""WFM-1 multi-task training loop — E12.1.

Trains a :class:`~models.wfm1.WFM1Policy` simultaneously across multiple
echelon environments (battalion, brigade, division, corps) using PPO.  Each
update step samples a mini-batch from each active environment, computes a
joint PPO loss, and applies a single gradient update.

Architecture
------------
:class:`WFM1TrainConfig`
    Frozen configuration for a complete WFM-1 training run.

:class:`EchelonTaskConfig`
    Per-echelon environment and hyper-parameter configuration.

:class:`WFM1RolloutBuffer`
    Simple ring-buffer that collects (tokens, actions, rewards, …) for one
    echelon level and computes GAE advantages.

:class:`WFM1MultiTaskTrainer`
    Orchestrates rollout collection and PPO updates across all configured
    echelon tasks.

Typical usage (dry-run / CI)::

    from training.wfm1_multitask import WFM1MultiTaskTrainer, WFM1TrainConfig

    cfg = WFM1TrainConfig(total_steps=1000, n_steps=32, batch_size=16)
    trainer = WFM1MultiTaskTrainer(cfg)
    result = trainer.train()
    print(result)

With W&B logging::

    cfg = WFM1TrainConfig(
        total_steps=10_000_000,
        wandb_project="wargames-wfm1",
        wandb_run_name="wfm1-v1",
    )
    WFM1MultiTaskTrainer(cfg).train()
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.wfm1 import (
    ECHELON_BATTALION,
    ECHELON_BRIGADE,
    ECHELON_DIVISION,
    ECHELON_CORPS,
    ScenarioCard,
    WFM1Policy,
)
from models.entity_encoder import ENTITY_TOKEN_DIM

log = logging.getLogger(__name__)

__all__ = [
    "EchelonTaskConfig",
    "WFM1TrainConfig",
    "WFM1RolloutBuffer",
    "WFM1MultiTaskTrainer",
    "WFM1TrainResult",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EchelonTaskConfig:
    """Per-echelon task configuration.

    Attributes
    ----------
    echelon:
        Echelon level (0–3).
    env_id:
        Gymnasium environment identifier or factory key.
    task_weight:
        Relative loss weight for this echelon task in the joint PPO update.
    max_entities:
        Maximum number of entity tokens per observation (for buffer sizing).
    action_dim:
        Action space dimensionality for this echelon.
    """

    echelon: int = ECHELON_BATTALION
    env_id: str = "battalion"
    task_weight: float = 1.0
    max_entities: int = 32
    action_dim: int = 3


@dataclass
class WFM1TrainConfig:
    """Configuration for a WFM-1 multi-task training run.

    Attributes
    ----------
    total_steps:
        Total environment interactions across all echelon tasks.
    n_steps:
        Number of steps to collect per rollout before each PPO update.
    batch_size:
        PPO mini-batch size (number of transitions per gradient step).
    n_epochs:
        Number of PPO update epochs per rollout.
    learning_rate:
        Adam learning rate.
    gamma:
        Discount factor.
    gae_lambda:
        GAE lambda parameter.
    clip_coef:
        PPO clipping coefficient ε.
    vf_coef:
        Value-function loss coefficient.
    ent_coef:
        Entropy bonus coefficient.
    max_grad_norm:
        Gradient clipping norm.
    d_model:
        WFM-1 transformer hidden dimension.
    n_heads:
        Number of attention heads.
    n_echelon_layers:
        Echelon encoder transformer depth.
    n_cross_layers:
        Cross-echelon transformer depth.
    actor_hidden_sizes:
        Actor MLP hidden sizes.
    critic_hidden_sizes:
        Critic MLP hidden sizes.
    dropout:
        Transformer dropout probability.
    share_echelon_encoders:
        Share echelon encoder weights across all echelon levels.
    echelon_tasks:
        List of :class:`EchelonTaskConfig` objects (one per active echelon).
    checkpoint_dir:
        Directory for saving model checkpoints.
    checkpoint_interval:
        Save a checkpoint every *n* PPO updates.
    wandb_project:
        W&B project name.  Set to ``None`` to disable W&B logging.
    wandb_run_name:
        W&B run name.
    seed:
        Global RNG seed.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    """

    total_steps: int = 10_000_000
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    d_model: int = 128
    n_heads: int = 8
    n_echelon_layers: int = 4
    n_cross_layers: int = 2
    actor_hidden_sizes: Tuple[int, ...] = (256, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 128)
    dropout: float = 0.0
    share_echelon_encoders: bool = True
    card_hidden_size: int = 64
    echelon_tasks: List[EchelonTaskConfig] = field(
        default_factory=lambda: [EchelonTaskConfig(echelon=ECHELON_BATTALION)]
    )
    checkpoint_dir: str = "checkpoints/wfm1"
    checkpoint_interval: int = 100
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = "wfm1"
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# WFM1RolloutBuffer
# ---------------------------------------------------------------------------


class WFM1RolloutBuffer:
    """Simple per-echelon rollout buffer with GAE support.

    Stores transitions for one echelon task and computes generalised advantage
    estimates after a rollout completes.

    Parameters
    ----------
    n_steps:
        Maximum number of transitions to store (rollout length).
    max_entities:
        Maximum entity count per observation.
    token_dim:
        Entity token dimensionality.
    action_dim:
        Action dimensionality.
    gamma:
        Discount factor.
    gae_lambda:
        GAE λ parameter.
    """

    def __init__(
        self,
        n_steps: int,
        max_entities: int,
        token_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.max_entities = max_entities
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._reset()

    def _reset(self) -> None:
        T, N, A = self.n_steps, self.max_entities, self.action_dim
        self.tokens = np.zeros((T, N, self.token_dim), dtype=np.float32)
        self.pad_masks = np.zeros((T, N), dtype=bool)
        self.actions = np.zeros((T, A), dtype=np.float32)
        self.log_probs = np.zeros(T, dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.values = np.zeros(T, dtype=np.float32)
        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)
        self._ptr = 0
        self._full = False

    def reset(self) -> None:
        """Reset buffer before a new rollout."""
        self._reset()

    @property
    def is_full(self) -> bool:
        return self._full

    def add(
        self,
        tokens: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        pad_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store one environment transition."""
        if self._full:
            raise RuntimeError("Buffer is full — call reset() first.")
        t = self._ptr
        n_obs = min(tokens.shape[0], self.max_entities)
        self.tokens[t, :n_obs] = tokens[:n_obs]
        if pad_mask is not None:
            self.pad_masks[t, :n_obs] = pad_mask[:n_obs]
        self.pad_masks[t, n_obs:] = True
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = float(done)
        self.values[t] = value
        self._ptr += 1
        if self._ptr == self.n_steps:
            self._full = True

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool
    ) -> None:
        """Compute GAE advantages and discounted returns (in-place)."""
        gae = 0.0
        next_val = last_value
        for t in reversed(range(self.n_steps)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_val * not_done - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_val = self.values[t]

    def get_batches(
        self,
        batch_size: int,
        device: torch.device,
        normalize_advantages: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Return shuffled mini-batches of stored transitions.

        Parameters
        ----------
        batch_size:
            Number of transitions per mini-batch.
        device:
            Target device for tensors.
        normalize_advantages:
            Normalise advantages to zero mean, unit variance.

        Returns
        -------
        list of dicts with keys: tokens, pad_masks, actions, log_probs_old,
        advantages, returns, values.
        """
        adv = self.advantages.copy()
        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.random.permutation(self.n_steps)
        batches = []
        for start in range(0, self.n_steps, batch_size):
            sl = idx[start : start + batch_size]
            batches.append(
                {
                    "tokens": torch.as_tensor(
                        self.tokens[sl], dtype=torch.float32, device=device
                    ),
                    "pad_masks": torch.as_tensor(
                        self.pad_masks[sl], dtype=torch.bool, device=device
                    ),
                    "actions": torch.as_tensor(
                        self.actions[sl], dtype=torch.float32, device=device
                    ),
                    "log_probs_old": torch.as_tensor(
                        self.log_probs[sl], dtype=torch.float32, device=device
                    ),
                    "advantages": torch.as_tensor(
                        adv[sl], dtype=torch.float32, device=device
                    ),
                    "returns": torch.as_tensor(
                        self.returns[sl], dtype=torch.float32, device=device
                    ),
                    "values": torch.as_tensor(
                        self.values[sl], dtype=torch.float32, device=device
                    ),
                }
            )
        return batches


# ---------------------------------------------------------------------------
# WFM1TrainResult
# ---------------------------------------------------------------------------


@dataclass
class WFM1TrainResult:
    """Summary of a completed WFM-1 training run.

    Attributes
    ----------
    total_steps:
        Total environment steps taken.
    total_updates:
        Number of PPO update iterations completed.
    elapsed_seconds:
        Wall-clock training time.
    mean_reward_per_echelon:
        Mean episodic reward per echelon key at the end of training.
    final_checkpoint_path:
        Path to the final saved checkpoint (or ``None`` if not saved).
    """

    total_steps: int
    total_updates: int
    elapsed_seconds: float
    mean_reward_per_echelon: Dict[str, float]
    final_checkpoint_path: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            "WFM-1 Training Result",
            f"  Steps       : {self.total_steps:,}",
            f"  Updates     : {self.total_updates:,}",
            f"  Time        : {self.elapsed_seconds:.1f} s",
        ]
        for name, rew in self.mean_reward_per_echelon.items():
            lines.append(f"  Reward[{name:10s}]: {rew:.3f}")
        if self.final_checkpoint_path:
            lines.append(f"  Checkpoint  : {self.final_checkpoint_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WFM1MultiTaskTrainer
# ---------------------------------------------------------------------------


class WFM1MultiTaskTrainer:
    """Multi-task PPO trainer for WFM-1.

    Collects rollouts from each configured echelon environment and applies a
    joint PPO update.  When no real Gymnasium environments are available
    (e.g., in CI), a lightweight synthetic environment is used for each
    echelon.

    Parameters
    ----------
    config:
        Training configuration.
    policy:
        Optional pre-built :class:`~models.wfm1.WFM1Policy`.  When ``None``
        a fresh policy is instantiated from the config.
    """

    def __init__(
        self,
        config: WFM1TrainConfig,
        policy: Optional[WFM1Policy] = None,
    ) -> None:
        self.cfg = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Build policy
        if policy is not None:
            self.policy = policy.to(self.device)
        else:
            # Validate that all echelon tasks share a single action_dim.
            # WFM-1 uses a single actor head, so all tasks must agree on the
            # output dimensionality.
            task_dims = [t.action_dim for t in config.echelon_tasks]
            if len(set(task_dims)) > 1:
                raise ValueError(
                    f"All echelon tasks must share the same action_dim for a "
                    f"single-head WFM-1 policy, but got dims {task_dims}. "
                    f"Set every task's action_dim to the same value, or supply "
                    f"a pre-built policy with the desired action_dim."
                )
            shared_action_dim = task_dims[0] if task_dims else 3
            self.policy = WFM1Policy(
                token_dim=ENTITY_TOKEN_DIM,
                action_dim=shared_action_dim,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_echelon_layers=config.n_echelon_layers,
                n_cross_layers=config.n_cross_layers,
                actor_hidden_sizes=config.actor_hidden_sizes,
                critic_hidden_sizes=config.critic_hidden_sizes,
                dropout=config.dropout,
                share_echelon_encoders=config.share_echelon_encoders,
                card_hidden_size=config.card_hidden_size,
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

        # Build rollout buffers and environments
        self.buffers: Dict[int, WFM1RolloutBuffer] = {}
        self.envs: Dict[int, Any] = {}
        self._episode_rewards: Dict[int, List[float]] = {}
        self._ep_reward_acc: Dict[int, float] = {}

        for task in config.echelon_tasks:
            self.buffers[task.echelon] = WFM1RolloutBuffer(
                n_steps=config.n_steps,
                max_entities=task.max_entities,
                token_dim=ENTITY_TOKEN_DIM,
                action_dim=task.action_dim,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            self.envs[task.echelon] = self._make_env(task)
            self._episode_rewards[task.echelon] = []
            self._ep_reward_acc[task.echelon] = 0.0

        # W&B
        self._wandb_run = None
        if config.wandb_project:
            try:
                import wandb  # type: ignore

                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config={
                        "total_steps": config.total_steps,
                        "d_model": config.d_model,
                        "n_echelon_layers": config.n_echelon_layers,
                        "echelons": [t.echelon for t in config.echelon_tasks],
                    },
                )
            except ImportError:
                log.warning("wandb not installed; W&B logging disabled.")

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    def _make_env(self, task: EchelonTaskConfig) -> Any:
        """Create a Gymnasium-compatible environment for *task*.

        Falls back to a synthetic step-function if the real env cannot be
        imported, has an incompatible observation shape, or requires an
        integer (MultiDiscrete) action space that is incompatible with the
        continuous Gaussian actor (useful in CI / unit tests).
        """
        echelon = task.echelon

        # Try to import real environment
        try:
            import gymnasium as gym  # type: ignore

            if echelon == ECHELON_BATTALION:
                from envs.battalion_env import BattalionEnv

                env = BattalionEnv()
            elif echelon == ECHELON_BRIGADE:
                from envs.brigade_env import BrigadeEnv

                env = BrigadeEnv()
            elif echelon == ECHELON_DIVISION:
                from envs.division_env import DivisionEnv

                env = DivisionEnv()
            elif echelon == ECHELON_CORPS:
                from envs.corps_env import CorpsEnv

                env = CorpsEnv()
            else:
                env = None

            if env is None:
                raise ValueError("No env class for echelon.")

            # Only use real envs with continuous (Box) action spaces so the
            # Gaussian actor output is directly applicable.
            if not isinstance(env.action_space, gym.spaces.Box):
                env.close()
                raise ValueError("Real env has non-Box action space; using synthetic.")

            # Validate that the action dimensionality matches
            real_action_dim = env.action_space.shape[0] if env.action_space.shape else 0
            if real_action_dim != task.action_dim:
                env.close()
                raise ValueError(
                    f"Action dim mismatch: env={real_action_dim}, task={task.action_dim}."
                )

            return env
        except Exception as exc:
            log.debug(
                "Could not construct real env for echelon %d (task %r): %s — "
                "falling back to synthetic environment.",
                task.echelon,
                task.env_id,
                exc,
                exc_info=True,
            )

        return _SyntheticEnv(
            n_entities=task.max_entities,
            token_dim=ENTITY_TOKEN_DIM,
            action_dim=task.action_dim,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, echelon: int) -> None:
        """Fill the buffer for *echelon* by interacting with its env."""
        buf = self.buffers[echelon]
        env = self.envs[echelon]
        buf.reset()

        obs, _ = env.reset()
        tokens, pad_mask = self._obs_to_tokens(obs, echelon)

        for _ in range(self.cfg.n_steps):
            with torch.no_grad():
                t = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
                pm = (
                    torch.as_tensor(pad_mask, dtype=torch.bool, device=self.device)
                    if pad_mask is not None
                    else None
                )
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                    pm = pm.unsqueeze(0) if pm is not None else None
                actions, log_probs = self.policy.act(
                    t, pad_mask=pm, echelon=echelon, deterministic=False
                )
                value = self.policy.get_value(t, pad_mask=pm, echelon=echelon)

            action_np = actions.squeeze(0).cpu().numpy()
            lp = log_probs.squeeze(0).item() if log_probs.dim() > 0 else log_probs.item()
            val = value.squeeze(0).item() if value.dim() > 0 else value.item()

            obs_new, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            self._ep_reward_acc[echelon] += float(reward)

            buf.add(
                tokens=tokens,
                action=action_np,
                log_prob=lp,
                reward=float(reward),
                done=done,
                value=val,
                pad_mask=pad_mask,
            )

            if done:
                self._episode_rewards[echelon].append(self._ep_reward_acc[echelon])
                self._ep_reward_acc[echelon] = 0.0
                obs, _ = env.reset()
                tokens, pad_mask = self._obs_to_tokens(obs, echelon)
            else:
                obs = obs_new
                tokens, pad_mask = self._obs_to_tokens(obs, echelon)

        # Bootstrap last value
        with torch.no_grad():
            t = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
            pm = (
                torch.as_tensor(pad_mask, dtype=torch.bool, device=self.device)
                if pad_mask is not None
                else None
            )
            if t.dim() == 2:
                t = t.unsqueeze(0)
                pm = pm.unsqueeze(0) if pm is not None else None
            last_val = self.policy.get_value(t, pad_mask=pm, echelon=echelon)
        lv = last_val.squeeze().item()
        buf.compute_returns_and_advantages(last_value=lv, last_done=False)

    def _obs_to_tokens(
        self, obs: Any, echelon: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert a raw environment observation to entity tokens + pad mask.

        For real envs this would call the environment's obs-to-token converter.
        Here we use a simple heuristic: if obs is already a 2-D array
        ``(N, token_dim)`` we use it directly; otherwise we reshape.
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 2 and obs.shape[-1] == ENTITY_TOKEN_DIM:
                return obs, None
            # Flat vector — reshape to (n_entities, token_dim) zero-padding as needed
            n_full = (obs.size + ENTITY_TOKEN_DIM - 1) // ENTITY_TOKEN_DIM
            padded = np.zeros(n_full * ENTITY_TOKEN_DIM, dtype=np.float32)
            padded[: obs.size] = obs.ravel()
            tokens = padded.reshape(n_full, ENTITY_TOKEN_DIM)
            return tokens, None
        # Fallback — zero tokens sized by task config
        task = next(
            (t for t in self.cfg.echelon_tasks if t.echelon == echelon), None
        )
        max_entities = task.max_entities if task is not None else 8
        return np.zeros((max_entities, ENTITY_TOKEN_DIM), dtype=np.float32), None

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> Dict[str, float]:
        """Run PPO update over all echelon buffers.

        Returns a dict of scalar loss metrics.
        """
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        for epoch in range(self.cfg.n_epochs):
            for task in self.cfg.echelon_tasks:
                echelon = task.echelon
                buf = self.buffers[echelon]
                batches = buf.get_batches(
                    batch_size=self.cfg.batch_size,
                    device=self.device,
                    normalize_advantages=True,
                )
                for batch in batches:
                    log_probs, entropy, values = self.policy.evaluate_actions(
                        tokens=batch["tokens"],
                        actions=batch["actions"],
                        pad_mask=batch["pad_masks"],
                        echelon=echelon,
                    )

                    ratio = torch.exp(log_probs - batch["log_probs_old"])
                    adv = batch["advantages"]

                    # PPO clipped surrogate loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    vf_loss = nn.functional.mse_loss(values, batch["returns"])

                    # Entropy bonus
                    ent_loss = -entropy.mean()

                    loss = (
                        task.task_weight
                        * (
                            pg_loss
                            + self.cfg.vf_coef * vf_loss
                            + self.cfg.ent_coef * ent_loss
                        )
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_vf_loss += vf_loss.item()
                    total_ent += (-ent_loss).item()
                    n_updates += 1

        denom = max(n_updates, 1)
        return {
            "pg_loss": total_pg_loss / denom,
            "vf_loss": total_vf_loss / denom,
            "entropy": total_ent / denom,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> WFM1TrainResult:
        """Run the full multi-task training loop.

        Returns
        -------
        :class:`WFM1TrainResult`
        """
        cfg = self.cfg
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        steps_per_update = cfg.n_steps * len(cfg.echelon_tasks)
        n_updates = cfg.total_steps // max(steps_per_update, 1)

        log.info(
            "WFM-1 training: %d updates × %d steps/update = %d total steps",
            n_updates,
            steps_per_update,
            n_updates * steps_per_update,
        )

        t_start = time.perf_counter()
        global_step = 0

        for update_idx in range(1, n_updates + 1):
            # Collect rollouts for all echelons
            for task in cfg.echelon_tasks:
                self._collect_rollout(task.echelon)
                global_step += cfg.n_steps

            # PPO update
            metrics = self._ppo_update()

            # Logging
            if update_idx % 10 == 0 or update_idx == 1:
                mean_rews = {
                    str(t.echelon): (
                        float(np.mean(self._episode_rewards[t.echelon][-10:]))
                        if self._episode_rewards[t.echelon]
                        else 0.0
                    )
                    for t in cfg.echelon_tasks
                }
                log.info(
                    "update=%d step=%d pg=%.4f vf=%.4f ent=%.4f rew=%s",
                    update_idx,
                    global_step,
                    metrics["pg_loss"],
                    metrics["vf_loss"],
                    metrics["entropy"],
                    mean_rews,
                )
                if self._wandb_run is not None:
                    try:
                        import wandb  # type: ignore

                        wandb.log(
                            {
                                "step": global_step,
                                "pg_loss": metrics["pg_loss"],
                                "vf_loss": metrics["vf_loss"],
                                "entropy": metrics["entropy"],
                                **{
                                    f"reward/{k}": v for k, v in mean_rews.items()
                                },
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

            # Checkpointing
            if update_idx % cfg.checkpoint_interval == 0 or update_idx == n_updates:
                ckpt_path = ckpt_dir / f"wfm1_step{global_step}.pt"
                self.policy.save_checkpoint(ckpt_path)
                log.info("Checkpoint saved: %s", ckpt_path)

        # Final result
        elapsed = time.perf_counter() - t_start
        mean_rews_final = {
            str(t.echelon): (
                float(np.mean(self._episode_rewards[t.echelon][-100:]))
                if self._episode_rewards[t.echelon]
                else 0.0
            )
            for t in cfg.echelon_tasks
        }
        final_ckpt = str(ckpt_dir / f"wfm1_step{global_step}.pt")

        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.finish()
            except Exception:
                pass

        return WFM1TrainResult(
            total_steps=global_step,
            total_updates=n_updates,
            elapsed_seconds=elapsed,
            mean_reward_per_echelon=mean_rews_final,
            final_checkpoint_path=final_ckpt,
        )


# ---------------------------------------------------------------------------
# Synthetic environment (fallback for CI)
# ---------------------------------------------------------------------------


class _SyntheticEnv:
    """Minimal synthetic environment for testing/CI that produces entity tokens.

    Implements the Gymnasium ``reset`` / ``step`` interface with random
    observations and rewards.
    """

    def __init__(
        self,
        n_entities: int = 8,
        token_dim: int = ENTITY_TOKEN_DIM,
        action_dim: int = 3,
        ep_length: int = 64,
        seed: int = 0,
    ) -> None:
        self.n_entities = n_entities
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.ep_length = ep_length
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        obs = self._rng.standard_normal((self.n_entities, self.token_dim)).astype(
            np.float32
        )
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        obs = self._rng.standard_normal((self.n_entities, self.token_dim)).astype(
            np.float32
        )
        reward = float(self._rng.standard_normal())
        terminated = self._step_count >= self.ep_length
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        pass
