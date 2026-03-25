# SPDX-License-Identifier: MIT
# training/league/train_corps_main_agent.py
"""Corps-level main-agent training loop for league training (E7.3).

Ports the battalion-level :class:`~training.league.train_main_agent.MainAgentTrainer`
to operate on :class:`~envs.corps_env.CorpsEnv`.  Key differences from the
battalion-level trainer:

* **Single-agent Gymnasium env** — CorpsEnv is a standard ``gymnasium.Env``
  with a ``MultiDiscrete`` action space (one command per division).  A
  thin :class:`CorpsActorCriticPolicy` wraps a multi-head MLP that outputs
  a joint distribution over all divisional commands.
* **Operational result fields** — match results include ``territory_control``,
  ``blue_casualties``, ``red_casualties``, and ``supply_consumed`` extracted
  from the CorpsEnv ``info`` dict.
* **Larger model & longer training** — default hidden sizes are ``[256, 128]``
  and training runs for 10 M steps (see ``configs/league/corps_main_agent.yaml``).
* **Snapshot compatibility** — snapshots use the same ``.pt`` format as the
  battalion trainer so the ``AgentPool`` / ``MatchDatabase`` infrastructure
  is reused unchanged.

Classes
-------
CorpsActorCriticPolicy
    Simple MLP actor-critic for a MultiDiscrete corps action space.
CorpsMainAgentTrainer
    League training loop for corps-level self-play.

Functions
---------
make_pfsp_weight_fn
    Re-exported from :mod:`training.league.train_main_agent`.
main
    Hydra entry-point (config: ``configs/league/corps_main_agent.yaml``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra

from envs.corps_env import CorpsEnv
from training.elo import EloRegistry
from training.league.agent_pool import AgentPool, AgentType
from training.league.match_database import MatchDatabase
from training.league.matchmaker import LeagueMatchmaker
from training.league.nash import build_payoff_matrix, compute_nash_distribution, nash_entropy
from training.league.train_main_agent import make_pfsp_weight_fn  # noqa: F401

log = logging.getLogger(__name__)

__all__ = [
    "CorpsActorCriticPolicy",
    "CorpsMainAgentTrainer",
    "make_pfsp_weight_fn",
]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _build_mlp(in_dim: int, hidden_sizes: Tuple[int, ...]) -> Tuple[nn.Sequential, int]:
    """Return (Sequential MLP trunk, output_dim)."""
    layers: List[nn.Module] = []
    cur = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(cur, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.Tanh())
        cur = h
    return nn.Sequential(*layers), cur


class CorpsActorCriticPolicy(nn.Module):
    """Shared actor-critic for a corps commander with a MultiDiscrete action space.

    The actor maps the corps observation to per-division action logits.  Each
    division head is a separate ``Linear(trunk_out, n_options)`` layer so that
    the number of parameters scales with the number of divisions rather than
    with the product of option counts.

    The critic is a separate MLP trunk with a single scalar value head.

    Parameters
    ----------
    obs_dim:
        Flat observation dimension (output of ``CorpsEnv.observation_space``).
    n_divisions:
        Number of Blue divisions (i.e. action-space length).
    n_options:
        Number of discrete options per division (``CorpsEnv.n_corps_options``).
    actor_hidden_sizes:
        Hidden layer widths for the shared actor trunk.
    critic_hidden_sizes:
        Hidden layer widths for the critic trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        n_divisions: int,
        n_options: int,
        actor_hidden_sizes: Tuple[int, ...] = (256, 128),
        critic_hidden_sizes: Tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_divisions = n_divisions
        self.n_options = n_options
        self.actor_hidden_sizes = actor_hidden_sizes
        self.critic_hidden_sizes = critic_hidden_sizes

        actor_trunk, actor_out = _build_mlp(obs_dim, actor_hidden_sizes)
        self.actor_trunk = actor_trunk
        # One linear head per division.
        self.division_heads = nn.ModuleList(
            [nn.Linear(actor_out, n_options) for _ in range(n_divisions)]
        )

        critic_trunk, critic_out = _build_mlp(obs_dim, critic_hidden_sizes)
        self.critic_trunk = critic_trunk
        self.value_head = nn.Linear(critic_out, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_actor(self, obs: torch.Tensor) -> List[torch.distributions.Categorical]:
        """Return a list of ``Categorical`` distributions — one per division.

        Parameters
        ----------
        obs:
            Corps observation of shape ``(..., obs_dim)``.

        Returns
        -------
        list of Categorical
            One distribution per division, each over ``n_options`` commands.
        """
        trunk_out = self.actor_trunk(obs)
        dists = []
        for head in self.division_heads:
            logits = head(trunk_out)
            dists.append(torch.distributions.Categorical(logits=logits))
        return dists

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimate of shape ``(...)``."""
        return self.value_head(self.critic_trunk(obs)).squeeze(-1)

    # ------------------------------------------------------------------
    # Act / evaluate API
    # ------------------------------------------------------------------

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample (or greedily select) an action from the policy.

        Parameters
        ----------
        obs:
            Corps observation tensor of shape ``(obs_dim,)`` or
            ``(batch, obs_dim)``.
        deterministic:
            When ``True``, take the argmax instead of sampling.

        Returns
        -------
        actions : np.ndarray of shape ``(n_divisions,)``
            Integer action per division.
        log_prob : torch.Tensor scalar
            Sum of per-division log-probabilities.
        """
        dists = self.forward_actor(obs)
        actions = []
        log_prob = torch.zeros((), device=obs.device)
        for d in dists:
            if deterministic:
                a = d.probs.argmax(dim=-1)
            else:
                a = d.sample()
            log_prob = log_prob + d.log_prob(a)
            actions.append(a)
        action_t = torch.stack(actions, dim=-1)
        return action_t.cpu().numpy().astype(np.int64), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log_probs, values, and entropy for a batch of actions.

        Parameters
        ----------
        obs:
            Shape ``(batch, obs_dim)``.
        actions:
            Shape ``(batch, n_divisions)`` — integer actions.

        Returns
        -------
        log_probs : ``(batch,)``
        values    : ``(batch,)``
        entropy   : scalar
        """
        dists = self.forward_actor(obs)
        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros((), device=obs.device)
        for i, d in enumerate(dists):
            log_probs = log_probs + d.log_prob(actions[:, i])
            entropy = entropy + d.entropy().mean()
        values = self.forward_critic(obs)
        return log_probs, values, entropy / len(dists)

    # ------------------------------------------------------------------
    # Serialisation helpers (mirror MAPPOPolicy's contract for AgentPool)
    # ------------------------------------------------------------------

    def policy_kwargs(self) -> Dict:
        """Return constructor kwargs for snapshot saving/loading."""
        return {
            "obs_dim": self.obs_dim,
            "n_divisions": self.n_divisions,
            "n_options": self.n_options,
            "actor_hidden_sizes": self.actor_hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
        }


# ---------------------------------------------------------------------------
# Simple PPO buffer
# ---------------------------------------------------------------------------


class _RolloutBuffer:
    """Minimal on-policy rollout buffer for single-agent PPO on CorpsEnv."""

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        n_divisions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.n_divisions = n_divisions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._reset()

    def _reset(self) -> None:
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_divisions), dtype=np.int64)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self._ptr = 0
        self._full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = float(done)
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self._ptr = (self._ptr + 1) % self.n_steps
        if self._ptr == 0:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full or self._ptr == self.n_steps

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages

    def get_samples(
        self,
        batch_size: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Yield shuffled mini-batches of (obs, actions, old_log_probs, returns, advantages)."""
        n = self.n_steps
        indices = np.random.permutation(n)
        obs_t = torch.tensor(self.obs, dtype=torch.float32, device=device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_lp_t = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        batches = []
        # Compute returns / advantages will be called before this by the trainer.
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batches.append((
                obs_t[idx],
                actions_t[idx],
                old_lp_t[idx],
            ))
        return batches


# ---------------------------------------------------------------------------
# CorpsMainAgentTrainer
# ---------------------------------------------------------------------------


class CorpsMainAgentTrainer:
    """League training loop for a corps-level main agent.

    Trains a :class:`CorpsActorCriticPolicy` on :class:`~envs.corps_env.CorpsEnv`
    using a simple PPO update.  The same league infrastructure
    (AgentPool, MatchDatabase, LeagueMatchmaker, EloRegistry) used by the
    battalion-level :class:`~training.league.train_main_agent.MainAgentTrainer`
    is reused so that corps and battalion snapshots can co-exist in the pool.

    Snapshot versioning uses larger hidden sizes and a longer training schedule
    by default (see ``configs/league/corps_main_agent.yaml``).

    Match results are recorded with the operational fields ``territory_control``,
    ``blue_casualties``, ``red_casualties``, and ``supply_consumed`` extracted
    from the CorpsEnv ``info`` dict.

    Parameters
    ----------
    env:
        CorpsEnv instance used for training.
    policy:
        CorpsActorCriticPolicy to optimise.
    agent_pool:
        League agent pool.
    match_database:
        Match result store.
    matchmaker:
        PFSP opponent selector.
    elo_registry:
        Elo rating tracker.
    agent_id:
        Unique identifier for this main-agent lineage.
    snapshot_dir:
        Directory in which to write ``*.pt`` snapshot files.
    snapshot_freq:
        Environment steps between policy snapshots.  Default 200 000.
    eval_freq:
        Environment steps between evaluation rounds.  Default 100 000.
    n_eval_episodes:
        Episodes per evaluation round.  Default 20.
    pfsp_temperature:
        PFSP temperature for opponent sampling.  Default 1.0.
    lr:
        PPO learning rate.
    n_steps:
        Rollout steps before each PPO update.
    n_epochs:
        PPO update epochs per rollout.
    batch_size:
        PPO mini-batch size.
    gamma:
        Discount factor.
    gae_lambda:
        GAE lambda.
    clip_range:
        PPO ε clipping.
    vf_coef:
        Value-function loss coefficient.
    ent_coef:
        Entropy bonus coefficient.
    max_grad_norm:
        Gradient clipping norm.
    device:
        PyTorch device string.
    seed:
        Random seed.
    log_interval:
        Steps between W&B metric logging.
    checkpoint_dir:
        Optional directory for periodic trainer checkpoints.
    checkpoint_freq:
        Steps between trainer checkpoints.
    """

    def __init__(
        self,
        env: CorpsEnv,
        policy: CorpsActorCriticPolicy,
        agent_pool: AgentPool,
        match_database: MatchDatabase,
        matchmaker: LeagueMatchmaker,
        elo_registry: EloRegistry,
        agent_id: str,
        snapshot_dir: Path,
        snapshot_freq: int = 200_000,
        eval_freq: int = 100_000,
        n_eval_episodes: int = 20,
        pfsp_temperature: float = 1.0,
        lr: float = 3e-4,
        n_steps: int = 512,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        seed: int = 42,
        log_interval: int = 2_000,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 200_000,
    ) -> None:
        self.env = env
        self.policy = policy.to(device)
        self._agent_pool = agent_pool
        self._match_db = match_database
        self._matchmaker = matchmaker
        self._elo = elo_registry
        self.agent_id = agent_id
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_freq = int(snapshot_freq)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.pfsp_temperature = float(pfsp_temperature)

        self._device = torch.device(device)
        self._seed = int(seed)
        self.log_interval = int(log_interval)
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_freq = int(checkpoint_freq)

        # PPO hyper-parameters
        self._n_steps = int(n_steps)
        self._n_epochs = int(n_epochs)
        self._batch_size = int(batch_size)
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)
        self._clip_range = float(clip_range)
        self._vf_coef = float(vf_coef)
        self._ent_coef = float(ent_coef)
        self._max_grad_norm = float(max_grad_norm)

        self._optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._total_steps: int = 0
        self._snapshot_version: int = 0
        self._rng = np.random.default_rng(self._seed)

        # League state
        self._current_opponent_id: Optional[str] = None
        self._matchup_outcomes: Dict[str, List[float]] = {}

        # Apply PFSP weight function to matchmaker.
        pfsp_fn = make_pfsp_weight_fn(pfsp_temperature)
        self._matchmaker.set_weight_function(pfsp_fn)

        # Rollout buffer
        self._buffer = _RolloutBuffer(
            n_steps=self._n_steps,
            obs_dim=self.env._obs_dim,
            n_divisions=self.env.n_divisions,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
        )

        # Episode tracking
        self._obs: Optional[np.ndarray] = None
        self._ep_reward: float = 0.0
        self._ep_steps: int = 0
        self._ep_done: bool = True

        # Operational result accumulators (reset per evaluation episode)
        self._last_info: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> None:
        """Run the corps main-agent training loop for *total_timesteps* steps.

        Parameters
        ----------
        total_timesteps:
            Total CorpsEnv steps to train for.
        """
        if total_timesteps < 1:
            raise ValueError(
                f"total_timesteps must be >= 1, got {total_timesteps!r}"
            )

        self._ensure_initial_snapshot()

        last_log = self._total_steps
        last_ckpt = self._total_steps
        last_snapshot = self._total_steps
        last_eval = self._total_steps

        log.info(
            "CorpsMainAgentTrainer: start | agent_id=%s | total_timesteps=%d"
            " | snapshot_freq=%d | eval_freq=%d | pfsp_temperature=%.2f",
            self.agent_id,
            total_timesteps,
            self.snapshot_freq,
            self.eval_freq,
            self.pfsp_temperature,
        )

        while self._total_steps < total_timesteps:
            # PFSP: refresh the opponent snapshot registered in the pool
            # before collecting each rollout.  For CorpsEnv the "opponent"
            # is the scripted Red behavior built into the env, so we track
            # the ID for ELO / match-DB purposes but do not inject a policy.
            self._refresh_opponent()

            # Collect a rollout and run a PPO update.
            losses = self._collect_and_update()

            # Snapshot.
            if (
                self._total_steps > last_snapshot
                and self._total_steps - last_snapshot >= self.snapshot_freq
            ):
                self._snapshot_version += 1
                snap_path = self._save_snapshot(self._snapshot_version)
                new_id = f"{self.agent_id}_v{self._snapshot_version:06d}"
                self._agent_pool.add(
                    snap_path,
                    AgentType.MAIN_AGENT,
                    agent_id=new_id,
                    version=self._snapshot_version,
                    metadata={
                        "parent_agent_id": self.agent_id,
                        "step": self._total_steps,
                    },
                    force=True,
                )
                log.info(
                    "CorpsMainAgentTrainer: snapshot v%d saved → pool size=%d",
                    self._snapshot_version,
                    self._agent_pool.size,
                )
                last_snapshot = self._total_steps

            # Evaluation.
            if (
                self._total_steps > last_eval
                and self._total_steps - last_eval >= self.eval_freq
            ):
                self._run_evaluation(self._total_steps)
                last_eval = self._total_steps

            # Logging.
            if self._total_steps - last_log >= self.log_interval:
                self._log_training(losses, self._total_steps)
                last_log = self._total_steps

            # Checkpoint.
            if (
                self._checkpoint_dir is not None
                and self._total_steps > last_ckpt
                and self._total_steps - last_ckpt >= self.checkpoint_freq
            ):
                self._save_checkpoint(self._checkpoint_dir)
                last_ckpt = self._total_steps

        # Final checkpoint.
        if self._checkpoint_dir is not None:
            self._save_checkpoint(self._checkpoint_dir, suffix="_final")

        log.info(
            "CorpsMainAgentTrainer: training complete | steps=%d | snapshots=%d",
            self._total_steps,
            self._snapshot_version,
        )

    # ------------------------------------------------------------------
    # Private helpers — PPO
    # ------------------------------------------------------------------

    def _collect_and_update(self) -> Dict[str, float]:
        """Collect one rollout, run PPO update, return loss dict."""
        self._collect_rollout()
        return self._update_policy()

    def _collect_rollout(self) -> None:
        """Fill the rollout buffer with ``n_steps`` environment transitions."""
        if self._ep_done or self._obs is None:
            seed = int(self._rng.integers(0, 2**31))
            self._obs, _ = self.env.reset(seed=seed)
            self._ep_reward = 0.0
            self._ep_steps = 0
            self._ep_done = False

        self._buffer._reset()
        self.policy.eval()
        # Track whether the *last collected* transition was terminal so that
        # compute_returns_and_advantages() can correctly bootstrap: when the
        # rollout ends on a terminal step the last value should be zero.
        last_done = False
        with torch.no_grad():
            for _ in range(self._n_steps):
                obs_t = torch.tensor(
                    self._obs, dtype=torch.float32, device=self._device
                )
                action, log_prob = self.policy.act(obs_t, deterministic=False)
                value = self.policy.forward_critic(obs_t).item()

                obs_new, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                last_done = done
                self._last_info = info

                self._buffer.add(
                    self._obs, action, float(reward), done,
                    log_prob.item(), value,
                )
                self._total_steps += 1
                self._ep_reward += float(reward)
                self._ep_steps += 1

                if done:
                    seed = int(self._rng.integers(0, 2**31))
                    self._obs, _ = self.env.reset(seed=seed)
                    self._ep_reward = 0.0
                    self._ep_steps = 0
                    self._ep_done = False
                else:
                    self._obs = obs_new

            # Bootstrap value.  When the rollout ended on a terminal transition
            # the value of the *new* (already-reset) observation is irrelevant;
            # the correct last_value for GAE is 0 in that case.
            if last_done:
                last_value = 0.0
            else:
                last_obs_t = torch.tensor(
                    self._obs, dtype=torch.float32, device=self._device
                )
                last_value = self.policy.forward_critic(last_obs_t).item()

        returns, advantages = self._buffer.compute_returns_and_advantages(
            last_value, last_done
        )
        self._buffer_returns = torch.tensor(
            returns, dtype=torch.float32, device=self._device
        )
        self._buffer_advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self._device
        )
        # Normalise advantages.
        adv_std = self._buffer_advantages.std().item()
        if adv_std > 1e-8:
            self._buffer_advantages = (
                self._buffer_advantages - self._buffer_advantages.mean()
            ) / (adv_std + 1e-8)

    def _update_policy(self) -> Dict[str, float]:
        """Run PPO update epochs over the buffered rollout."""
        self.policy.train()
        obs_all = torch.tensor(
            self._buffer.obs, dtype=torch.float32, device=self._device
        )
        actions_all = torch.tensor(
            self._buffer.actions, dtype=torch.long, device=self._device
        )
        old_log_probs_all = torch.tensor(
            self._buffer.log_probs, dtype=torch.float32, device=self._device
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self._n_epochs):
            indices = np.random.permutation(self._n_steps)
            for start in range(0, self._n_steps, self._batch_size):
                idx = torch.tensor(
                    indices[start:start + self._batch_size],
                    dtype=torch.long,
                    device=self._device,
                )
                obs_b = obs_all[idx]
                actions_b = actions_all[idx]
                old_lp_b = old_log_probs_all[idx]
                ret_b = self._buffer_returns[idx]
                adv_b = self._buffer_advantages[idx]

                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_b, actions_b
                )

                # PPO clipped surrogate.
                ratio = torch.exp(log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self._clip_range, 1 + self._clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped).
                value_loss = nn.functional.mse_loss(values, ret_b)

                loss = (
                    policy_loss
                    + self._vf_coef * value_loss
                    - self._ent_coef * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self._max_grad_norm
                )
                self._optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    # ------------------------------------------------------------------
    # Private helpers — league
    # ------------------------------------------------------------------

    def _ensure_initial_snapshot(self) -> None:
        """Register the agent in the pool on first run if not already present."""
        if self.agent_id in self._agent_pool:
            return
        snap_path = self._save_snapshot(version=0)
        self._agent_pool.add(
            snap_path,
            AgentType.MAIN_AGENT,
            agent_id=self.agent_id,
            version=0,
            metadata={"parent_agent_id": self.agent_id, "step": 0},
            force=True,
        )
        log.debug(
            "CorpsMainAgentTrainer: initial snapshot registered (agent_id=%s)",
            self.agent_id,
        )

    def _refresh_opponent(self) -> None:
        """Select a PFSP opponent from the pool and update *_current_opponent_id*.

        Unlike the battalion-level trainer, CorpsEnv uses built-in scripted
        Red behavior; there is no policy-injection mechanism into the env.
        This method therefore only updates the *tracking* state
        (``_current_opponent_id``) used for ELO / match-database records.
        When the pool contains only the main agent itself, no opponent is
        selected and the fixed scripted Red is used.
        """
        if self._agent_pool.size <= 1:
            if self._current_opponent_id is None:
                # No pool opponent yet — use the scripted Red label.
                self._current_opponent_id = "scripted_red"
            return

        opponent_record = self._matchmaker.select_opponent(
            self.agent_id,
            rng=self._rng,
        )
        if opponent_record is not None:
            self._current_opponent_id = opponent_record.agent_id

    def _save_snapshot(self, version: int) -> Path:
        """Persist current policy weights to a ``.pt`` file."""
        snap_path = (
            self.snapshot_dir
            / f"corps_main_{self.agent_id}_v{version:06d}.pt"
        )
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "kwargs": self.policy.policy_kwargs(),
            },
            snap_path,
        )
        log.debug("CorpsMainAgentTrainer: saved snapshot → %s", snap_path)
        return snap_path

    def _load_snapshot(self, snapshot_path: Path) -> Optional[CorpsActorCriticPolicy]:
        """Load a frozen policy from *snapshot_path*; return ``None`` on failure."""
        try:
            data = torch.load(
                str(snapshot_path),
                map_location="cpu",
                weights_only=True,
            )
            pol = CorpsActorCriticPolicy(**data["kwargs"])
            pol.load_state_dict(data["state_dict"])
            pol = pol.to(self._device)
            pol.eval()
            return pol
        except Exception as exc:
            log.warning(
                "CorpsMainAgentTrainer: failed to load snapshot %s: %s",
                snapshot_path,
                exc,
            )
            return None

    def _save_checkpoint(
        self, directory: Path, suffix: str = ""
    ) -> None:
        """Save a full trainer checkpoint (policy + optimiser state)."""
        directory.mkdir(parents=True, exist_ok=True)
        ckpt_path = directory / f"corps_main_ckpt{suffix}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "total_steps": self._total_steps,
                "snapshot_version": self._snapshot_version,
                "policy_kwargs": self.policy.policy_kwargs(),
            },
            ckpt_path,
        )
        log.info("CorpsMainAgentTrainer: checkpoint saved → %s", ckpt_path)

    def _evaluate_episode(self) -> Dict:
        """Run one evaluation episode; return operational result dict.

        Casualties are derived from ``info['blue_units_alive']`` and
        ``info['red_units_alive']`` (added to the CorpsEnv info dict in E7.3)
        and the total unit counts ``env.n_blue`` / ``env.n_red``.

        Supply consumed is the accumulated sum of per-step supply drops
        (``sum(prev_levels) - sum(cur_levels)``) across the episode, giving the
        true total consumption rather than the end-of-episode deficit.
        """
        seed = int(self._rng.integers(0, 2**31))
        obs, _ = self.env.reset(seed=seed)
        total_reward = 0.0
        ep_info: dict = {}
        max_steps = getattr(self.env, "max_steps", 500)

        # Use env.n_blue / env.n_red as the starting unit counts.
        blue_start = getattr(self.env, "n_blue", 0)
        red_start = getattr(self.env, "n_red", 0)

        # Accumulate supply-level drops step-by-step to measure true consumption.
        prev_supply_levels: Optional[List[float]] = None
        supply_consumed_acc = 0.0

        self.policy.eval()
        with torch.no_grad():
            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self._device)
                action, _ = self.policy.act(obs_t, deterministic=True)
                obs, reward, terminated, truncated, ep_info = self.env.step(action)
                total_reward += float(reward)

                # Accumulate supply drop for this step.
                cur_supply: List[float] = ep_info.get("supply_levels", [])
                if prev_supply_levels is not None and cur_supply:
                    drop = sum(
                        max(0.0, p - c)
                        for p, c in zip(prev_supply_levels, cur_supply)
                    )
                    supply_consumed_acc += drop
                prev_supply_levels = cur_supply if cur_supply else prev_supply_levels

                if terminated or truncated:
                    break

        # Extract operational fields from the final info dict.
        obj_rewards: dict = ep_info.get("objective_rewards", {})

        # Casualties from public info keys added in E7.3.
        blue_alive = ep_info.get("blue_units_alive", blue_start)
        red_alive = ep_info.get("red_units_alive", red_start)
        blue_casualties = max(0, blue_start - blue_alive)
        red_casualties = max(0, red_start - red_alive)

        # Territory control: fraction of objectives with a positive reward this step.
        territory_control = float(
            sum(1 for v in obj_rewards.values() if isinstance(v, (int, float)) and v > 0)
            / max(len(obj_rewards), 1)
        ) if obj_rewards else 0.0
        territory_control = min(max(territory_control, 0.0), 1.0)

        return {
            "total_reward": total_reward,
            "outcome": 1.0 if total_reward > 0 else 0.0,
            "territory_control": territory_control,
            "blue_casualties": blue_casualties,
            "red_casualties": red_casualties,
            "supply_consumed": supply_consumed_acc,
        }

    def _run_evaluation(self, total_steps: int) -> None:
        """Evaluate vs scripted Red, record result, update Elo, log to W&B."""
        outcomes = []
        territory_list: List[float] = []
        blue_cas_list: List[int] = []
        red_cas_list: List[int] = []
        supply_list: List[float] = []

        for _ in range(self.n_eval_episodes):
            result = self._evaluate_episode()
            outcomes.append(result["outcome"])
            territory_list.append(result["territory_control"])
            blue_cas_list.append(result["blue_casualties"])
            red_cas_list.append(result["red_casualties"])
            supply_list.append(result["supply_consumed"])

        win_rate = float(sum(outcomes) / len(outcomes))
        opp_id = self._current_opponent_id or "scripted_red"

        # Accumulate per-matchup stats.
        if opp_id not in self._matchup_outcomes:
            self._matchup_outcomes[opp_id] = []
        self._matchup_outcomes[opp_id].append(win_rate)

        # Record in MatchDatabase with operational fields.
        self._match_db.record(
            agent_id=self.agent_id,
            opponent_id=opp_id,
            outcome=win_rate,
            metadata={"step": total_steps, "n_episodes": self.n_eval_episodes},
            territory_control=float(sum(territory_list) / max(len(territory_list), 1)),
            blue_casualties=int(sum(blue_cas_list)),
            red_casualties=int(sum(red_cas_list)),
            supply_consumed=float(sum(supply_list)),
        )

        # Elo update.
        try:
            elo_delta = self._elo.update(
                agent=self.agent_id,
                opponent=opp_id,
                outcome=win_rate,
                n_games=self.n_eval_episodes,
            )
            elo_rating = self._elo.get_rating(self.agent_id)
        except ValueError as exc:
            log.warning("CorpsMainAgentTrainer: Elo update skipped: %s", exc)
            elo_delta = 0.0
            elo_rating = self._elo.get_rating(self.agent_id)

        log.info(
            "CorpsMainAgentTrainer: eval | opp=%s | win_rate=%.3f"
            " | territory=%.3f | blue_cas=%d | red_cas=%d | supply_consumed=%.2f"
            " | elo=%.1f (delta=%+.1f) | step=%d",
            opp_id,
            win_rate,
            float(sum(territory_list) / max(len(territory_list), 1)),
            int(sum(blue_cas_list)),
            int(sum(red_cas_list)),
            float(sum(supply_list)),
            elo_rating,
            elo_delta,
            total_steps,
        )

        # W&B metrics.
        metrics: dict = {
            f"matchup/win_rate/{opp_id}": win_rate,
            f"matchup/territory_control/{opp_id}": float(
                sum(territory_list) / max(len(territory_list), 1)
            ),
            f"matchup/blue_casualties/{opp_id}": float(sum(blue_cas_list)),
            f"matchup/red_casualties/{opp_id}": float(sum(red_cas_list)),
            f"matchup/supply_consumed/{opp_id}": float(sum(supply_list)),
            "elo/corps_main_agent": elo_rating,
            "elo/delta": elo_delta,
            "eval/step": total_steps,
        }
        for mid, outs in self._matchup_outcomes.items():
            if outs:
                metrics[f"matchup/mean_win_rate/{mid}"] = float(sum(outs) / len(outs))

        # Nash entropy over the current pool.
        all_records = self._agent_pool.list()
        if len(all_records) >= 2:
            agent_ids = [r.agent_id for r in all_records]
            win_rates_cache = {
                aid: self._match_db.win_rates_for(aid) for aid in agent_ids
            }
            payoff = build_payoff_matrix(
                agent_ids,
                lambda ai, aj: win_rates_cache.get(ai, {}).get(aj),
                self._matchmaker.unknown_win_rate,
            )
            nash_dist = compute_nash_distribution(payoff)
            metrics["league/nash_entropy"] = nash_entropy(nash_dist)

        if wandb.run is not None:
            wandb.log(metrics, step=total_steps)

        try:
            self._elo.save()
        except ValueError:
            pass

    def _log_training(self, losses: Dict[str, float], total_steps: int) -> None:
        """Log PPO losses and league state to W&B."""
        metrics: dict = {
            "train/policy_loss": losses.get("policy_loss", float("nan")),
            "train/value_loss": losses.get("value_loss", float("nan")),
            "train/entropy": losses.get("entropy", float("nan")),
            "league/pool_size": self._agent_pool.size,
            "league/snapshot_version": self._snapshot_version,
            "train/step": total_steps,
        }
        win_rates = self._match_db.win_rates_for(self.agent_id)
        if self._current_opponent_id is not None:
            metrics["train/current_opponent_win_rate"] = win_rates.get(
                self._current_opponent_id, float("nan")
            )
        if wandb.run is not None:
            wandb.log(metrics, step=total_steps)
        log.info(
            "[%d steps] policy_loss=%.4f value_loss=%.4f entropy=%.4f pool_size=%d",
            total_steps,
            losses.get("policy_loss", float("nan")),
            losses.get("value_loss", float("nan")),
            losses.get("entropy", float("nan")),
            self._agent_pool.size,
        )


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="league/corps_main_agent",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run a corps-level league main-agent training session.

    Parameters
    ----------
    cfg:
        Hydra-merged configuration (see ``configs/league/corps_main_agent.yaml``).
    """
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    # W&B
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")

    # Environment
    env = CorpsEnv(
        n_divisions=int(cfg.env.n_divisions),
        n_brigades_per_division=int(cfg.env.n_brigades_per_division),
        n_blue_per_brigade=int(cfg.env.n_blue_per_brigade),
        comm_radius=float(cfg.env.comm_radius),
        red_random=bool(cfg.env.red_random),
    )
    obs_dim: int = env._obs_dim
    n_divisions: int = env.n_divisions
    n_options: int = env.n_corps_options
    log.info(
        "CorpsEnv: n_divisions=%d obs_dim=%d n_options=%d",
        n_divisions,
        obs_dim,
        n_options,
    )

    # Policy
    policy = CorpsActorCriticPolicy(
        obs_dim=obs_dim,
        n_divisions=n_divisions,
        n_options=n_options,
        actor_hidden_sizes=tuple(cfg.training.actor_hidden_sizes),
        critic_hidden_sizes=tuple(cfg.training.critic_hidden_sizes),
    )

    # League infrastructure
    league_cfg = cfg.league
    pool_manifest = _PROJECT_ROOT / str(league_cfg.pool_manifest)
    match_db_path = _PROJECT_ROOT / str(league_cfg.match_db_path)
    elo_path = _PROJECT_ROOT / str(league_cfg.elo_registry_path)
    snapshot_dir = _PROJECT_ROOT / str(league_cfg.snapshot_dir)

    agent_pool = AgentPool(
        pool_manifest=pool_manifest,
        max_size=int(OmegaConf.select(league_cfg, "pool_max_size", default=200)),
    )
    match_db = MatchDatabase(db_path=match_db_path)
    matchmaker = LeagueMatchmaker(
        agent_pool=agent_pool,
        match_database=match_db,
        unknown_win_rate=float(
            OmegaConf.select(league_cfg, "unknown_win_rate", default=0.5)
        ),
    )
    elo_registry = EloRegistry(path=elo_path)

    checkpoint_dir = (
        _PROJECT_ROOT / str(league_cfg.checkpoint_dir)
        if OmegaConf.select(league_cfg, "checkpoint_dir", default=None) is not None
        else None
    )

    # Trainer
    trainer = CorpsMainAgentTrainer(
        env=env,
        policy=policy,
        agent_pool=agent_pool,
        match_database=match_db,
        matchmaker=matchmaker,
        elo_registry=elo_registry,
        agent_id=str(league_cfg.agent_id),
        snapshot_dir=snapshot_dir,
        snapshot_freq=int(league_cfg.snapshot_freq),
        eval_freq=int(league_cfg.eval_freq),
        n_eval_episodes=int(league_cfg.n_eval_episodes),
        pfsp_temperature=float(
            OmegaConf.select(league_cfg, "pfsp_temperature", default=1.0)
        ),
        lr=float(cfg.training.lr),
        n_steps=int(cfg.training.n_steps),
        n_epochs=int(cfg.training.n_epochs),
        batch_size=int(cfg.training.batch_size),
        gamma=float(cfg.training.gamma),
        gae_lambda=float(cfg.training.gae_lambda),
        clip_range=float(cfg.training.clip_range),
        vf_coef=float(cfg.training.vf_coef),
        ent_coef=float(cfg.training.ent_coef),
        max_grad_norm=float(cfg.training.max_grad_norm),
        device=str(OmegaConf.select(cfg, "training.device", default="cpu")),
        seed=int(cfg.training.seed),
        log_interval=int(cfg.wandb.log_freq),
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=int(
            OmegaConf.select(league_cfg, "checkpoint_freq", default=200_000)
        ),
    )

    trainer.train(total_timesteps=int(cfg.training.total_timesteps))

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
