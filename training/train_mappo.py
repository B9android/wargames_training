# training/train_mappo.py
"""MAPPO training entry point for multi-battalion cooperative combat.

Implements Multi-Agent PPO (MAPPO, Yu et al. 2021) with:

* A **shared actor** across all Blue agents (parameter sharing) or
  separate actors per agent (configurable via ``training.share_parameters``).
* A **centralized critic** conditioned on the global state tensor returned
  by :meth:`~envs.multi_battalion_env.MultiBattalionEnv.state`.
* Shared rollout buffer collecting transitions from all Blue agents.
* W&B logging with per-agent and aggregate reward curves.

Usage::

    # With default MAPPO 2v2 config
    python training/train_mappo.py --config-name experiment_mappo_2v2

    # Hydra CLI overrides
    python training/train_mappo.py training.total_timesteps=200000 training.lr=3e-4

Architecture
------------
- Blue agents (n_blue) are controlled by MAPPO.  Their local observations
  drive the shared actor; the global state drives the centralized critic.
- Red agents receive zero actions (stationary) by default, creating a
  cooperative training scenario where Blue must learn to win together.
  Set ``env.red_random=true`` to use random Red actions instead.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from envs.multi_battalion_env import MultiBattalionEnv
from models.mappo_policy import MAPPOPolicy

log = logging.getLogger(__name__)

__all__ = ["MAPPORolloutBuffer", "MAPPOTrainer"]

# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


@dataclass
class MAPPORolloutBuffer:
    """Rollout buffer for multi-agent MAPPO trajectories.

    Stores ``n_steps`` timesteps of experience for ``n_agents`` agents.
    The centralized critic sees global state (one per timestep), while each
    agent has its own observation, action, log-probability and reward.

    Parameters
    ----------
    n_steps:
        Number of environment steps per rollout.
    n_agents:
        Number of controlled (Blue) agents.
    obs_dim:
        Per-agent local observation dimensionality.
    action_dim:
        Per-agent action dimensionality.
    state_dim:
        Global state dimensionality.
    gamma:
        Discount factor for GAE.
    gae_lambda:
        GAE smoothing parameter λ.
    """

    n_steps: int
    n_agents: int
    obs_dim: int
    action_dim: int
    state_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Populated by __post_init__
    obs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)         # (n_steps, n_agents) — per-agent
    values: np.ndarray = field(init=False)
    global_states: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    _ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero all storage and reset the write pointer."""
        self.obs = np.zeros((self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_agents, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.n_steps,), dtype=np.float32)
        self.global_states = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self._ptr = 0

    @property
    def full(self) -> bool:
        return self._ptr >= self.n_steps

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
        global_state: np.ndarray,
    ) -> None:
        """Store one timestep of experience.

        Parameters
        ----------
        obs:
            Local observations of shape ``(n_agents, obs_dim)``.
        actions:
            Actions of shape ``(n_agents, action_dim)`` — already clipped to
            the environment's action-space bounds.
        log_probs:
            Log-probabilities of shape ``(n_agents,)`` — evaluated on the
            clipped actions.
        rewards:
            Per-agent rewards of shape ``(n_agents,)``.
        dones:
            Per-agent done flags of shape ``(n_agents,)`` — ``1.0`` when an
            agent's episode ended (terminated or truncated) this step.
        value:
            Centralized critic value estimate (scalar float).
        global_state:
            Global state of shape ``(state_dim,)``.
        """
        if self._ptr >= self.n_steps:
            raise RuntimeError("Buffer is full; call reset() before adding more data.")
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = value
        self.global_states[t] = global_state
        self._ptr += 1

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute per-agent GAE advantages and discounted returns in-place.

        Each agent uses its own reward stream and its own termination mask,
        while bootstrapping from the **shared centralized critic value**
        V(s_t).  This follows the MAPPO formulation where advantages are
        computed independently per agent while the value function is
        centralized.

        Parameters
        ----------
        last_value:
            Centralized critic estimate for the state **after** the last
            collected step (used for bootstrapping).
        """
        last_gae: np.ndarray = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]

            # Per-agent non-terminal mask: (n_agents,)
            next_non_terminal = 1.0 - self.dones[t]  # (n_agents,)

            # δ_t^i = r_t^i + γ · V(s_{t+1}) · (1 − done_t^i) − V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values[:, np.newaxis]

    def get_batches(
        self, batch_size: int, device: torch.device
    ) -> list[dict[str, torch.Tensor]]:
        """Flatten the buffer across agents and time, then split into minibatches.

        Returns a list of dicts each containing keys:
        ``obs``, ``actions``, ``old_log_probs``, ``advantages``,
        ``returns``, ``global_states``, ``agent_indices``.
        """
        T, A = self.n_steps, self.n_agents

        # Normalize advantages (across the entire buffer)
        adv = self.advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.reshape(T, A)

        # Expand global states to match (T, A, state_dim) for convenience
        states_exp = np.broadcast_to(
            self.global_states[:, np.newaxis, :], (T, A, self.state_dim)
        ).copy()

        # Flat views: (T*A, ...)
        obs_flat = self.obs.reshape(T * A, self.obs_dim)
        actions_flat = self.actions.reshape(T * A, self.action_dim)
        lp_flat = self.log_probs.reshape(T * A)
        adv_flat = adv.reshape(T * A)
        ret_flat = self.returns.reshape(T * A)
        states_flat = states_exp.reshape(T * A, self.state_dim)
        # agent indices (0 … A-1) repeated T times
        agent_idx_flat = np.tile(np.arange(A, dtype=np.int64), T)

        n_samples = T * A
        indices = np.random.permutation(n_samples)
        batches = []
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            batches.append(
                {
                    "obs": torch.as_tensor(obs_flat[idx], device=device),
                    "actions": torch.as_tensor(actions_flat[idx], device=device),
                    "old_log_probs": torch.as_tensor(lp_flat[idx], device=device),
                    "advantages": torch.as_tensor(adv_flat[idx], device=device),
                    "returns": torch.as_tensor(ret_flat[idx], device=device),
                    "global_states": torch.as_tensor(states_flat[idx], device=device),
                    "agent_indices": torch.as_tensor(agent_idx_flat[idx], device=device),
                }
            )
        return batches


# ---------------------------------------------------------------------------
# MAPPO Trainer
# ---------------------------------------------------------------------------


class MAPPOTrainer:
    """MAPPO training loop for cooperative multi-battalion combat.

    Runs ``MultiBattalionEnv`` with Blue agents controlled by MAPPO and
    Red agents receiving zero (stationary) or random actions.  Logs
    per-agent and aggregate metrics to W&B.

    Parameters
    ----------
    policy:
        :class:`~models.mappo_policy.MAPPOPolicy` instance.
    env:
        :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance.
    n_steps:
        Rollout length before each policy update.
    n_epochs:
        PPO update epochs per rollout.
    batch_size:
        Minibatch size used inside each epoch.
    lr:
        Adam learning rate for both actor and critic.
    gamma:
        Discount factor.
    gae_lambda:
        GAE λ parameter.
    clip_range:
        PPO ε clipping range.
    vf_coef:
        Value-function loss coefficient.
    ent_coef:
        Entropy bonus coefficient.
    max_grad_norm:
        Gradient clipping norm.
    red_random:
        When ``True`` Red agents receive random actions; when ``False``
        they receive zero (stationary) actions.
    device:
        Torch device for policy computations.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        policy: MAPPOPolicy,
        env: MultiBattalionEnv,
        n_steps: int = 256,
        n_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        red_random: bool = False,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.policy = policy
        self.env = env
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.red_random = red_random
        self.device = torch.device(device)
        self.seed = seed

        self.policy = self.policy.to(self.device)

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.n_blue = env.n_blue
        self.n_red = env.n_red
        self.obs_dim = env._obs_dim
        self.action_dim = env._act_space.shape[0]
        self.state_dim = env._state_dim
        # Action-space bounds used to clip sampled actions before env step.
        # We use the public action_space() accessor so the bounds stay in sync
        # with any future changes to the environment's action space definition.
        _act_space = env.action_space(env.possible_agents[0])
        self._act_low: np.ndarray = _act_space.low    # shape (action_dim,)
        self._act_high: np.ndarray = _act_space.high  # shape (action_dim,)

        self.buffer = MAPPORolloutBuffer(
            n_steps=n_steps,
            n_agents=self.n_blue,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self._total_steps: int = 0
        self._episode: int = 0
        self._rng = np.random.default_rng(seed)

        # Current episode state
        self._obs_buf: dict[str, np.ndarray] = {}
        self._ep_rewards: dict[str, float] = {f"blue_{i}": 0.0 for i in range(self.n_blue)}
        self._ep_len: int = 0
        self._ep_reward_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _reset_env(self, seed: Optional[int] = None) -> None:
        """Reset the environment and update internal obs buffer."""
        s = seed if seed is not None else int(self._rng.integers(0, 2**31))
        obs, _ = self.env.reset(seed=s)
        self._obs_buf = obs
        self._ep_rewards = {f"blue_{i}": 0.0 for i in range(self.n_blue)}
        self._ep_len = 0

    def _get_blue_obs_array(self) -> np.ndarray:
        """Return Blue observations as a ``(n_blue, obs_dim)`` array.

        Any Blue agent not present in ``env.agents`` (terminated or truncated
        in a prior step) is assigned a zero observation, even if a stale entry
        remains in ``_obs_buf`` from the step when it died.  Within a
        PettingZoo episode, once an agent leaves ``env.agents`` it does not
        return, so the zero observation persists for the remainder of the
        rollout until the next ``_reset_env()`` call.
        """
        zero_obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs_list: list[np.ndarray] = []
        for i in range(self.n_blue):
            agent_id = f"blue_{i}"
            if agent_id in self.env.agents:
                obs_list.append(self._obs_buf.get(agent_id, zero_obs))
            else:
                obs_list.append(zero_obs)
        return np.stack(obs_list)

    def _red_actions(self) -> dict[str, np.ndarray]:
        """Return action dict for Red agents (random or zero)."""
        red_acts = {}
        for i in range(self.n_red):
            agent_id = f"red_{i}"
            if agent_id in self.env.agents:
                if self.red_random:
                    red_acts[agent_id] = self.env.action_space(agent_id).sample()
                else:
                    red_acts[agent_id] = np.zeros(self.action_dim, dtype=np.float32)
        return red_acts

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> None:
        """Collect ``n_steps`` transitions into the rollout buffer.

        Handles episode resets mid-rollout transparently.
        """
        self.buffer.reset()

        if not self.env.agents:
            self._reset_env()

        for _ in range(self.n_steps):
            # Make sure there are live blue agents
            blue_alive = [a for a in self.env.agents if a.startswith("blue_")]
            if not blue_alive:
                self._reset_env()
                blue_alive = [a for a in self.env.agents if a.startswith("blue_")]

            obs_arr = self._get_blue_obs_array()  # (n_blue, obs_dim)
            global_state = self.env.state()       # (state_dim,)

            obs_t = torch.as_tensor(obs_arr, device=self.device)
            state_t = torch.as_tensor(global_state, device=self.device)

            # Centralized critic value
            value = float(self.policy.get_value(state_t.unsqueeze(0)).item())

            # Actor: sample actions (unclipped) for each Blue agent.
            # Log-probs are recomputed after clipping below.
            with torch.no_grad():
                if self.policy.share_parameters:
                    dist = self.policy.get_actor().get_distribution(obs_t)
                    actions_t = dist.rsample()  # (n_blue, action_dim)
                else:
                    actions_list = []
                    for i in range(self.n_blue):
                        actor = self.policy.get_actor(i)
                        d = actor.get_distribution(obs_t[i : i + 1])
                        actions_list.append(d.rsample())
                    actions_t = torch.cat(actions_list, dim=0)

            actions_np = actions_t.cpu().numpy()   # (n_blue, action_dim) — unclipped

            # Clip actions to the env's action-space bounds and recompute
            # log-probs on the clipped actions so that the stored (action,
            # log_prob) pair is consistent (avoids PPO ratio errors caused by
            # the Normal distribution being unbounded while the env expects a
            # bounded action, e.g. fire ∈ [0, 1]).
            actions_clipped = np.clip(actions_np, self._act_low, self._act_high)
            actions_clipped_t = torch.as_tensor(actions_clipped, device=self.device)
            with torch.no_grad():
                if self.policy.share_parameters:
                    log_probs_np = (
                        dist.log_prob(actions_clipped_t).sum(-1).cpu().numpy()
                    )
                else:
                    lp_list = []
                    for i in range(self.n_blue):
                        actor = self.policy.get_actor(i)
                        d = actor.get_distribution(obs_t[i : i + 1])
                        lp = d.log_prob(actions_clipped_t[i : i + 1]).sum(-1)
                        lp_list.append(lp)
                    log_probs_np = torch.cat(lp_list, dim=0).cpu().numpy()

            # Build full action dict for env (using the clipped actions)
            action_dict: dict[str, np.ndarray] = {}
            for i, agent_id in enumerate([f"blue_{j}" for j in range(self.n_blue)]):
                if agent_id in self.env.agents:
                    action_dict[agent_id] = actions_clipped[i]
            action_dict.update(self._red_actions())

            obs_new, rewards, terminated, truncated, _ = self.env.step(action_dict)

            # Collect per-blue rewards
            rew_arr = np.array(
                [float(rewards.get(f"blue_{i}", 0.0)) for i in range(self.n_blue)],
                dtype=np.float32,
            )
            for i in range(self.n_blue):
                self._ep_rewards[f"blue_{i}"] += rew_arr[i]
            self._ep_len += 1

            # Per-agent done flags: agent is done if terminated or truncated this step
            dones_arr = np.array(
                [
                    float(terminated.get(f"blue_{i}", False) or truncated.get(f"blue_{i}", False))
                    for i in range(self.n_blue)
                ],
                dtype=np.float32,
            )

            all_blue_done = bool(dones_arr.all())

            self.buffer.add(
                obs=obs_arr,
                actions=actions_clipped,
                log_probs=log_probs_np,
                rewards=rew_arr,
                dones=dones_arr,
                value=value,
                global_state=global_state,
            )

            self._obs_buf = obs_new
            self._total_steps += 1

            if all_blue_done or not self.env.agents:
                self._episode += 1
                self._ep_reward_history.append(dict(self._ep_rewards))
                self._reset_env()

        # Bootstrap value for GAE
        global_state_last = self.env.state()
        state_last_t = torch.as_tensor(global_state_last, device=self.device)
        with torch.no_grad():
            last_value = float(self.policy.get_value(state_last_t.unsqueeze(0)).item())

        self.buffer.compute_returns_and_advantages(last_value)

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self) -> dict[str, float]:
        """Run PPO update epochs over the collected rollout buffer.

        Returns
        -------
        A dict of mean losses:
        ``policy_loss``, ``value_loss``, ``entropy``, ``total_loss``.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_batches = 0

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)
            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                global_states = batch["global_states"]
                agent_indices = batch["agent_indices"]

                if self.policy.share_parameters:
                    # All samples use the same actor
                    new_log_probs, entropy, values = self.policy.evaluate_actions(
                        obs, actions, global_states
                    )
                else:
                    # Evaluate each sample with its own agent's actor and
                    # reconstruct results in the original (shuffled) order.
                    new_log_probs = torch.empty_like(old_log_probs)
                    entropy_t = torch.empty_like(old_log_probs)
                    values = torch.empty_like(old_log_probs)
                    for idx in agent_indices.unique():
                        mask = agent_indices == idx
                        pos = mask.nonzero(as_tuple=True)[0]
                        lp, ent, val = self.policy.evaluate_actions(
                            obs[mask], actions[mask], global_states[mask], int(idx.item())
                        )
                        new_log_probs[pos] = lp
                        entropy_t[pos] = ent
                        values[pos] = val
                    entropy = entropy_t

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(
                    ratio * advantages, clipped_ratio * advantages
                ).mean()

                # Value function loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_batches += 1

        n = max(total_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_policy_loss + self.vf_coef * total_value_loss
                           - self.ent_coef * total_entropy) / n,
        }

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def _log_wandb(self, losses: dict[str, float]) -> None:
        """Log per-agent reward curves and training losses to W&B."""
        log_dict: dict = {
            "time/total_steps": self._total_steps,
            "time/episodes": self._episode,
        }

        # Per-agent and aggregate rewards from last completed episodes
        if self._ep_reward_history:
            # Use episodes completed since last log
            recent = self._ep_reward_history
            self._ep_reward_history = []
            for i in range(self.n_blue):
                agent_id = f"blue_{i}"
                ep_rews = [ep.get(agent_id, 0.0) for ep in recent]
                log_dict[f"reward/agent_{i}"] = float(np.mean(ep_rews))
            # Aggregate: mean total reward per episode
            total_rews = [sum(ep.values()) for ep in recent]
            log_dict["reward/team_mean"] = float(np.mean(total_rews))
            log_dict["reward/n_episodes"] = len(recent)

        log_dict.update({f"train/{k}": v for k, v in losses.items()})
        wandb.log(log_dict, step=self._total_steps)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 2000,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 50_000,
    ) -> None:
        """Run the MAPPO training loop.

        Parameters
        ----------
        total_timesteps:
            Total environment steps to train for.
        log_interval:
            Log to W&B every this many steps.
        checkpoint_dir:
            Directory in which to save model checkpoints (``None``
            disables checkpointing).
        checkpoint_freq:
            Save a checkpoint every this many timesteps.
        """
        self._reset_env(seed=self.seed)
        last_log = 0
        last_ckpt = 0

        log.info(
            "MAPPO training start | total_timesteps=%d | n_blue=%d | share_params=%s",
            total_timesteps,
            self.n_blue,
            self.policy.share_parameters,
        )

        while self._total_steps < total_timesteps:
            self.collect_rollout()
            losses = self.update_policy()

            if self._total_steps - last_log >= log_interval:
                self._log_wandb(losses)
                log.info(
                    "[%d/%d steps] policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                    self._total_steps,
                    total_timesteps,
                    losses["policy_loss"],
                    losses["value_loss"],
                    losses["entropy"],
                )
                last_log = self._total_steps

            if checkpoint_dir is not None and self._total_steps - last_ckpt >= checkpoint_freq:
                self._save_checkpoint(checkpoint_dir)
                last_ckpt = self._total_steps

        # Final checkpoint
        if checkpoint_dir is not None:
            self._save_checkpoint(checkpoint_dir, suffix="_final")

        log.info("MAPPO training complete. Total steps: %d", self._total_steps)

    def _save_checkpoint(self, checkpoint_dir: Path, suffix: str = "") -> None:
        """Save policy weights to *checkpoint_dir*."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"mappo_policy{suffix}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self._total_steps,
                "episodes": self._episode,
            },
            path,
        )
        log.info("Saved checkpoint → %s", path)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="experiment_mappo_2v2",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run a MAPPO training session from a Hydra configuration.

    Parameters
    ----------
    cfg:
        Hydra-merged configuration (see
        ``configs/experiment_mappo_2v2.yaml``).
    """
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    checkpoint_dir = _PROJECT_ROOT / cfg.eval.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = MultiBattalionEnv(
        n_blue=int(cfg.env.n_blue),
        n_red=int(cfg.env.n_red),
        map_width=float(cfg.env.map_width),
        map_height=float(cfg.env.map_height),
        max_steps=int(cfg.env.max_steps),
        randomize_terrain=bool(cfg.env.randomize_terrain),
        hill_speed_factor=float(cfg.env.hill_speed_factor),
        visibility_radius=float(cfg.env.visibility_radius),
    )

    obs_dim: int = env._obs_dim
    action_dim: int = env._act_space.shape[0]
    state_dim: int = env._state_dim

    log.info(
        "Env: n_blue=%d n_red=%d obs_dim=%d action_dim=%d state_dim=%d",
        env.n_blue,
        env.n_red,
        obs_dim,
        action_dim,
        state_dim,
    )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    share_params = bool(OmegaConf.select(cfg, "training.share_parameters", default=True))
    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=env.n_blue,
        share_parameters=share_params,
        actor_hidden_sizes=tuple(cfg.training.actor_hidden_sizes),
        critic_hidden_sizes=tuple(cfg.training.critic_hidden_sizes),
    )
    param_counts = policy.parameter_count()
    log.info(
        "MAPPOPolicy | share_parameters=%s | actor_params=%d | critic_params=%d | total=%d",
        share_params,
        param_counts["actor"],
        param_counts["critic"],
        param_counts["total"],
    )
    if run:
        wandb.log({"model/actor_params": param_counts["actor"],
                   "model/critic_params": param_counts["critic"],
                   "model/total_params": param_counts["total"]}, step=0)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = MAPPOTrainer(
        policy=policy,
        env=env,
        n_steps=int(cfg.training.n_steps),
        n_epochs=int(cfg.training.n_epochs),
        batch_size=int(cfg.training.batch_size),
        lr=float(cfg.training.lr),
        gamma=float(cfg.training.gamma),
        gae_lambda=float(cfg.training.gae_lambda),
        clip_range=float(cfg.training.clip_range),
        vf_coef=float(cfg.training.vf_coef),
        ent_coef=float(cfg.training.ent_coef),
        max_grad_norm=float(cfg.training.max_grad_norm),
        red_random=bool(OmegaConf.select(cfg, "env.red_random", default=False)),
        device=str(OmegaConf.select(cfg, "training.device", default="cpu")),
        seed=int(cfg.training.seed),
    )

    trainer.learn(
        total_timesteps=int(cfg.training.total_timesteps),
        log_interval=int(cfg.wandb.log_freq),
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=int(cfg.eval.checkpoint_freq),
    )

    env.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
