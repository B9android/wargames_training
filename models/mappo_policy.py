# models/mappo_policy.py
"""MAPPO actor and centralized critic for multi-battalion cooperative training.

Implements the Multi-Agent PPO (MAPPO) policy from Yu et al. (2021),
"The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games".

The key components are:

* :class:`MAPPOActor` — shared actor conditioned on **local observations**.
  All homogeneous agents on the same team share actor weights when
  ``share_parameters=True`` (the default).

* :class:`MAPPOCritic` — centralized critic conditioned on the **global
  state** tensor returned by :meth:`~envs.multi_battalion_env.MultiBattalionEnv.state`.

* :class:`MAPPOPolicy` — container that wires actor(s) and critic together
  and exposes a clean API used by :mod:`training.train_mappo`.

Typical usage::

    from models.mappo_policy import MAPPOPolicy

    policy = MAPPOPolicy(
        obs_dim=22,      # MultiBattalionEnv obs_dim for 2v2
        action_dim=3,    # move, rotate, fire
        state_dim=25,    # MultiBattalionEnv state_dim for 2v2
        n_agents=2,      # number of blue agents
        share_parameters=True,
    )
    # Sample actions for a batch of agents
    obs = torch.zeros(2, 22)
    actions, log_probs = policy.act(obs)           # shape (2,3), (2,)
    values = policy.get_value(state)               # shape ()
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = ["MAPPOActor", "MAPPOCritic", "MAPPOPolicy"]


def _build_mlp(in_dim: int, hidden_sizes: Tuple[int, ...]) -> Tuple[nn.Sequential, int]:
    """Return (Sequential trunk, output_dim)."""
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(cur, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.Tanh())
        cur = h
    return nn.Sequential(*layers), cur


class MAPPOActor(nn.Module):
    """Shared actor network for homogeneous MAPPO agents.

    Takes a **local observation** vector and produces a diagonal Gaussian
    action distribution.  The ``log_std`` parameters are learned but shared
    across the batch (not conditioned on the observation).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the per-agent local observation.
    action_dim:
        Dimensionality of the continuous action space.
    hidden_sizes:
        Sizes of the hidden layers in the shared trunk MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        trunk, out_dim = _build_mlp(obs_dim, hidden_sizes)
        self.trunk = trunk
        self.action_mean = nn.Linear(out_dim, action_dim)
        # Learnable log standard deviation (not observation-conditioned)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_mean, action_std)`` tensors.

        Parameters
        ----------
        obs:
            Local observation of shape ``(..., obs_dim)``.

        Returns
        -------
        mean : torch.Tensor — shape ``(..., action_dim)``
        std  : torch.Tensor — shape ``(..., action_dim)``
        """
        h = self.trunk(obs)
        mean = self.action_mean(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Return a :class:`~torch.distributions.Normal` over actions."""
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities and entropy for given (obs, action) pairs.

        Parameters
        ----------
        obs:
            Local observations of shape ``(batch, obs_dim)``.
        actions:
            Actions of shape ``(batch, action_dim)``.

        Returns
        -------
        log_probs : torch.Tensor — shape ``(batch,)``
        entropy   : torch.Tensor — shape ``(batch,)``
        """
        dist = self.get_distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class MAPPOCritic(nn.Module):
    """Centralized critic conditioned on the global state tensor.

    Receives the **global state** (all agents' positions, headings,
    strengths and morale) and produces a scalar value estimate used for
    advantage computation in MAPPO.

    Parameters
    ----------
    state_dim:
        Dimensionality of the global state vector (output of
        :meth:`~envs.multi_battalion_env.MultiBattalionEnv.state`).
    hidden_sizes:
        Sizes of the hidden layers in the critic MLP.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        trunk, out_dim = _build_mlp(state_dim, hidden_sizes)
        self.trunk = trunk
        self.value_head = nn.Linear(out_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimates.

        Parameters
        ----------
        state:
            Global state tensor of shape ``(..., state_dim)``.

        Returns
        -------
        values : torch.Tensor — shape ``(...)``
        """
        return self.value_head(self.trunk(state)).squeeze(-1)


class MAPPOPolicy(nn.Module):
    """MAPPO policy: shared actor(s) plus a centralized critic.

    Supports two parameter-sharing modes for the actor:

    * ``share_parameters=True`` (default) — all *n_agents* agents use the
      **same** :class:`MAPPOActor` weights.  Memory scales as
      O(actor_params + critic_params) instead of O(n_agents * actor_params).
    * ``share_parameters=False`` — each agent gets its own
      :class:`MAPPOActor`; useful for ablation studies.

    In both cases there is a **single** :class:`MAPPOCritic` that all
    agents share.

    Parameters
    ----------
    obs_dim:
        Per-agent local observation dimensionality.
    action_dim:
        Per-agent action dimensionality (continuous).
    state_dim:
        Global state dimensionality.
    n_agents:
        Number of controlled agents (used only when
        ``share_parameters=False`` to build separate actor heads).
    share_parameters:
        Whether all agents share one actor.  Defaults to ``True``.
    actor_hidden_sizes:
        Hidden layer sizes for the actor trunk.
    critic_hidden_sizes:
        Hidden layer sizes for the critic trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int = 1,
        share_parameters: bool = True,
        actor_hidden_sizes: Tuple[int, ...] = (128, 64),
        critic_hidden_sizes: Tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.share_parameters = share_parameters
        self.actor_hidden_sizes: Tuple[int, ...] = tuple(actor_hidden_sizes)
        self.critic_hidden_sizes: Tuple[int, ...] = tuple(critic_hidden_sizes)

        if share_parameters:
            self.actor: nn.Module = MAPPOActor(obs_dim, action_dim, actor_hidden_sizes)
        else:
            self.actors: nn.ModuleList = nn.ModuleList(
                [MAPPOActor(obs_dim, action_dim, actor_hidden_sizes) for _ in range(n_agents)]
            )

        self.critic = MAPPOCritic(state_dim, critic_hidden_sizes)

    # ------------------------------------------------------------------
    # Actor helpers
    # ------------------------------------------------------------------

    def get_actor(self, agent_idx: int = 0) -> MAPPOActor:
        """Return the actor for *agent_idx*.

        When ``share_parameters=True`` the same actor is returned
        regardless of *agent_idx*.
        """
        if self.share_parameters:
            return self.actor  # type: ignore[return-value]
        return self.actors[agent_idx]  # type: ignore[index]

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent_idx: int = 0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions for a batch of observations.

        A batch dimension is added internally when *obs* is 1-D so that
        the return shapes are **always** ``(batch, action_dim)`` and
        ``(batch,)`` regardless of whether the caller passes a single
        observation or a batch.

        Parameters
        ----------
        obs:
            Local observations of shape ``(batch, obs_dim)`` or
            ``(obs_dim,)`` for a single observation.
        agent_idx:
            Index of the agent whose actor should be used.  Ignored when
            ``share_parameters=True``.
        deterministic:
            When ``True`` returns the distribution mean instead of
            sampling.

        Returns
        -------
        actions  : torch.Tensor — shape ``(batch, action_dim)``
        log_probs: torch.Tensor — shape ``(batch,)``
        """
        squeezed = obs.dim() == 1
        if squeezed:
            obs = obs.unsqueeze(0)
        actor = self.get_actor(agent_idx)
        dist = actor.get_distribution(obs)
        actions = dist.mean if deterministic else dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate(s) for the given global state(s).

        Parameters
        ----------
        state:
            Global state tensor of shape ``(state_dim,)`` or
            ``(batch, state_dim)``.

        Returns
        -------
        values : torch.Tensor — scalar or shape ``(batch,)``
        """
        return self.critic(state)

    # ------------------------------------------------------------------
    # Evaluation (with gradients, used in the update step)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        agent_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under the current policy for the PPO update.

        Parameters
        ----------
        obs:
            Local observations of shape ``(batch, obs_dim)``.
        actions:
            Actions of shape ``(batch, action_dim)``.
        state:
            Global states of shape ``(batch, state_dim)``.
        agent_idx:
            Actor index (ignored when sharing parameters).

        Returns
        -------
        log_probs : torch.Tensor — shape ``(batch,)``
        entropy   : torch.Tensor — shape ``(batch,)``
        values    : torch.Tensor — shape ``(batch,)``
        """
        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate_actions(obs, actions)
        values = self.critic(state)
        return log_probs, entropy, values

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict[str, int]:
        """Return a dict with actor and critic parameter counts."""
        if self.share_parameters:
            actor_params = sum(p.numel() for p in self.actor.parameters())  # type: ignore[attr-defined]
        else:
            actor_params = sum(
                p.numel() for actor in self.actors for p in actor.parameters()  # type: ignore[attr-defined]
            )
        critic_params = sum(p.numel() for p in self.critic.parameters())
        return {
            "actor": actor_params,
            "critic": critic_params,
            "total": actor_params + critic_params,
        }
