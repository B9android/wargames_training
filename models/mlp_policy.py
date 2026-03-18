# models/mlp_policy.py
"""Feedforward MLP policy for battalion RL training.

``BattalionMlpPolicy`` is a Stable-Baselines3-compatible custom policy
designed for the 12-dimensional battalion observation space.

Architecture
~~~~~~~~~~~~
* **Shared feature extractor** — ``obs_dim → 128 → 64`` (Linear + ReLU +
  LayerNorm at each hidden layer).
* **Actor head** — ``64 → 32 → action_dim`` with a final ``nn.Tanh``
  activation appended in ``_build()`` to bound action means to [-1, 1].
* **Critic head** — ``64 → 32 → 1``.

The policy is registered in ``PPO.policy_aliases`` so it can be passed by
name::

    from stable_baselines3 import PPO
    import models.mlp_policy  # triggers registration as a side-effect

    model = PPO("BattalionMlpPolicy", env)

It can also be passed directly as a class::

    from models.mlp_policy import BattalionMlpPolicy
    model = PPO(BattalionMlpPolicy, env)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

__all__ = ["BattalionFeaturesExtractor", "BattalionMlpPolicy"]

# ---------------------------------------------------------------------------
# Shared feature extractor — obs_dim → 128 → 64
# ---------------------------------------------------------------------------


class BattalionFeaturesExtractor(BaseFeaturesExtractor):
    """Two-layer MLP trunk shared by actor and critic.

    Parameters
    ----------
    observation_space:
        The gymnasium observation space (must be a flat ``Box``).
    features_dim:
        Dimensionality of the output embedding (default: ``64``).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: Any) -> Any:  # type: ignore[override]
        return self.net(observations)


# ---------------------------------------------------------------------------
# Custom ActorCritic policy
# ---------------------------------------------------------------------------


class BattalionMlpPolicy(ActorCriticPolicy):
    """SB3 ``ActorCriticPolicy`` with a battalion-tuned MLP architecture.

    Network layout
    ~~~~~~~~~~~~~~
    * Feature extractor: ``obs_dim → 128 → 64`` (ReLU + LayerNorm)
    * Actor net: ``64 → 32 → action_dim`` (Tanh-squashed output)
    * Critic net: ``64 → 32 → 1``

    All keyword arguments are forwarded to ``ActorCriticPolicy`` so the
    policy remains fully configurable by the caller.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("features_extractor_class", BattalionFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"features_dim": 64})
        # Separate 32-unit hidden layers for actor (pi) and critic (vf).
        kwargs.setdefault("net_arch", {"pi": [32], "vf": [32]})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build the networks and append Tanh to the action mean head."""
        super()._build(lr_schedule)
        # Wrap the action mean network with a Tanh activation so the
        # predicted mean always lies in [-1, 1], matching the action space.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


# ---------------------------------------------------------------------------
# Registration — enables PPO("BattalionMlpPolicy", env)
# ---------------------------------------------------------------------------

PPO.policy_aliases["BattalionMlpPolicy"] = BattalionMlpPolicy
