# models/mlp_policy.py
"""Custom MLP policy for the BattalionEnv.

Provides :class:`BattalionFeaturesExtractor` (a small feature-encoding MLP)
and :class:`BattalionMlpPolicy` (an SB3 ``ActorCriticPolicy`` subclass that
uses it as the shared feature backbone).

Typical usage with Stable-Baselines3::

    from stable_baselines3 import PPO
    from models.mlp_policy import BattalionMlpPolicy

    model = PPO(BattalionMlpPolicy, env, verbose=1)
    model.learn(total_timesteps=500_000)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class BattalionFeaturesExtractor(BaseFeaturesExtractor):
    """Encode the 12-dimensional BattalionEnv observation into a feature vector.

    Architecture: Linear(obs_dim → 128) → ReLU → LayerNorm →
                  Linear(128 → 64) → ReLU → LayerNorm

    Parameters
    ----------
    observation_space:
        The environment's observation space (``Box(12,)``).
    features_dim:
        Size of the output feature vector (default 64).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)

        obs_dim = int(observation_space.shape[0])

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class BattalionMlpPolicy(ActorCriticPolicy):
    """Actor-critic policy for BattalionEnv, backed by :class:`BattalionFeaturesExtractor`.

    The feature extractor encodes the 12-dim observation into a 64-dim
    feature vector.  The actor and critic heads are each a two-layer MLP
    (64 → 32 → output) with ``Tanh`` activations.

    Parameters
    ----------
    observation_space, action_space, lr_schedule:
        Standard SB3 policy arguments.
    net_arch:
        Optional override for the actor/critic head architecture.
        Defaults to ``dict(pi=[32], vf=[32])``.
    features_extractor_class:
        Defaults to :class:`BattalionFeaturesExtractor`.
    features_extractor_kwargs:
        Kwargs forwarded to the features extractor; default ``{"features_dim": 64}``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Dict]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = BattalionFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        if net_arch is None:
            net_arch = dict(pi=[32], vf=[32])
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": 64}

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Register in PPO policy aliases so users can pass the string "BattalionMlpPolicy"
# ---------------------------------------------------------------------------

try:
    from stable_baselines3 import PPO

    PPO.policy_aliases["BattalionMlpPolicy"] = BattalionMlpPolicy
except Exception:  # pragma: no cover
    pass
