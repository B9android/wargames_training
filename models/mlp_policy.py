# SPDX-License-Identifier: MIT
# models/mlp_policy.py
"""Simple MLP actor-critic policy for BattalionEnv.

Uses Stable-Baselines3's ``ActorCriticPolicy`` as the base class with a
configurable two-hidden-layer MLP architecture suited for the 12-dimensional
observation space of BattalionEnv.

Typical usage::

    from models.mlp_policy import BattalionMlpPolicy
    from stable_baselines3 import PPO
    from envs.battalion_env import BattalionEnv

    env = BattalionEnv()
    model = PPO(BattalionMlpPolicy, env)
    model.learn(total_timesteps=10_000)
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

__all__ = ["BattalionMlpPolicy"]


class BattalionMlpPolicy(ActorCriticPolicy):
    """MLP actor-critic policy for BattalionEnv.

    A configurable fully-connected network with separate actor and critic
    heads, designed for the 12-dimensional observation space of
    :class:`~envs.battalion_env.BattalionEnv`.

    Architecture (default)::

        obs(12) → Linear(128) → Tanh → Linear(128) → Tanh
                ↓                                         ↓
        actor head → action(3) + log_std        critic head → value(1)

    Parameters
    ----------
    observation_space:
        Gymnasium observation space.
    action_space:
        Gymnasium action space.
    lr_schedule:
        Learning-rate schedule passed in by the SB3 algorithm.
    net_arch:
        Hidden-layer sizes shared by actor and critic.  Defaults to
        ``[128, 128]``.
    activation_fn:
        Activation function class applied after each hidden layer.
        Defaults to :class:`torch.nn.Tanh`.
    **kwargs:
        Forwarded to
        :class:`~stable_baselines3.common.policies.ActorCriticPolicy`.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        **kwargs: Any,
    ) -> None:
        if net_arch is None:
            net_arch = [128, 128]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )
