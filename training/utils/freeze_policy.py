# training/utils/freeze_policy.py
"""Policy freezing utilities for the E3.4 Hierarchical Curriculum.

These helpers implement the **freeze** half of the bottom-up training
protocol: once a lower-level policy has been promoted it must be frozen
(no gradient updates) so that the higher-level agent trains against a
stable, fixed foundation.

Three concrete use-cases are covered:

1. **MAPPO battalion policy** — loaded from a ``.pt`` checkpoint and used
   inside :class:`~envs.brigade_env.BrigadeEnv` to drive Red battalions.
2. **SB3 PPO brigade policy** — loaded from a ``.zip`` checkpoint and used
   inside :class:`~envs.division_env.DivisionEnv` to drive Red brigades.
3. **Generic PyTorch module** — any ``torch.nn.Module`` can be frozen with
   :func:`freeze_mappo_policy` by passing it directly.

Usage::

    from training.utils.freeze_policy import (
        freeze_mappo_policy,
        freeze_sb3_policy,
        assert_frozen,
    )

    # Freeze an already-loaded MAPPOPolicy (in-place)
    freeze_mappo_policy(policy)
    assert_frozen(policy)

    # Freeze an already-loaded SB3 PPO model (in-place)
    freeze_sb3_policy(ppo_model)
    assert_frozen(ppo_model.policy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = [
    "freeze_mappo_policy",
    "freeze_sb3_policy",
    "assert_frozen",
    "load_and_freeze_mappo",
    "load_and_freeze_sb3",
]


# ---------------------------------------------------------------------------
# Core freeze helpers
# ---------------------------------------------------------------------------


def freeze_mappo_policy(module: nn.Module) -> nn.Module:
    """Freeze all parameters of a PyTorch module **in-place**.

    Sets ``requires_grad=False`` on every parameter and switches the module
    to ``eval()`` mode so that batch-norm / dropout layers also behave as
    expected during inference.

    Parameters
    ----------
    module:
        Any :class:`torch.nn.Module` — typically a
        :class:`~models.mappo_policy.MAPPOPolicy` instance.

    Returns
    -------
    torch.nn.Module
        The same *module* object (mutated in-place) for convenient chaining.
    """
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    n_params = sum(p.numel() for p in module.parameters())
    log.info(
        "freeze_mappo_policy: frozen %d parameters in %s",
        n_params,
        type(module).__name__,
    )
    return module


def freeze_sb3_policy(ppo_model: object) -> object:
    """Freeze the internal policy network of a Stable-Baselines3 model.

    Iterates over ``ppo_model.policy.parameters()`` and sets
    ``requires_grad=False`` on each, then calls ``ppo_model.policy.eval()``.

    Parameters
    ----------
    ppo_model:
        A loaded SB3 ``PPO`` (or ``A2C``, etc.) model whose ``.policy``
        attribute is a :class:`torch.nn.Module`.

    Returns
    -------
    object
        The same *ppo_model* (mutated in-place) for convenient chaining.

    Raises
    ------
    AttributeError
        If *ppo_model* does not have a ``.policy`` attribute.
    """
    if not hasattr(ppo_model, "policy"):
        raise AttributeError(
            f"Expected an SB3 model with a .policy attribute, "
            f"got {type(ppo_model).__name__}"
        )
    for param in ppo_model.policy.parameters():
        param.requires_grad_(False)
    ppo_model.policy.eval()
    n_params = sum(p.numel() for p in ppo_model.policy.parameters())
    log.info(
        "freeze_sb3_policy: frozen %d parameters in %s.policy",
        n_params,
        type(ppo_model).__name__,
    )
    return ppo_model


def assert_frozen(module: nn.Module) -> None:
    """Assert that *module* has **no** trainable parameters.

    Parameters
    ----------
    module:
        Any :class:`torch.nn.Module`.

    Raises
    ------
    AssertionError
        If any parameter has ``requires_grad=True``.
    """
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert trainable == 0, (
        f"Expected 0 trainable parameters in {type(module).__name__}, "
        f"got {trainable}."
    )


# ---------------------------------------------------------------------------
# Convenience load-and-freeze helpers
# ---------------------------------------------------------------------------


def load_and_freeze_mappo(
    checkpoint_path: Union[str, Path],
    obs_dim: int,
    action_dim: int,
    state_dim: int,
    n_agents: int = 2,
    device: str = "cpu",
) -> "MAPPOPolicy":  # noqa: F821  (forward reference OK at runtime)
    """Load a MAPPO ``.pt`` checkpoint and return a frozen policy.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``mappo_policy*.pt`` file produced by
        :mod:`training.train_mappo`.
    obs_dim:
        Per-agent observation dimensionality (must match checkpoint).
    action_dim:
        Action dimensionality (must match checkpoint).
    state_dim:
        Global state dimensionality (must match checkpoint).
    n_agents:
        Number of agents the policy was trained with.
    device:
        PyTorch device string.

    Returns
    -------
    MAPPOPolicy
        Loaded, frozen MAPPO policy on *device*.
    """
    from models.mappo_policy import MAPPOPolicy  # local import to keep module importable

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MAPPO checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=n_agents,
        share_parameters=True,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy = policy.to(device)
    freeze_mappo_policy(policy)
    assert_frozen(policy)
    log.info(
        "load_and_freeze_mappo: loaded from %s (step=%d)",
        checkpoint_path,
        checkpoint.get("total_steps", -1),
    )
    return policy


def load_and_freeze_sb3(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
) -> "PPO":  # noqa: F821  (forward reference OK at runtime)
    """Load an SB3 PPO ``.zip`` checkpoint and return a frozen model.

    Parameters
    ----------
    checkpoint_path:
        Path to an SB3 PPO ``.zip`` checkpoint.
    device:
        PyTorch device string.

    Returns
    -------
    PPO
        Loaded SB3 PPO model with frozen policy parameters.
    """
    from stable_baselines3 import PPO  # local import

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SB3 checkpoint not found: {checkpoint_path}")

    model = PPO.load(str(checkpoint_path), device=device)
    freeze_sb3_policy(model)
    assert_frozen(model.policy)
    log.info("load_and_freeze_sb3: loaded from %s", checkpoint_path)
    return model
