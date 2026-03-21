# envs/remote_multi_battalion_env.py
"""Ray remote actor wrapper for :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

Exposes ``MultiBattalionEnv`` as a Ray remote actor so that multiple
environment instances can be driven in parallel across workers without
requiring a shared-memory PettingZoo vectorization layer.

Typical usage::

    import ray
    from envs.remote_multi_battalion_env import RemoteMultiBattalionEnv

    ray.init()
    env = RemoteMultiBattalionEnv.remote(n_blue=2, n_red=2)
    obs_ref, info_ref = ray.get(env.reset.remote(seed=0))
    obs, info = ray.get(obs_ref), ray.get(info_ref)

Or use the convenience factory :func:`make_remote_envs` to spin up a
pool of workers::

    envs = make_remote_envs(num_envs=4, n_blue=2, n_red=2)
    refs = [e.reset.remote(seed=i) for i, e in enumerate(envs)]
    results = ray.get(refs)

Classes
-------
RemoteMultiBattalionEnv
    Ray ``@remote`` actor wrapping ``MultiBattalionEnv``.

Functions
---------
make_remote_envs
    Factory that creates *num_envs* remote environment actors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import ray
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ray is required for RemoteMultiBattalionEnv. "
        "Install it with: pip install 'ray[rllib]>=2.9.0'"
    ) from exc

from envs.multi_battalion_env import MultiBattalionEnv

__all__ = ["RemoteMultiBattalionEnv", "make_remote_envs"]


@ray.remote
class RemoteMultiBattalionEnv:
    """Ray remote actor wrapping :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

    All ``MultiBattalionEnv`` constructor parameters are forwarded as-is.
    The actor holds a single environment instance and exposes its public
    interface as remote methods.

    Parameters
    ----------
    **kwargs:
        Keyword arguments forwarded directly to
        :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

    Notes
    -----
    * Each actor runs in its own process, giving true parallelism.
    * The actor is **not** thread-safe — only one caller should be active
      at a time per actor.
    * ``observation_space`` / ``action_space`` return
      :class:`gymnasium.spaces.Box` objects serialised via pickle (Ray's
      default object serialiser), so they are safe to pass across the
      object store.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._env = MultiBattalionEnv(**kwargs)

    # ------------------------------------------------------------------
    # Core PettingZoo interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment and return initial observations and infos."""
        return self._env.reset(seed=seed, options=options)

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Advance the environment by one step.

        Returns
        -------
        observations, rewards, terminations, truncations, infos
        """
        return self._env.step(actions)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Return the global state tensor (for centralized critics)."""
        return self._env.state()

    @property
    def agents(self) -> List[str]:
        """Currently active agents."""
        return self._env.agents

    def get_agents(self) -> List[str]:
        """Return the list of currently active agents (remote-callable)."""
        return self._env.agents

    def get_possible_agents(self) -> List[str]:
        """Return the list of all possible agent IDs."""
        return self._env.possible_agents

    def get_n_blue(self) -> int:
        """Return the number of Blue battalions."""
        return self._env.n_blue

    def get_n_red(self) -> int:
        """Return the number of Red battalions."""
        return self._env.n_red

    def observation_space(self, agent: str):
        """Return the observation space for *agent*."""
        return self._env.observation_space(agent)

    def action_space(self, agent: str):
        """Return the action space for *agent*."""
        return self._env.action_space(agent)

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()

    def run_episode(
        self,
        policy_fn=None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete episode and return a summary dict.

        Parameters
        ----------
        policy_fn:
            Callable ``(observations) -> actions`` mapping agent observations
            to actions.  If ``None``, random actions are sampled from each
            agent's action space.
        seed:
            RNG seed forwarded to :meth:`reset`.
        max_steps:
            Maximum number of steps before truncating.  Defaults to the
            environment's own ``max_steps``.

        Returns
        -------
        dict with keys:
            ``steps`` (int), ``total_reward`` (float), ``blue_won`` (bool),
            ``observations`` (final obs dict), ``agent_rewards`` (per-agent totals).
        """
        obs, _ = self._env.reset(seed=seed)
        effective_max_steps = max_steps or self._env.max_steps
        total_rewards: Dict[str, float] = {}
        steps = 0
        blue_won = False

        for _ in range(effective_max_steps):
            if not self._env.agents:
                break

            if policy_fn is not None:
                actions = policy_fn(obs)
            else:
                actions = {
                    agent: self._env.action_space(agent).sample()
                    for agent in self._env.agents
                }

            obs, rewards, terminations, truncations, _ = self._env.step(actions)
            steps += 1

            for agent, r in rewards.items():
                total_rewards[agent] = total_rewards.get(agent, 0.0) + r

            # Detect Blue win: all reds gone, at least one blue alive
            alive = self._env.agents
            blues_alive = any(a.startswith("blue_") for a in alive)
            reds_alive = any(a.startswith("red_") for a in alive)
            if blues_alive and not reds_alive:
                blue_won = True

            if all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in (list(terminations) + list(truncations))
            ):
                break

        return {
            "steps": steps,
            "total_reward": sum(total_rewards.values()),
            "agent_rewards": total_rewards,
            "blue_won": blue_won,
            "observations": obs,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_remote_envs(
    num_envs: int,
    num_cpus_per_env: float = 1.0,
    **env_kwargs: Any,
) -> List[ray.actor.ActorHandle]:
    """Create a pool of :class:`RemoteMultiBattalionEnv` Ray actors.

    Parameters
    ----------
    num_envs:
        Number of remote environment actors to create.
    num_cpus_per_env:
        CPU fractional allocation per actor (passed to Ray's ``num_cpus``
        resource hint).  Use a fraction (e.g. ``0.5``) to over-subscribe
        CPUs on a single machine during development.
    **env_kwargs:
        Keyword arguments forwarded to
        :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

    Returns
    -------
    list of :class:`ray.actor.ActorHandle`
        Ready-to-use remote actor handles.

    Raises
    ------
    RuntimeError
        If Ray has not been initialised before calling this function.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray must be initialized before calling make_remote_envs(). "
            "Call ray.init() first."
        )
    actor_cls = RemoteMultiBattalionEnv.options(num_cpus=num_cpus_per_env)
    return [actor_cls.remote(**env_kwargs) for _ in range(num_envs)]
