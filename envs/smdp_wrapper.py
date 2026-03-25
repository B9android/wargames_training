# SPDX-License-Identifier: MIT
# envs/smdp_wrapper.py
"""SMDP Wrapper — wraps MultiBattalionEnv with semi-MDP / Options framework.

The :class:`SMDPWrapper` replaces primitive continuous actions with
**macro-actions** (options).  Each macro-action is an :class:`~envs.options.Option`
from the six-element vocabulary defined in :mod:`envs.options`.

At every *macro-step*:

1. Each agent selects a macro-action (integer index 0–5).
2. The wrapper executes the chosen option by running primitive steps until
   the option's termination condition fires, the hard time-limit is reached,
   or the underlying environment terminates the agent.
3. Rewards are aggregated across all primitive steps and returned as a
   single scalar per agent.

The wrapper is a fully-compliant PettingZoo ``ParallelEnv`` and passes
``parallel_api_test(SMDPWrapper(MultiBattalionEnv()))``.

Typical usage::

    from envs.multi_battalion_env import MultiBattalionEnv
    from envs.smdp_wrapper import SMDPWrapper

    env = SMDPWrapper(MultiBattalionEnv(n_blue=2, n_red=2))
    obs, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample()
                   for agent in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)

Temporal abstraction
--------------------
Every ``infos[agent]`` dict contains a ``"temporal_abstraction"`` entry::

    {
        "macro_steps":    int,   # macro-steps completed this episode
        "primitive_steps": int,  # primitive steps completed this episode
        "ratio":          float, # macro_steps / primitive_steps
        "option_name":    str,   # name of the option just executed
        "option_steps":   int,   # number of primitive steps this option ran
    }

This can be logged directly to W&B::

    for agent, info in infos.items():
        wandb.log(info["temporal_abstraction"])
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from envs.multi_battalion_env import MultiBattalionEnv
from envs.options import Option, make_default_options

__all__ = ["SMDPWrapper"]

# Number of primitive actions (move, rotate, fire)
_PRIM_ACTION_DIM: int = 3


class SMDPWrapper(ParallelEnv):
    """PettingZoo ParallelEnv wrapping MultiBattalionEnv with SMDP options.

    Parameters
    ----------
    env:
        The underlying :class:`~envs.multi_battalion_env.MultiBattalionEnv`
        instance.
    options:
        Option vocabulary — a list of :class:`~envs.options.Option` objects.
        Index ``i`` corresponds to macro-action ``i``.  When ``None``,
        :func:`~envs.options.make_default_options` is used to build the
        standard six-element vocabulary.
    """

    metadata: dict = {"render_modes": [], "name": "smdp_multi_battalion_v0"}

    def __init__(
        self,
        env: MultiBattalionEnv,
        options: Optional[list[Option]] = None,
    ) -> None:
        if options is None:
            options = make_default_options()

        if len(options) == 0:
            raise ValueError(
                "options must contain at least one Option; received an empty list."
            )

        self._env = env
        self._options: list[Option] = list(options)
        self.n_options: int = len(self._options)

        # ------------------------------------------------------------------
        # PettingZoo required attributes
        # ------------------------------------------------------------------
        self.possible_agents: list[str] = list(env.possible_agents)
        self.agents: list[str] = []
        self.render_mode: Optional[str] = getattr(env, "render_mode", None)

        # ------------------------------------------------------------------
        # Spaces
        # Observation space is the same as the underlying environment;
        # action space is Discrete(n_options).
        # ------------------------------------------------------------------
        self._obs_spaces: dict[str, spaces.Space] = {
            a: env.observation_space(a) for a in self.possible_agents
        }
        self._act_space: spaces.Discrete = spaces.Discrete(self.n_options)

        # ------------------------------------------------------------------
        # Episode-level counters
        # ------------------------------------------------------------------
        self._macro_steps: int = 0
        self._primitive_steps: int = 0

        # Last observations keyed by agent (populated by reset / step)
        self._last_obs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # PettingZoo API: spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        """Return the observation space for *agent* (same as underlying env)."""
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        """Return the macro-action space for *agent* — ``Discrete(n_options)``."""
        return self._act_space

    # ------------------------------------------------------------------
    # Temporal abstraction property
    # ------------------------------------------------------------------

    @property
    def temporal_abstraction_ratio(self) -> float:
        """Macro-steps / primitive-steps for the current episode.

        Returns ``0.0`` before the first macro-step completes.
        """
        if self._primitive_steps == 0:
            return 0.0
        return self._macro_steps / self._primitive_steps

    # ------------------------------------------------------------------
    # PettingZoo API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment and return initial observations.

        Parameters
        ----------
        seed:
            RNG seed forwarded to the underlying environment.
        options:
            Currently unused; accepted for API compatibility.

        Returns
        -------
        observations : dict[agent_id, np.ndarray]
        infos        : dict[agent_id, dict]
        """
        obs, infos = self._env.reset(seed=seed, options=options)
        self.agents = list(self._env.agents)
        self._macro_steps = 0
        self._primitive_steps = 0
        self._last_obs = {a: o.copy() for a, o in obs.items()}
        return obs, infos

    # ------------------------------------------------------------------
    # PettingZoo API: step
    # ------------------------------------------------------------------

    def step(
        self,
        macro_actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Execute one macro-step.

        Runs each agent's chosen option for multiple primitive steps until
        **all** active options have terminated (option termination condition
        fires, hard time-limit reached, or the underlying env terminates the
        agent).

        Parameters
        ----------
        macro_actions:
            Dict mapping each live agent ID to a macro-action index
            ``[0, n_options)``.  Missing agents default to index ``0``
            (``advance_sector``).

        Returns
        -------
        observations, aggregate_rewards, terminated, truncated, infos
            Keyed by agent IDs that were alive at the *start* of this
            macro-step, matching the PettingZoo convention.
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents: list[str] = list(self.agents)

        # ------------------------------------------------------------------
        # Initialise per-agent tracking for this macro-step
        # ------------------------------------------------------------------
        selected_options: dict[str, Option] = {}
        option_steps: dict[str, int] = {}
        aggregate_rewards: dict[str, float] = {a: 0.0 for a in current_agents}
        option_done: dict[str, bool] = {a: False for a in current_agents}

        for agent in current_agents:
            # Missing agents default to option index 0 (e.g., "advance_sector").
            if agent not in macro_actions:
                idx = 0
            else:
                raw_idx = macro_actions[agent]
                idx = int(raw_idx)
                if idx < 0 or idx >= self.n_options:
                    raise ValueError(
                        f"Invalid macro-action index {idx!r} for agent {agent!r}; "
                        f"expected integer in [0, {self.n_options - 1}] or omit "
                        f"the agent key to use the default option 0."
                    )
            selected_options[agent] = self._options[idx]
            option_steps[agent] = 0

        # Initialise final output dicts with safe defaults
        final_obs: dict[str, np.ndarray] = {
            a: self._last_obs[a].copy() for a in current_agents
        }
        final_terminated: dict[str, bool] = {a: False for a in current_agents}
        final_truncated: dict[str, bool] = {a: False for a in current_agents}
        final_infos: dict[str, dict] = {a: {} for a in current_agents}

        # ------------------------------------------------------------------
        # Inner primitive-step loop
        # Run until all current_agents have finished their options OR
        # the underlying environment has no more live agents.
        # ------------------------------------------------------------------
        while any(not option_done[a] for a in current_agents):
            # Exit if the underlying env has exhausted all agents
            if not self._env.agents:
                for agent in current_agents:
                    option_done[agent] = True
                break

            # Build primitive actions for all env-live agents
            prim_actions: dict[str, np.ndarray] = {}
            for agent in self._env.agents:
                if agent in current_agents and not option_done[agent]:
                    prim_actions[agent] = selected_options[agent].get_action(
                        self._last_obs[agent]
                    )
                else:
                    # No-op for agents whose option has already terminated
                    prim_actions[agent] = np.zeros(_PRIM_ACTION_DIM, dtype=np.float32)

            # Primitive step
            obs, rewards, terminated, truncated, infos = self._env.step(prim_actions)
            self._primitive_steps += 1

            # ----------------------------------------------------------------
            # Process results for each agent alive at macro-step start
            # ----------------------------------------------------------------
            for agent in current_agents:
                if option_done[agent]:
                    continue

                # Accumulate reward
                if agent in rewards:
                    aggregate_rewards[agent] += float(rewards[agent])

                # Update latest observation
                if agent in obs:
                    self._last_obs[agent] = obs[agent].copy()
                    final_obs[agent] = obs[agent].copy()

                # Update info (keep the latest primitive info)
                if agent in infos:
                    final_infos[agent] = dict(infos[agent])

                # Check underlying-env termination first
                env_terminated = terminated.get(agent, False)
                env_truncated = truncated.get(agent, False)
                if env_terminated or env_truncated:
                    final_terminated[agent] = bool(env_terminated)
                    final_truncated[agent] = bool(env_truncated)
                    option_done[agent] = True
                else:
                    # Increment option step counter and check option termination
                    option_steps[agent] += 1
                    if selected_options[agent].should_terminate(
                        self._last_obs[agent], option_steps[agent]
                    ):
                        option_done[agent] = True

        # ------------------------------------------------------------------
        # Post-loop: handle agents that the env removed without explicit
        # terminated/truncated signals (edge case when the underlying env
        # empties mid-option and the agent was never seen in a terminated
        # dict during this macro-step).  Treating them as truncated preserves
        # PettingZoo semantics — the agent was alive at macro-step start but
        # absent at the end because the episode was cut short by the env.
        # ------------------------------------------------------------------
        for agent in current_agents:
            if (
                agent not in self._env.agents
                and not final_terminated[agent]
                and not final_truncated[agent]
            ):
                final_truncated[agent] = True

        self._macro_steps += 1

        # Update wrapper-level agent list from underlying env
        self.agents = list(self._env.agents)

        # ------------------------------------------------------------------
        # Attach temporal abstraction metadata to every agent's info dict
        # ------------------------------------------------------------------
        ta_ratio = self.temporal_abstraction_ratio
        for agent in current_agents:
            final_infos[agent]["temporal_abstraction"] = {
                "macro_steps": self._macro_steps,
                "primitive_steps": self._primitive_steps,
                "ratio": ta_ratio,
                "option_name": selected_options[agent].name,
                "option_steps": option_steps.get(agent, 0),
            }

        return final_obs, aggregate_rewards, final_terminated, final_truncated, final_infos

    # ------------------------------------------------------------------
    # PettingZoo API: state (delegated)
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Return the global state tensor from the underlying environment."""
        return self._env.state()

    # ------------------------------------------------------------------
    # PettingZoo API: render / close (delegated)
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Delegate rendering to the underlying environment."""
        return self._env.render()

    def close(self) -> None:
        """Delegate resource cleanup to the underlying environment."""
        return self._env.close()
