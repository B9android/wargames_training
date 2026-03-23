# training/human_feedback.py
"""After-Action Review (AAR) & Training Feedback Loop (Epic E9.3).

Provides utilities for capturing human player decisions, annotating them with
AI-predicted quality scores, and feeding disagreements back into training as
demonstration data via **DAgger** (Dataset Aggregation) and **GAIL**
(Generative Adversarial Imitation Learning).

Key components
--------------
``HumanDemonstration``
    Immutable dataclass representing a single recorded (obs, action, …) step.

``DemonstrationBuffer``
    Fixed-capacity ring buffer that stores :class:`HumanDemonstration` objects
    and exposes efficient random-sampling for supervised learning updates.
    Can be serialised to / deserialised from a NumPy ``.npz`` archive.

``HumanFeedbackRecorder``
    Wraps a :class:`~envs.battalion_env.BattalionEnv` (or any ``gymnasium.Env``)
    and records one episode of human (or policy-driven) gameplay into a
    :class:`DemonstrationBuffer`.

``DAggerTrainer``
    Implements the **DAgger** (Ross et al. 2011) algorithm over a gymnasium
    environment.  Each iteration rolls-out the current trained policy while
    simultaneously querying an *expert policy* to label the states; the
    resulting (state, expert-action) pairs are aggregated into the buffer and
    used for supervised behavioural-cloning updates.

``GAILDiscriminator``
    Two-layer MLP (PyTorch ``nn.Module``) that distinguishes human
    demonstrations from policy-generated rollouts.  The learned discriminator
    score is converted to an intrinsic reward signal via the GAIL reward
    formula ``−log(1 − D(s, a))``.

``GAILRewardWrapper``
    ``gymnasium.Wrapper`` that replaces (or augments) the environment's
    extrinsic reward with the GAIL intrinsic reward produced by a
    :class:`GAILDiscriminator`.

``AARAnnotator``
    After-Action Review engine.  Given a recorded episode it uses a reference
    policy to score each decision, identifies *decisive turning points* where
    the human's choice diverged most from the AI recommendation, and returns
    an :class:`AARReport` containing a timeline of :class:`DecisionAnnotation`
    objects plus a list of :class:`TurningPoint` objects.

Typical usage — DAgger
~~~~~~~~~~~~~~~~~~~~~~
::

    from envs.battalion_env import BattalionEnv
    from training.human_feedback import (
        DemonstrationBuffer, DAggerTrainer,
    )
    from stable_baselines3 import PPO

    env = BattalionEnv(max_steps=200)
    buffer = DemonstrationBuffer(obs_dim=17, action_dim=3, capacity=50_000)
    policy = PPO("MlpPolicy", env, verbose=0)
    expert = PPO("MlpPolicy", env, verbose=0)  # pre-trained expert

    trainer = DAggerTrainer(
        env=env,
        buffer=buffer,
        obs_dim=17,
        action_dim=3,
        lr=1e-3,
        batch_size=64,
    )
    trainer.train(n_iterations=20, expert_policy=expert)

Typical usage — GAIL
~~~~~~~~~~~~~~~~~~~~
::

    from training.human_feedback import GAILDiscriminator, GAILRewardWrapper

    disc = GAILDiscriminator(obs_dim=17, action_dim=3)
    wrapped = GAILRewardWrapper(env, discriminator=disc, gail_coef=1.0)

Typical usage — AAR
~~~~~~~~~~~~~~~~~~~
::

    from training.human_feedback import HumanFeedbackRecorder, AARAnnotator

    recorder = HumanFeedbackRecorder(env, buffer)
    recorder.record_episode(policy=human_policy)
    episode = recorder.last_episode

    annotator = AARAnnotator(reference_policy=ai_policy)
    report = annotator.annotate(episode)
    for tp in report.turning_points:
        print(tp.step_id, tp.divergence_score)
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)

__all__ = [
    # Data structures
    "HumanDemonstration",
    "DecisionAnnotation",
    "TurningPoint",
    "AARReport",
    # Core components
    "DemonstrationBuffer",
    "HumanFeedbackRecorder",
    "DAggerTrainer",
    "GAILDiscriminator",
    "GAILRewardWrapper",
    "AARAnnotator",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HumanDemonstration:
    """A single recorded step from a human (or expert) gameplay session.

    Parameters
    ----------
    observation:
        Environment observation **before** the action was taken,
        shape ``(obs_dim,)``.
    action:
        Action chosen by the human / expert, shape ``(action_dim,)``.
    reward:
        Scalar reward received after the action.
    next_observation:
        Environment observation **after** the action, shape ``(obs_dim,)``.
    terminated:
        Whether the episode ended due to a terminal condition.
    truncated:
        Whether the episode ended due to a step-limit truncation.
    episode_id:
        Zero-based episode index within the current recording session.
    step_id:
        Zero-based step index within the episode.
    info:
        Additional info dict returned by the environment step.
    """

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int
    info: Dict[str, Any] = dataclasses.field(default_factory=dict, compare=False)


@dataclasses.dataclass
class DecisionAnnotation:
    """Quality annotation for a single human decision.

    Parameters
    ----------
    step_id:
        Step index within the episode.
    human_action:
        Action actually taken by the human, shape ``(action_dim,)``.
    ai_action:
        Action recommended by the reference AI policy, shape ``(action_dim,)``.
    ai_value:
        Estimated state value (critic output) under the reference policy.
    action_divergence:
        Euclidean distance between *human_action* and *ai_action*; large values
        indicate a significant disagreement.
    quality_score:
        Normalised quality score in ``[0, 1]`` estimated from the AI critic;
        higher is better for the human.
    """

    step_id: int
    human_action: np.ndarray
    ai_action: np.ndarray
    ai_value: float
    action_divergence: float
    quality_score: float


@dataclasses.dataclass
class TurningPoint:
    """A decisive moment in the episode where the human diverged from the AI.

    Parameters
    ----------
    step_id:
        Step index within the episode.
    divergence_score:
        How much the human diverged from the AI recommendation at this point
        (higher = more decisive).
    human_action:
        The action the human actually took.
    ai_action:
        The action the AI would have recommended.
    description:
        Human-readable description of the turning point.
    """

    step_id: int
    divergence_score: float
    human_action: np.ndarray
    ai_action: np.ndarray
    description: str = ""


@dataclasses.dataclass
class AARReport:
    """After-Action Review report for a single episode.

    Parameters
    ----------
    episode_id:
        Identifier of the reviewed episode.
    n_steps:
        Total number of steps in the episode.
    annotations:
        Per-step :class:`DecisionAnnotation` objects (one per step).
    turning_points:
        Subset of steps identified as *decisive* turning points, sorted by
        ``divergence_score`` descending.
    mean_quality_score:
        Average quality score across all annotated steps.
    agreement_rate:
        Fraction of steps where the human's action was within
        ``agreement_threshold`` of the AI's recommendation (default distance
        < 0.2 in action space).
    """

    episode_id: int
    n_steps: int
    annotations: List[DecisionAnnotation]
    turning_points: List[TurningPoint]
    mean_quality_score: float
    agreement_rate: float


# ---------------------------------------------------------------------------
# DemonstrationBuffer
# ---------------------------------------------------------------------------


class DemonstrationBuffer:
    """Fixed-capacity ring buffer of :class:`HumanDemonstration` objects.

    Internally the buffer stores observations, actions, rewards and done-flags
    as pre-allocated NumPy arrays to avoid repeated list-copy overhead during
    mini-batch sampling.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    action_dim:
        Dimensionality of the action vector.
    capacity:
        Maximum number of transitions stored before the oldest are overwritten
        (default 100 000).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 100_000,
    ) -> None:
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)

        self._obs: np.ndarray = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs: np.ndarray = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions: np.ndarray = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._terminated: np.ndarray = np.zeros(capacity, dtype=bool)
        self._truncated: np.ndarray = np.zeros(capacity, dtype=bool)
        self._episode_ids: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self._step_ids: np.ndarray = np.zeros(capacity, dtype=np.int64)

        self._ptr: int = 0       # write pointer (ring)
        self._size: int = 0      # current fill level

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, demo: HumanDemonstration) -> None:
        """Append a demonstration, evicting the oldest if at capacity.

        Parameters
        ----------
        demo:
            A single :class:`HumanDemonstration` step to add.
        """
        i = self._ptr
        self._obs[i] = np.asarray(demo.observation, dtype=np.float32)
        self._next_obs[i] = np.asarray(demo.next_observation, dtype=np.float32)
        self._actions[i] = np.asarray(demo.action, dtype=np.float32)
        self._rewards[i] = float(demo.reward)
        self._terminated[i] = bool(demo.terminated)
        self._truncated[i] = bool(demo.truncated)
        self._episode_ids[i] = int(demo.episode_id)
        self._step_ids[i] = int(demo.step_id)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, demos: Sequence[HumanDemonstration]) -> None:
        """Add a sequence of demonstrations in order.

        Parameters
        ----------
        demos:
            Iterable of :class:`HumanDemonstration` objects.
        """
        for demo in demos:
            self.add(demo)

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of ``(obs, actions, rewards)``.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        rng:
            Optional NumPy random generator for reproducible sampling.

        Returns
        -------
        obs : np.ndarray, shape ``(batch_size, obs_dim)``
        actions : np.ndarray, shape ``(batch_size, action_dim)``
        rewards : np.ndarray, shape ``(batch_size,)``

        Raises
        ------
        ValueError
            If the buffer contains fewer entries than *batch_size*.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} entries but batch_size={batch_size} requested."
            )
        _rng = rng if rng is not None else np.random.default_rng()
        idxs = _rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._obs[idxs].copy(),
            self._actions[idxs].copy(),
            self._rewards[idxs].copy(),
        )

    def sample_all(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all stored ``(obs, actions, rewards)`` as numpy arrays."""
        return (
            self._obs[: self._size].copy(),
            self._actions[: self._size].copy(),
            self._rewards[: self._size].copy(),
        )

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Remove all stored demonstrations."""
        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise the buffer to a compressed NumPy ``.npz`` archive.

        Parameters
        ----------
        path:
            File path (with or without the ``.npz`` extension).

        Returns
        -------
        Path
            Resolved path of the written file (always ends with ``.npz``).
        """
        p = Path(path)
        # Normalise extension so the returned path matches the file on disk.
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            obs=self._obs[: self._size],
            next_obs=self._next_obs[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            terminated=self._terminated[: self._size],
            truncated=self._truncated[: self._size],
            episode_ids=self._episode_ids[: self._size],
            step_ids=self._step_ids[: self._size],
            meta=np.array([self.obs_dim, self.action_dim, self.capacity]),
        )
        log.info("DemonstrationBuffer saved %d transitions → %s", self._size, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DemonstrationBuffer":
        """Deserialise a buffer previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` archive (with or without extension).

        Returns
        -------
        DemonstrationBuffer
            A new buffer instance populated with the saved transitions.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        data = np.load(str(p), allow_pickle=False)
        obs_dim = int(data["meta"][0])
        action_dim = int(data["meta"][1])
        capacity = int(data["meta"][2])
        buf = cls(obs_dim=obs_dim, action_dim=action_dim, capacity=capacity)
        n = len(data["obs"])
        buf._obs[:n] = data["obs"]
        buf._next_obs[:n] = data["next_obs"]
        buf._actions[:n] = data["actions"]
        buf._rewards[:n] = data["rewards"]
        buf._terminated[:n] = data["terminated"]
        buf._truncated[:n] = data["truncated"]
        buf._episode_ids[:n] = data["episode_ids"]
        buf._step_ids[:n] = data["step_ids"]
        buf._size = n
        buf._ptr = n % capacity
        log.info("DemonstrationBuffer loaded %d transitions ← %s", n, p)
        return buf


# ---------------------------------------------------------------------------
# HumanFeedbackRecorder
# ---------------------------------------------------------------------------


class HumanFeedbackRecorder:
    """Records gameplay transitions into a :class:`DemonstrationBuffer`.

    Wraps any ``gymnasium.Env`` and provides a simple API for running one
    episode (using an optional policy to drive actions) while recording every
    ``(obs, action, reward, next_obs, done)`` tuple.

    Parameters
    ----------
    env:
        A gymnasium-compatible environment.
    buffer:
        :class:`DemonstrationBuffer` to append recorded transitions to.
    """

    def __init__(self, env: Any, buffer: DemonstrationBuffer) -> None:
        self._env = env
        self._buffer = buffer
        self._episode_counter: int = 0
        self._last_episode: List[HumanDemonstration] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_episode(self) -> List[HumanDemonstration]:
        """List of :class:`HumanDemonstration` objects from the most recent episode."""
        return list(self._last_episode)

    def record_episode(
        self,
        policy: Optional[Any] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> List[HumanDemonstration]:
        """Run a full episode using *policy* (or random actions) and record it.

        Parameters
        ----------
        policy:
            Any object with a ``predict(obs, deterministic)`` method.  When
            ``None``, actions are sampled uniformly from the action space.
        seed:
            Optional random seed forwarded to ``env.reset()``.
        deterministic:
            Whether to use deterministic actions when *policy* is provided.

        Returns
        -------
        List[HumanDemonstration]
            All steps recorded during the episode.
        """
        obs, _ = self._env.reset(seed=seed)
        episode: List[HumanDemonstration] = []
        step_id = 0

        while True:
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                action = self._env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self._env.step(action)

            demo = HumanDemonstration(
                observation=np.array(obs, dtype=np.float32),
                action=np.array(action, dtype=np.float32),
                reward=float(reward),
                next_observation=np.array(next_obs, dtype=np.float32),
                terminated=bool(terminated),
                truncated=bool(truncated),
                episode_id=self._episode_counter,
                step_id=step_id,
                info=dict(info),
            )
            episode.append(demo)
            self._buffer.add(demo)

            obs = next_obs
            step_id += 1
            if terminated or truncated:
                break

        self._last_episode = episode
        self._episode_counter += 1
        log.debug(
            "HumanFeedbackRecorder: episode %d recorded (%d steps)",
            self._episode_counter - 1,
            len(episode),
        )
        return episode


# ---------------------------------------------------------------------------
# DAggerTrainer
# ---------------------------------------------------------------------------


class DAggerTrainer:
    """DAgger (Dataset Aggregation) trainer for imitation learning.

    Implements the DAgger algorithm (Ross et al., 2011):

    1. Initialise the dataset ``D`` with expert demonstrations.
    2. For *N* iterations:
       a. Roll out the **current trained policy** in the environment.
       b. For each visited state, query the **expert policy** to get the
          expert-recommended action and add ``(state, expert_action)`` to D.
       c. Train the policy on the full aggregated dataset D with supervised
          behavioural cloning (MSE loss on continuous actions).

    The trained policy is a simple two-hidden-layer MLP that maps observations
    to actions.  At evaluation time the policy's ``predict`` method mirrors the
    SB3 interface (returns ``(action, state)``).

    Parameters
    ----------
    env:
        A gymnasium-compatible environment.
    buffer:
        :class:`DemonstrationBuffer` pre-populated with initial demonstrations
        (may be empty; DAgger will aggregate from the first iteration).
    obs_dim:
        Observation dimensionality.
    action_dim:
        Action dimensionality.
    action_low:
        Lower bound of action space (scalar or array), used to clip outputs.
    action_high:
        Upper bound of action space (scalar or array), used to clip outputs.
    hidden_sizes:
        Hidden-layer sizes for the cloned policy MLP (default ``[128, 128]``).
    lr:
        Adam learning rate for supervised updates (default ``1e-3``).
    batch_size:
        Mini-batch size for supervised updates (default ``64``).
    n_grad_steps:
        Number of gradient steps per training iteration (default ``200``).
    device:
        PyTorch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        env: Any,
        buffer: DemonstrationBuffer,
        obs_dim: int,
        action_dim: int,
        action_low: float | np.ndarray = -1.0,
        action_high: float | np.ndarray = 1.0,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 1e-3,
        batch_size: int = 64,
        n_grad_steps: int = 200,
        device: str = "cpu",
    ) -> None:
        self._env = env
        self._buffer = buffer
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.n_grad_steps = int(n_grad_steps)
        self.device = torch.device(device)

        # Build the cloned-policy MLP.
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.policy_net: nn.Module = nn.Sequential(*layers).to(self.device)

        self._optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self._loss_fn = nn.MSELoss()

        # Training state
        self.total_grad_steps: int = 0
        self.last_bc_loss: float = float("nan")
        # Episode counter for DAgger rollouts (independent of buffer internals).
        self._dagger_episode_counter: int = 0

    # ------------------------------------------------------------------
    # Policy interface (SB3-compatible)
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """Return ``(action, None)`` for a given observation (SB3-like API).

        Parameters
        ----------
        obs:
            Observation array, shape ``(obs_dim,)`` or ``(B, obs_dim)``.
        deterministic:
            Unused; present for API compatibility.

        Returns
        -------
        Tuple[np.ndarray, None]
            Clipped action and ``None`` state.
        """
        self.policy_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.device
            )
            action_t = self.policy_net(obs_t)
        action = action_t.cpu().numpy()
        action = np.clip(action, self.action_low, self.action_high)
        return action, None

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def collect_dagger_rollout(
        self,
        expert_policy: Any,
        seed: Optional[int] = None,
        deterministic_expert: bool = True,
    ) -> int:
        """Roll out the current policy; label each state with the expert.

        States visited by the trained policy are passed to *expert_policy*
        which returns the ground-truth expert action.  These
        ``(obs, expert_action)`` pairs are added to the buffer.

        Parameters
        ----------
        expert_policy:
            Object with ``predict(obs, deterministic)`` method (e.g. SB3 PPO).
        seed:
            Optional seed for the episode reset.
        deterministic_expert:
            Whether to call the expert deterministically.

        Returns
        -------
        int
            Number of transitions collected.
        """
        obs, _ = self._env.reset(seed=seed)
        n_collected = 0
        episode_id = self._dagger_episode_counter
        self._dagger_episode_counter += 1
        step_id = 0

        while True:
            # Trained policy drives the environment.
            trained_action, _ = self.predict(obs, deterministic=True)
            # Expert labels the current state.
            expert_action, _ = expert_policy.predict(
                obs, deterministic=deterministic_expert
            )
            next_obs, reward, terminated, truncated, info = self._env.step(
                trained_action
            )

            demo = HumanDemonstration(
                observation=np.array(obs, dtype=np.float32),
                action=np.array(expert_action, dtype=np.float32),
                reward=float(reward),
                next_observation=np.array(next_obs, dtype=np.float32),
                terminated=bool(terminated),
                truncated=bool(truncated),
                episode_id=int(episode_id),
                step_id=step_id,
            )
            self._buffer.add(demo)
            n_collected += 1
            obs = next_obs
            step_id += 1
            if terminated or truncated:
                break

        return n_collected

    def train_step(self, rng: Optional[np.random.Generator] = None) -> float:
        """Perform one mini-batch supervised behavioural-cloning update.

        Parameters
        ----------
        rng:
            Optional random generator for reproducible sampling.

        Returns
        -------
        float
            Scalar BC loss for the update.

        Raises
        ------
        ValueError
            If the buffer is empty.

        Notes
        -----
        When the buffer contains fewer than ``batch_size`` entries the entire
        buffer is used as the mini-batch.
        """
        if len(self._buffer) == 0:
            raise ValueError(
                "DAggerTrainer.train_step() called on an empty buffer. "
                "Collect at least one demonstration before calling train_step()."
            )
        actual_batch = min(self.batch_size, len(self._buffer))
        obs_b, actions_b, _ = self._buffer.sample(actual_batch, rng=rng)

        obs_t = torch.as_tensor(obs_b, device=self.device)
        act_t = torch.as_tensor(actions_b, device=self.device)

        self.policy_net.train()
        self._optimizer.zero_grad()
        pred = self.policy_net(obs_t)
        loss = self._loss_fn(pred, act_t)
        loss.backward()
        self._optimizer.step()

        loss_val = float(loss.item())
        self.last_bc_loss = loss_val
        self.total_grad_steps += 1
        return loss_val

    def train(
        self,
        n_iterations: int,
        expert_policy: Any,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Run the full DAgger training loop.

        Each iteration:
        1. Collects a DAgger rollout (trained policy acts; expert labels states).
        2. Runs ``n_grad_steps`` supervised updates on the aggregated buffer.

        Parameters
        ----------
        n_iterations:
            Number of DAgger outer iterations.
        expert_policy:
            Expert policy with ``predict`` interface.
        seed:
            Base seed; each iteration uses ``seed + i`` when provided.

        Returns
        -------
        List[float]
            Mean BC loss per iteration.
        """
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations!r}")

        mean_losses: List[float] = []
        rng = np.random.default_rng(seed)

        for i in range(n_iterations):
            ep_seed = None if seed is None else seed + i
            n_new = self.collect_dagger_rollout(expert_policy, seed=ep_seed)

            if len(self._buffer) < 1:
                log.warning("DAgger iteration %d: buffer empty after rollout.", i)
                mean_losses.append(float("nan"))
                continue

            iter_losses: List[float] = []
            for _ in range(self.n_grad_steps):
                if len(self._buffer) < 1:
                    break
                iter_losses.append(self.train_step(rng=rng))

            mean_loss = float(np.mean(iter_losses)) if iter_losses else float("nan")
            mean_losses.append(mean_loss)
            log.info(
                "DAgger iteration %d/%d: collected %d steps, mean_bc_loss=%.4f",
                i + 1,
                n_iterations,
                n_new,
                mean_loss,
            )

        return mean_losses


# ---------------------------------------------------------------------------
# GAILDiscriminator
# ---------------------------------------------------------------------------


class GAILDiscriminator(nn.Module):
    """GAIL discriminator that distinguishes human demos from policy rollouts.

    Architecture: ``(obs, action) → Linear → Tanh → Linear → Tanh → Linear(1)``

    The output is an **unnormalised logit**.  A sigmoid converts it to
    ``P(human | obs, action)``.  The GAIL intrinsic reward is computed as::

        r_gail = −log(1 − sigmoid(logit))

    Parameters
    ----------
    obs_dim:
        Observation dimensionality.
    action_dim:
        Action dimensionality.
    hidden_sizes:
        Hidden-layer widths (default ``[64, 64]``).
    lr:
        Adam learning rate for discriminator updates (default ``3e-4``).
    device:
        PyTorch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._device = torch.device(device)

        input_dim = obs_dim + action_dim
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self._optimizer = optim.Adam(self.parameters(), lr=float(lr))
        self._bce = nn.BCEWithLogitsLoss()
        self.to(self._device)

        # Tracking
        self.last_discriminator_loss: float = float("nan")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised logit ``D(obs, action)``.

        Parameters
        ----------
        obs:
            Tensor of shape ``(B, obs_dim)``.
        action:
            Tensor of shape ``(B, action_dim)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(B, 1)``.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    # ------------------------------------------------------------------
    # GAIL reward
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Compute GAIL intrinsic reward ``−log(1 − sigmoid(D(s, a)))``.

        Parameters
        ----------
        obs:
            Observation(s), shape ``(obs_dim,)`` or ``(B, obs_dim)``.
        action:
            Action(s), shape ``(action_dim,)`` or ``(B, action_dim)``.

        Returns
        -------
        np.ndarray
            Reward(s), shape ``(B,)`` or scalar.
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.atleast_2d(np.asarray(obs, dtype=np.float32)),
                device=self._device,
            )
            act_t = torch.as_tensor(
                np.atleast_2d(np.asarray(action, dtype=np.float32)),
                device=self._device,
            )
            logit = self.forward(obs_t, act_t).squeeze(-1)
            # r = −log(1 − σ(logit))
            # Since 1 − σ(x) = σ(−x), we have:
            #   −log(σ(−logit)) = log(1 + exp(logit)) = softplus(logit)  ≥ 0
            reward = torch.nn.functional.softplus(logit)
        reward_np = reward.cpu().numpy()
        # Return scalar when input was 1-D
        if obs_t.shape[0] == 1 and np.ndim(obs) == 1:
            return float(reward_np[0])
        return reward_np

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        human_obs: np.ndarray,
        human_actions: np.ndarray,
        policy_obs: np.ndarray,
        policy_actions: np.ndarray,
    ) -> float:
        """Update the discriminator for one mini-batch.

        The human transitions are labelled **1** (real) and the policy
        transitions are labelled **0** (fake).

        Parameters
        ----------
        human_obs:
            Observations from human demonstrations, shape ``(B, obs_dim)``.
        human_actions:
            Actions from human demonstrations, shape ``(B, action_dim)``.
        policy_obs:
            Observations from policy rollouts, shape ``(B, obs_dim)``.
        policy_actions:
            Actions from policy rollouts, shape ``(B, action_dim)``.

        Returns
        -------
        float
            Binary cross-entropy discriminator loss.
        """
        self.train()
        h_obs = torch.as_tensor(
            np.asarray(human_obs, dtype=np.float32), device=self._device
        )
        h_act = torch.as_tensor(
            np.asarray(human_actions, dtype=np.float32), device=self._device
        )
        p_obs = torch.as_tensor(
            np.asarray(policy_obs, dtype=np.float32), device=self._device
        )
        p_act = torch.as_tensor(
            np.asarray(policy_actions, dtype=np.float32), device=self._device
        )

        real_logits = self.forward(h_obs, h_act)
        fake_logits = self.forward(p_obs, p_act)

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        loss = self._bce(
            torch.cat([real_logits, fake_logits], dim=0),
            torch.cat([real_labels, fake_labels], dim=0),
        )
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        loss_val = float(loss.item())
        self.last_discriminator_loss = loss_val
        return loss_val

    def accuracy(
        self,
        human_obs: np.ndarray,
        human_actions: np.ndarray,
        policy_obs: np.ndarray,
        policy_actions: np.ndarray,
    ) -> float:
        """Compute discriminator accuracy on a balanced evaluation set.

        Parameters
        ----------
        human_obs, human_actions:
            Observations and actions from human demonstrations.
        policy_obs, policy_actions:
            Observations and actions from policy rollouts.

        Returns
        -------
        float
            Fraction of correctly classified samples in ``[0, 1]``.
        """
        self.eval()
        with torch.no_grad():
            h_obs = torch.as_tensor(
                np.asarray(human_obs, dtype=np.float32), device=self._device
            )
            h_act = torch.as_tensor(
                np.asarray(human_actions, dtype=np.float32), device=self._device
            )
            p_obs = torch.as_tensor(
                np.asarray(policy_obs, dtype=np.float32), device=self._device
            )
            p_act = torch.as_tensor(
                np.asarray(policy_actions, dtype=np.float32), device=self._device
            )
            real_logits = self.forward(h_obs, h_act).squeeze(-1)
            fake_logits = self.forward(p_obs, p_act).squeeze(-1)

        real_correct = (real_logits.cpu().numpy() > 0).sum()
        fake_correct = (fake_logits.cpu().numpy() <= 0).sum()
        total = len(real_logits) + len(fake_logits)
        return float(real_correct + fake_correct) / max(total, 1)


# ---------------------------------------------------------------------------
# GAILRewardWrapper
# ---------------------------------------------------------------------------


class GAILRewardWrapper:
    """Wraps a gymnasium env replacing/augmenting reward with the GAIL reward.

    The GAIL intrinsic reward replaces the environment's extrinsic reward
    when *gail_coef* = 1.0 and *env_coef* = 0.0; it augments it otherwise.

    Parameters
    ----------
    env:
        Wrapped gymnasium-compatible environment.
    discriminator:
        A trained (or partially trained) :class:`GAILDiscriminator`.
    gail_coef:
        Weight applied to the GAIL intrinsic reward (default ``1.0``).
    env_coef:
        Weight applied to the environment's extrinsic reward (default ``0.0``).
    """

    def __init__(
        self,
        env: Any,
        discriminator: GAILDiscriminator,
        gail_coef: float = 1.0,
        env_coef: float = 0.0,
    ) -> None:
        self._env = env
        self._disc = discriminator
        self.gail_coef = float(gail_coef)
        self.env_coef = float(env_coef)

        # Expose the wrapped env's spaces directly.
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Cached pre-action observation; set by reset() and updated each step.
        self._prev_obs: Optional[np.ndarray] = None

    # Expose underlying env attributes transparently.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, **kwargs: Any) -> Any:
        """Reset the wrapped environment and cache the initial observation."""
        result = self._env.reset(**kwargs)
        # result may be (obs, info) or just obs depending on the env version.
        obs = result[0] if isinstance(result, tuple) else result
        self._prev_obs = np.array(obs, dtype=np.float32)
        return result

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment and replace reward with GAIL reward.

        The GAIL discriminator is evaluated on ``(prev_obs, action)`` — the
        pre-action state and the chosen action — which is the correct input
        for ``D(s, a)``.

        Parameters
        ----------
        action:
            Action to apply.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        prev_obs = self._prev_obs
        obs, env_reward, terminated, truncated, info = self._env.step(action)

        # Evaluate D on the pre-action state (or fall back to post-step obs
        # if reset() was never called and _prev_obs is not yet set).
        disc_obs = prev_obs if prev_obs is not None else obs
        gail_reward = float(self._disc.compute_reward(disc_obs, action))

        combined = self.gail_coef * gail_reward + self.env_coef * float(env_reward)
        info["gail_reward"] = gail_reward
        info["env_reward"] = float(env_reward)

        # Advance the cached observation for the next step.
        self._prev_obs = np.array(obs, dtype=np.float32)
        return obs, combined, terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()


# ---------------------------------------------------------------------------
# AARAnnotator
# ---------------------------------------------------------------------------


class AARAnnotator:
    """Annotates a recorded episode for After-Action Review.

    For each step the annotator:

    * Queries *reference_policy* for the AI-recommended action.
    * Estimates the state value (if the policy exposes a ``predict_values``
      method, otherwise falls back to a heuristic).
    * Computes the Euclidean distance between the human's action and the AI's.
    * Derives a quality score from the relative state value.
    * Identifies *decisive turning points* — steps where
      ``action_divergence > divergence_threshold``.

    Parameters
    ----------
    reference_policy:
        AI policy used as the quality oracle.  Must expose
        ``predict(obs, deterministic)`` returning ``(action, state)``.
        Optionally exposes ``predict_values(obs)`` returning a scalar value.
    divergence_threshold:
        Euclidean distance threshold above which a step is considered a
        *turning point* (default ``0.2``).
    top_k_turning_points:
        Maximum number of turning points to include in the report (default ``10``).
    """

    def __init__(
        self,
        reference_policy: Any,
        divergence_threshold: float = 0.2,
        top_k_turning_points: int = 10,
    ) -> None:
        self._policy = reference_policy
        self.divergence_threshold = float(divergence_threshold)
        self.top_k = int(top_k_turning_points)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def annotate(
        self,
        episode: List[HumanDemonstration],
        episode_id: int = 0,
    ) -> AARReport:
        """Annotate a full episode and return an :class:`AARReport`.

        Parameters
        ----------
        episode:
            Ordered list of :class:`HumanDemonstration` objects representing
            a complete or partial episode.
        episode_id:
            Identifier to embed in the report.

        Returns
        -------
        AARReport
        """
        if not episode:
            return AARReport(
                episode_id=episode_id,
                n_steps=0,
                annotations=[],
                turning_points=[],
                mean_quality_score=0.0,
                agreement_rate=0.0,
            )

        annotations: List[DecisionAnnotation] = []
        values: List[float] = []

        for demo in episode:
            obs = demo.observation
            ai_action, _ = self._policy.predict(obs, deterministic=True)
            ai_value = self._estimate_value(obs)

            divergence = float(
                np.linalg.norm(
                    np.asarray(demo.action, dtype=np.float32)
                    - np.asarray(ai_action, dtype=np.float32)
                )
            )

            ann = DecisionAnnotation(
                step_id=demo.step_id,
                human_action=np.array(demo.action, dtype=np.float32),
                ai_action=np.array(ai_action, dtype=np.float32),
                ai_value=float(ai_value),
                action_divergence=divergence,
                quality_score=0.0,  # placeholder; filled below
            )
            annotations.append(ann)
            values.append(ai_value)

        # Normalise AI values to [0, 1] as quality scores.
        v_arr = np.array(values, dtype=np.float32)
        v_min, v_max = float(v_arr.min()), float(v_arr.max())
        if v_max > v_min:
            q_scores = (v_arr - v_min) / (v_max - v_min)
        else:
            q_scores = np.ones_like(v_arr) * 0.5

        for i, ann in enumerate(annotations):
            annotations[i] = dataclasses.replace(
                ann, quality_score=float(q_scores[i])
            )

        mean_quality = float(np.mean(q_scores))

        # Agreement rate: human and AI agree when divergence < threshold.
        agreed = sum(
            1 for ann in annotations
            if ann.action_divergence < self.divergence_threshold
        )
        agreement_rate = agreed / len(annotations) if annotations else 0.0

        # Identify turning points: highest divergence steps.
        turning_points: List[TurningPoint] = []
        for ann in annotations:
            if ann.action_divergence >= self.divergence_threshold:
                turning_points.append(
                    TurningPoint(
                        step_id=ann.step_id,
                        divergence_score=ann.action_divergence,
                        human_action=ann.human_action.copy(),
                        ai_action=ann.ai_action.copy(),
                        description=(
                            f"Step {ann.step_id}: human deviated from AI by "
                            f"{ann.action_divergence:.3f} (threshold "
                            f"{self.divergence_threshold:.3f})"
                        ),
                    )
                )

        # Sort descending by divergence and keep top-k.
        turning_points.sort(key=lambda tp: tp.divergence_score, reverse=True)
        turning_points = turning_points[: self.top_k]

        return AARReport(
            episode_id=episode_id,
            n_steps=len(episode),
            annotations=annotations,
            turning_points=turning_points,
            mean_quality_score=mean_quality,
            agreement_rate=agreement_rate,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_value(self, obs: np.ndarray) -> float:
        """Estimate the state value using the reference policy.

        If the policy exposes ``predict_values(obs)`` (SB3 convention) it is
        called directly; otherwise a zero value is returned.

        Parameters
        ----------
        obs:
            Observation array.

        Returns
        -------
        float
            Estimated state value.
        """
        if hasattr(self._policy, "predict_values"):
            try:
                obs_t = torch.as_tensor(
                    np.atleast_2d(np.asarray(obs, dtype=np.float32))
                )
                val = self._policy.predict_values(obs_t)
                return float(val.squeeze().item())
            except Exception:
                pass
        if hasattr(self._policy, "policy") and hasattr(
            self._policy.policy, "predict_values"
        ):
            try:
                obs_t = torch.as_tensor(
                    np.atleast_2d(np.asarray(obs, dtype=np.float32))
                )
                val = self._policy.policy.predict_values(obs_t)
                return float(val.squeeze().item())
            except Exception:
                pass
        return 0.0
