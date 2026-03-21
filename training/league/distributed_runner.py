# training/league/distributed_runner.py
"""Distributed rollout runner using a Ray actor pool (E4.7).

Runs multiple :class:`~envs.remote_multi_battalion_env.RemoteMultiBattalionEnv`
workers in parallel to collect rollout batches for league training.  The
collected experience can be fed directly into
:class:`~training.train_mappo.MAPPOTrainer` or aggregated for centralized
policy updates.

Architecture
------------
* **Head node** — orchestrates policy parameters, aggregates gradients,
  writes checkpoints, and maintains the league pool / match database.
* **Rollout workers** — each runs one :class:`RemoteMultiBattalionEnv`
  actor.  Workers receive a snapshot of the current policy, execute one
  episode, and return the collected trajectory.

Throughput benchmark
--------------------
Run ``python -m training.league.distributed_runner benchmark`` to compare
single-process vs. multi-worker steps/sec on the local machine.

Classes
-------
RolloutResult
    NamedTuple holding a single episode trajectory.
DistributedRolloutRunner
    Ray actor pool that collects rollout batches in parallel.

Functions
---------
benchmark
    Compare single-process vs. Ray worker throughput.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import ray
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ray is required for DistributedRolloutRunner. "
        "Install it with: pip install 'ray[rllib]>=2.9.0'"
    ) from exc

from envs.remote_multi_battalion_env import RemoteMultiBattalionEnv, make_remote_envs

log = logging.getLogger(__name__)

__all__ = ["RolloutResult", "DistributedRolloutRunner", "benchmark"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class RolloutResult:
    """Container for a single episode rollout collected by a remote worker.

    Attributes
    ----------
    worker_id : int
        Zero-based index of the worker that collected this rollout.
    episode_steps : int
        Number of environment steps taken.
    total_reward : float
        Sum of all per-agent rewards over the episode.
    agent_rewards : dict[str, float]
        Per-agent total rewards.
    blue_won : bool
        ``True`` if the Blue team won the episode.
    observations : dict[str, np.ndarray]
        Final observations at episode end.
    elapsed_sec : float
        Wall-clock time for this rollout in seconds.
    metadata : dict[str, Any]
        Arbitrary additional metadata (opponent id, seed, etc.).
    """

    __slots__ = (
        "worker_id",
        "episode_steps",
        "total_reward",
        "agent_rewards",
        "blue_won",
        "observations",
        "elapsed_sec",
        "metadata",
    )

    def __init__(
        self,
        worker_id: int,
        episode_steps: int,
        total_reward: float,
        agent_rewards: Dict[str, float],
        blue_won: bool,
        observations: Dict[str, np.ndarray],
        elapsed_sec: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.worker_id = worker_id
        self.episode_steps = episode_steps
        self.total_reward = total_reward
        self.agent_rewards = agent_rewards
        self.blue_won = blue_won
        self.observations = observations
        self.elapsed_sec = elapsed_sec
        self.metadata = metadata or {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RolloutResult(worker={self.worker_id}, "
            f"steps={self.episode_steps}, "
            f"reward={self.total_reward:.2f}, "
            f"blue_won={self.blue_won})"
        )


# ---------------------------------------------------------------------------
# Remote worker task
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0)
def _run_episode_on_worker(
    worker_id: int,
    env_handle: ray.actor.ActorHandle,
    seed: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> RolloutResult:
    """Run one full episode on *env_handle* and return a :class:`RolloutResult`.

    This is a free Ray task (not an actor method) so it can be dispatched
    to any available worker process.  The environment actor *env_handle*
    must already exist and be ready.

    Parameters
    ----------
    worker_id:
        Index used to identify this rollout in the result.
    env_handle:
        Handle to a :class:`RemoteMultiBattalionEnv` actor.
    seed:
        Episode seed for reproducibility.
    metadata:
        Arbitrary key-value metadata to attach to the result.

    Returns
    -------
    :class:`RolloutResult`
    """
    t0 = time.perf_counter()
    result = ray.get(env_handle.run_episode.remote(seed=seed))
    elapsed = time.perf_counter() - t0
    return RolloutResult(
        worker_id=worker_id,
        episode_steps=result["steps"],
        total_reward=result["total_reward"],
        agent_rewards=result["agent_rewards"],
        blue_won=result["blue_won"],
        observations=result["observations"],
        elapsed_sec=elapsed,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class DistributedRolloutRunner:
    """Parallel rollout collector backed by a Ray actor pool.

    Manages *num_workers* :class:`RemoteMultiBattalionEnv` actors and
    dispatches episodes to them concurrently.  Results are aggregated
    locally after all workers complete.

    Parameters
    ----------
    num_workers:
        Number of parallel rollout worker actors to create.
    env_kwargs:
        Keyword arguments forwarded to each
        :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance.
    num_cpus_per_worker:
        CPU allocation hint per worker actor (can be fractional).
    ray_address:
        Optional Ray cluster address (``"auto"`` for a running cluster).
        Pass ``None`` (default) to use a local Ray instance.
    ray_init_kwargs:
        Extra keyword arguments forwarded to :func:`ray.init` when Ray
        is not already initialised.

    Examples
    --------
    >>> runner = DistributedRolloutRunner(num_workers=4, env_kwargs={"n_blue": 2, "n_red": 2})
    >>> results = runner.collect_rollouts(n_episodes=8, base_seed=0)
    >>> throughput = runner.steps_per_second(results)
    """

    def __init__(
        self,
        num_workers: int,
        env_kwargs: Optional[Dict[str, Any]] = None,
        num_cpus_per_worker: float = 1.0,
        ray_address: Optional[str] = None,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers!r}")

        self.num_workers = num_workers
        self._env_kwargs: Dict[str, Any] = env_kwargs or {}
        self._num_cpus_per_worker = num_cpus_per_worker

        # Initialize Ray if not already running.
        if not ray.is_initialized():
            init_kwargs_copy = dict(ray_init_kwargs or {})
            if ray_address is not None:
                init_kwargs_copy.setdefault("address", ray_address)
            ray.init(**init_kwargs_copy)
            log.info("Ray initialized (address=%s)", ray_address or "local")

        # Spin up environment actors.
        self._envs: List[ray.actor.ActorHandle] = make_remote_envs(
            num_envs=num_workers,
            num_cpus_per_env=num_cpus_per_worker,
            **self._env_kwargs,
        )
        log.info("Created %d remote environment actors.", num_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        n_episodes: int,
        base_seed: int = 0,
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[RolloutResult]:
        """Collect *n_episodes* rollouts using the worker pool.

        Episodes are distributed round-robin across workers; multiple
        waves are issued if ``n_episodes > num_workers``.

        Parameters
        ----------
        n_episodes:
            Total number of episodes to collect.
        base_seed:
            Starting seed.  Episode *i* uses seed ``base_seed + i``.
        metadata:
            Optional list of per-episode metadata dicts (length must equal
            *n_episodes* if provided).

        Returns
        -------
        list of :class:`RolloutResult`, ordered by episode index.
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes!r}")
        if metadata is not None and len(metadata) != n_episodes:
            raise ValueError(
                f"metadata length ({len(metadata)}) must equal n_episodes ({n_episodes})"
            )

        results: List[RolloutResult] = []
        remaining = list(range(n_episodes))
        t0 = time.perf_counter()

        while remaining:
            batch = remaining[: self.num_workers]
            remaining = remaining[self.num_workers :]

            refs = []
            for ep_idx in batch:
                worker_id = ep_idx % self.num_workers
                env_handle = self._envs[worker_id]
                seed = base_seed + ep_idx
                meta = metadata[ep_idx] if metadata is not None else None
                refs.append(
                    _run_episode_on_worker.remote(worker_id, env_handle, seed, meta)
                )

            batch_results: List[RolloutResult] = ray.get(refs)
            results.extend(batch_results)

        self._last_collect_elapsed_sec: float = time.perf_counter() - t0
        return results

    def collect_rollouts_async(
        self,
        n_episodes: int,
        base_seed: int = 0,
    ) -> List[ray.ObjectRef]:
        """Dispatch *n_episodes* rollouts asynchronously (non-blocking).

        Returns a list of :class:`ray.ObjectRef` handles.  Call
        ``ray.get(refs)`` to wait for completion.

        Parameters
        ----------
        n_episodes:
            Total number of episodes to dispatch.
        base_seed:
            Starting seed.

        Returns
        -------
        list of :class:`ray.ObjectRef`
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes!r}")
        refs = []
        for ep_idx in range(n_episodes):
            worker_id = ep_idx % self.num_workers
            refs.append(
                _run_episode_on_worker.remote(
                    worker_id, self._envs[worker_id], base_seed + ep_idx
                )
            )
        return refs

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def steps_per_second(
        results: Sequence[RolloutResult],
        total_elapsed_sec: Optional[float] = None,
    ) -> float:
        """Compute aggregate steps/sec across a collection of results.

        Parameters
        ----------
        results:
            Collection of :class:`RolloutResult` objects.
        total_elapsed_sec:
            Actual wall-clock time for the entire collection in seconds.
            Pass the value of :attr:`_last_collect_elapsed_sec` (set by
            :meth:`collect_rollouts`) for accurate multi-wave throughput.
            When ``None``, falls back to ``max(r.elapsed_sec)`` across
            results, which underestimates total time when more than one
            wave was needed (``n_episodes > num_workers``).

        Returns
        -------
        float
            Steps per second (``total_steps / elapsed``).
        """
        if not results:
            return 0.0
        total_steps = sum(r.episode_steps for r in results)
        elapsed = total_elapsed_sec if total_elapsed_sec is not None else max(r.elapsed_sec for r in results)
        if elapsed <= 0.0:
            return float("inf")
        return total_steps / elapsed

    @staticmethod
    def win_rate(results: Sequence[RolloutResult]) -> float:
        """Return the Blue win rate across *results*.

        Parameters
        ----------
        results:
            Collection of :class:`RolloutResult` objects.

        Returns
        -------
        float in [0, 1].
        """
        if not results:
            return 0.0
        return sum(1 for r in results if r.blue_won) / len(results)

    @staticmethod
    def mean_episode_length(results: Sequence[RolloutResult]) -> float:
        """Return the mean episode length across *results*.

        Parameters
        ----------
        results:
            Collection of :class:`RolloutResult` objects.

        Returns
        -------
        float
        """
        if not results:
            return 0.0
        return sum(r.episode_steps for r in results) / len(results)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, kill_actors: bool = True) -> None:
        """Shut down all worker actors.

        Parameters
        ----------
        kill_actors:
            If ``True`` (default), explicitly call :func:`ray.kill` on
            each actor before returning.

        Notes
        -----
        This method does **not** call :func:`ray.shutdown`.  To stop Ray
        entirely (e.g. at the end of a training script), call
        ``ray.shutdown()`` separately after invoking this method.
        """
        if kill_actors:
            for env in self._envs:
                try:
                    ray.kill(env)
                except Exception:  # noqa: BLE001
                    pass
        log.info("DistributedRolloutRunner shut down.")

    def __enter__(self) -> "DistributedRolloutRunner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark(
    num_workers: int = 4,
    n_episodes: int = 16,
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Benchmark single-process vs. Ray actor-pool throughput.

    Runs *n_episodes* episodes sequentially (single process) and then in
    parallel using *num_workers* Ray actors, then reports the speedup.

    Parameters
    ----------
    num_workers:
        Number of parallel Ray workers.
    n_episodes:
        Number of episodes per configuration.
    env_kwargs:
        Environment constructor arguments.
    seed:
        Base seed for episode reproducibility.

    Returns
    -------
    dict with keys:
        ``single_steps_per_sec``, ``ray_steps_per_sec``, ``speedup``.
    """
    from envs.multi_battalion_env import MultiBattalionEnv

    effective_env_kwargs = env_kwargs or {"n_blue": 2, "n_red": 2}

    # --- Single-process baseline ---
    t0 = time.perf_counter()
    single_steps = 0
    single_env = MultiBattalionEnv(**effective_env_kwargs)
    for ep in range(n_episodes):
        obs, _ = single_env.reset(seed=seed + ep)
        for _ in range(single_env.max_steps):
            if not single_env.agents:
                break
            actions = {a: single_env.action_space(a).sample() for a in single_env.agents}
            obs, _, terminated, truncated, _ = single_env.step(actions)
            single_steps += 1
    single_env.close()
    single_elapsed = time.perf_counter() - t0
    single_sps = single_steps / single_elapsed if single_elapsed > 0 else 0.0

    # --- Ray actor-pool ---
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    t1 = time.perf_counter()
    runner = DistributedRolloutRunner(num_workers=num_workers, env_kwargs=effective_env_kwargs)
    results = runner.collect_rollouts(n_episodes=n_episodes, base_seed=seed)
    ray_elapsed = time.perf_counter() - t1
    ray_steps = sum(r.episode_steps for r in results)
    ray_sps = ray_steps / ray_elapsed if ray_elapsed > 0 else 0.0
    runner.shutdown()

    speedup = ray_sps / single_sps if single_sps > 0 else float("nan")

    print(
        f"\nBenchmark ({n_episodes} episodes, {num_workers} workers):\n"
        f"  Single process : {single_sps:>10.1f} steps/sec\n"
        f"  Ray ({num_workers:2d} workers): {ray_sps:>10.1f} steps/sec\n"
        f"  Speedup        : {speedup:>10.2f}x\n"
    )
    return {
        "single_steps_per_sec": single_sps,
        "ray_steps_per_sec": ray_sps,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="E4.7 Distributed training utilities")
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("benchmark", help="Throughput benchmark: single vs. Ray")
    bench.add_argument("--workers", type=int, default=4)
    bench.add_argument("--episodes", type=int, default=16)
    bench.add_argument("--n-blue", type=int, default=2)
    bench.add_argument("--n-red", type=int, default=2)
    bench.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(
            num_workers=args.workers,
            n_episodes=args.episodes,
            env_kwargs={"n_blue": args.n_blue, "n_red": args.n_red},
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
