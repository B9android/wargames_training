# training/self_play.py
"""Self-play training utilities.

Provides:

* :class:`OpponentPool` — a fixed-size pool of frozen policy snapshots that
  can be sampled uniformly as opponents during self-play training.
* :class:`SelfPlayCallback` — SB3 callback that periodically snapshots the
  current policy into the pool and swaps the Red opponent in the vectorized
  training environment.
* :class:`WinRateVsPoolCallback` — SB3 callback that evaluates the current
  policy against a random opponent from the pool and logs the win rate to
  W&B.
* :func:`evaluate_vs_pool` — standalone helper that runs *n* evaluation
  episodes against an opponent sampled from the pool and returns the win
  rate.

Typical usage::

    from training.self_play import OpponentPool, SelfPlayCallback, WinRateVsPoolCallback
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from envs.battalion_env import BattalionEnv

    pool = OpponentPool(pool_dir="checkpoints/pool", max_size=10)
    env = make_vec_env(BattalionEnv, n_envs=8)

    model = PPO("MlpPolicy", env)
    sp_cb = SelfPlayCallback(pool=pool, snapshot_freq=50_000, vec_env=env)
    wr_cb = WinRateVsPoolCallback(pool=pool, eval_freq=50_000)

    model.learn(total_timesteps=1_000_000, callback=[sp_cb, wr_cb])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from envs.battalion_env import BattalionEnv, DESTROYED_THRESHOLD

log = logging.getLogger(__name__)

__all__ = [
    "OpponentPool",
    "SelfPlayCallback",
    "WinRateVsPoolCallback",
    "evaluate_vs_pool",
]


# ---------------------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------------------


class OpponentPool:
    """Fixed-size pool of frozen PPO policy snapshots.

    Snapshots are stored as Stable-Baselines3 ``.zip`` files under
    *pool_dir*.  The pool keeps at most *max_size* snapshots; when full,
    the oldest snapshot is evicted to make room for the newest one.

    Parameters
    ----------
    pool_dir:
        Directory where snapshot ``.zip`` files are persisted.  Created
        on first :meth:`add` if it does not already exist.
    max_size:
        Maximum number of snapshots to retain (default 10).

    Attributes
    ----------
    pool_dir : Path
        Resolved path of the snapshot directory.
    max_size : int
        Maximum number of snapshots retained in the pool.
    """

    def __init__(self, pool_dir: str | Path, max_size: int = 10) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.pool_dir = Path(pool_dir)
        self.max_size = int(max_size)
        # Ordered list of snapshot file paths (oldest first).
        self._snapshots: List[Path] = []
        # Shared RNG instance used for uniform sampling when no external RNG is given.
        self._rng: np.random.Generator = np.random.default_rng()
        # Restore existing snapshots from disk if the directory exists.
        self._reload_from_disk()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, model: PPO, version: int) -> Path:
        """Save *model* as a new snapshot and add it to the pool.

        If the pool already contains *max_size* snapshots the oldest is
        removed from disk before saving the new one.

        Parameters
        ----------
        model:
            The current PPO model to snapshot.
        version:
            Monotonically increasing version number embedded in the
            snapshot file name for traceability.

        Returns
        -------
        Path
            Path of the newly saved snapshot file (including ``.zip``).
        """
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.pool_dir / f"snapshot_v{version:06d}"
        model.save(str(snapshot_path))
        full_path = snapshot_path.with_suffix(".zip")

        # Evict oldest if at capacity.
        while len(self._snapshots) >= self.max_size:
            oldest = self._snapshots.pop(0)
            try:
                oldest.unlink(missing_ok=True)
                log.debug("Evicted snapshot %s", oldest)
            except OSError as exc:
                log.warning("Failed to evict snapshot %s: %s", oldest, exc)

        self._snapshots.append(full_path)
        log.info("Saved snapshot %s (pool size %d/%d)", full_path, len(self._snapshots), self.max_size)
        return full_path

    def sample(self, rng: Optional[np.random.Generator] = None) -> Optional[PPO]:
        """Load and return a uniformly sampled snapshot as a PPO model.

        Parameters
        ----------
        rng:
            Optional NumPy random generator for reproducible sampling.
            When ``None``, the pool's internal shared RNG instance is used
            (seeded once at pool construction time).

        Returns
        -------
        PPO or None
            A loaded PPO model, or ``None`` if the pool is empty.
        """
        if not self._snapshots:
            return None
        _rng = rng if rng is not None else self._rng
        idx = int(_rng.integers(0, len(self._snapshots)))
        path = self._snapshots[idx]
        try:
            model = PPO.load(str(path))
            log.debug("Sampled snapshot %s", path)
            return model
        except Exception as exc:
            log.warning("Failed to load snapshot %s: %s", path, exc)
            return None

    def sample_latest(self) -> Optional[PPO]:
        """Load and return the most recently added snapshot.

        Returns
        -------
        PPO or None
            The latest PPO model, or ``None`` if the pool is empty.
        """
        if not self._snapshots:
            return None
        path = self._snapshots[-1]
        try:
            return PPO.load(str(path))
        except Exception as exc:
            log.warning("Failed to load latest snapshot %s: %s", path, exc)
            return None

    @property
    def size(self) -> int:
        """Current number of snapshots in the pool."""
        return len(self._snapshots)

    @property
    def snapshot_paths(self) -> List[Path]:
        """Ordered list of snapshot paths (oldest first, read-only copy)."""
        return list(self._snapshots)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reload_from_disk(self) -> None:
        """Populate *_snapshots* from existing files in *pool_dir*.

        Files beyond the pool's *max_size* capacity (the oldest ones) are
        deleted from disk to enforce the pool invariant across restarts.
        """
        if not self.pool_dir.exists():
            return
        existing = sorted(self.pool_dir.glob("snapshot_v*.zip"))
        # Delete excess (oldest) snapshots so on-disk state matches the pool
        # invariant of keeping at most *max_size* files.
        excess = existing[: max(0, len(existing) - self.max_size)]  # guard for len <= max_size
        for path in excess:
            try:
                path.unlink(missing_ok=True)
                log.debug("_reload_from_disk: deleted excess snapshot %s", path)
            except OSError as exc:
                log.warning("_reload_from_disk: failed to delete %s: %s", path, exc)
        self._snapshots = existing[max(0, len(existing) - self.max_size) :]
        if self._snapshots:
            log.info(
                "Restored %d snapshot(s) from %s", len(self._snapshots), self.pool_dir
            )


# ---------------------------------------------------------------------------
# SelfPlayCallback
# ---------------------------------------------------------------------------


class SelfPlayCallback(BaseCallback):
    """Periodically snapshots the current policy and updates the Red opponent.

    Every *snapshot_freq* environment steps the current model is saved to
    the :class:`OpponentPool`.  If the pool contains at least one snapshot,
    a uniformly sampled opponent is loaded and injected into each
    environment in *vec_env* via :meth:`~envs.battalion_env.BattalionEnv.set_red_policy`.

    Parameters
    ----------
    pool:
        The :class:`OpponentPool` to save snapshots into.
    snapshot_freq:
        How often (in environment steps) to take a snapshot.
    vec_env:
        The vectorized training environment whose Red opponents should be
        updated.  When ``None``, the callback uses
        ``self.training_env`` (set automatically by SB3 during
        ``model.learn()``).
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        pool: OpponentPool,
        snapshot_freq: int = 50_000,
        vec_env: Optional[VecEnv] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if int(snapshot_freq) < 1:
            raise ValueError(f"snapshot_freq must be >= 1, got {snapshot_freq}")
        self.pool = pool
        self.snapshot_freq = int(snapshot_freq)
        self._vec_env = vec_env
        # Initialize version counter from any snapshots already in the pool so
        # that a training restart doesn't overwrite existing snapshot files.
        self._version: int = _max_version_in_pool(pool)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.snapshot_freq == 0 and self.num_timesteps > 0:
            self._take_snapshot_and_update()
        return True

    def _take_snapshot_and_update(self) -> None:
        """Save current model to pool and refresh all Red opponents."""
        self._version += 1
        self.pool.add(self.model, self._version)

        opponent = self.pool.sample()
        if opponent is None:
            return

        env = self._vec_env if self._vec_env is not None else self.training_env
        if env is None:
            log.warning("SelfPlayCallback: no environment available to update Red policy.")
            return

        # Propagate to every sub-environment.
        for env_instance in _iter_envs(env):
            env_instance.set_red_policy(opponent)

        if self.verbose >= 1:
            log.info(
                "SelfPlayCallback: snapshot v%d saved; Red policy updated (pool=%d).",
                self._version,
                self.pool.size,
            )


# ---------------------------------------------------------------------------
# WinRateVsPoolCallback
# ---------------------------------------------------------------------------


class WinRateVsPoolCallback(BaseCallback):
    """Evaluates the current policy vs. a pool opponent and logs win rate.

    Every *eval_freq* environment steps, runs *n_eval_episodes* episodes
    in a temporary :class:`~envs.battalion_env.BattalionEnv` where Red is
    driven by an opponent sampled from *pool*.  The resulting win rate is
    logged to W&B (if available) and to the SB3 logger.

    Parameters
    ----------
    pool:
        :class:`OpponentPool` to sample opponents from.
    eval_freq:
        How often (in environment steps) to run the evaluation.
    n_eval_episodes:
        Number of episodes per evaluation (default 20).
    deterministic:
        Whether the policy acts deterministically during evaluation
        (default ``True``).
    use_latest:
        When ``True``, always evaluate against the *latest* snapshot
        instead of a random one (default ``False``).
    verbose:
        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        pool: OpponentPool,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        use_latest: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if int(eval_freq) < 1:
            raise ValueError(f"eval_freq must be >= 1, got {eval_freq}")
        if int(n_eval_episodes) < 1:
            raise ValueError(f"n_eval_episodes must be >= 1, got {n_eval_episodes}")
        self.pool = pool
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.use_latest = use_latest

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Run evaluation and log the win rate."""
        if self.pool.size == 0:
            return

        opponent = (
            self.pool.sample_latest() if self.use_latest else self.pool.sample()
        )
        if opponent is None:
            return

        win_rate = evaluate_vs_pool(
            model=self.model,
            opponent=opponent,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )

        if self.verbose >= 1:
            log.info(
                "WinRateVsPoolCallback: win_rate_vs_pool=%.3f (n=%d, step=%d)",
                win_rate,
                self.n_eval_episodes,
                self.num_timesteps,
            )

        # Log to SB3 logger (also picked up by TensorBoard if configured).
        self.logger.record("self_play/win_rate_vs_pool", win_rate)

        # Log to W&B if active.
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "self_play/win_rate_vs_pool": win_rate,
                        "time/total_timesteps": self.num_timesteps,
                    },
                    step=self.num_timesteps,
                )
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# evaluate_vs_pool
# ---------------------------------------------------------------------------


def evaluate_vs_pool(
    model: PPO,
    opponent: PPO,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> float:
    """Evaluate *model* against *opponent* in self-play episodes.

    Runs *n_episodes* episodes of :class:`~envs.battalion_env.BattalionEnv`
    where Blue is driven by *model* and Red is driven by *opponent*.

    Parameters
    ----------
    model:
        The policy under evaluation (controls Blue).
    opponent:
        The frozen snapshot policy (controls Red).
    n_episodes:
        Number of evaluation episodes (default 20).
    deterministic:
        Whether *model* acts deterministically (default ``True``).
        *opponent* always acts stochastically to simulate a diverse pool.
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.

    Returns
    -------
    float
        Win rate in ``[0, 1]`` (Blue wins / total episodes).

    Raises
    ------
    ValueError
        If *n_episodes* < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    env = BattalionEnv(red_policy=opponent)
    wins = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        red_lost = (
            info.get("red_routed", False)
            or env.red.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
        )
        blue_lost = (
            info.get("blue_routed", False)
            or env.blue.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
        )
        if red_lost and not blue_lost:
            wins += 1

    env.close()
    return wins / n_episodes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_envs(vec_env: VecEnv):
    """Yield each underlying :class:`BattalionEnv` from a vectorized env."""
    # SB3 VecEnv exposes `envs` on DummyVecEnv and SubprocVecEnv wrappers.
    envs = getattr(vec_env, "envs", None)
    if envs is None:
        # Try unwrapping one level (e.g. VecMonitor → DummyVecEnv).
        inner = getattr(vec_env, "venv", None)
        if inner is not None:
            envs = getattr(inner, "envs", None)
    if envs is None:
        log.warning("_iter_envs: could not access underlying envs from %s", type(vec_env).__name__)
        return
    for env in envs:
        # Unwrap Monitor wrappers if present.
        inner_env = env
        while hasattr(inner_env, "env"):
            inner_env = inner_env.env
        if isinstance(inner_env, BattalionEnv):
            yield inner_env


def _max_version_in_pool(pool: OpponentPool) -> int:
    """Return the highest version number present in *pool*'s snapshot files.

    Snapshot file names follow the pattern ``snapshot_v{version:06d}.zip``.
    Returns ``0`` if the pool is empty or no version can be parsed.
    """
    max_ver = 0
    for path in pool.snapshot_paths:
        try:
            ver = int(path.stem.split("_v")[-1])
            if ver > max_ver:
                max_ver = ver
        except (ValueError, IndexError):
            pass
    return max_ver
