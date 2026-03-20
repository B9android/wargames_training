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

Multi-agent (MAPPO) additions:

* :class:`TeamOpponentPool` — fixed-size pool of frozen
  :class:`~models.mappo_policy.MAPPOPolicy` snapshots for team self-play.
* :func:`evaluate_team_vs_pool` — evaluate a MAPPO policy (Blue) against a
  frozen team opponent (Red) and return the win rate.
* :func:`nash_exploitability_proxy` — estimate exploitability as
  ``max(opp_win_rates) − mean(opp_win_rates)`` across all pool members.

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
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from envs.battalion_env import BattalionEnv, DESTROYED_THRESHOLD

if TYPE_CHECKING:
    import torch
    from models.mappo_policy import MAPPOPolicy

log = logging.getLogger(__name__)

__all__ = [
    "OpponentPool",
    "SelfPlayCallback",
    "WinRateVsPoolCallback",
    "evaluate_vs_pool",
    "TeamOpponentPool",
    "evaluate_team_vs_pool",
    "nash_exploitability_proxy",
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


# ---------------------------------------------------------------------------
# TeamOpponentPool
# ---------------------------------------------------------------------------


class TeamOpponentPool:
    """Fixed-size pool of frozen :class:`~models.mappo_policy.MAPPOPolicy` snapshots.

    Snapshots are stored as PyTorch ``.pt`` files under *pool_dir*.  Each
    file contains the policy ``state_dict`` plus the constructor kwargs
    (``obs_dim``, ``action_dim``, ``state_dim``, ``n_agents``,
    ``share_parameters``) needed to reconstruct the policy at load time.
    The pool keeps at most *max_size* snapshots; when full, the oldest is
    evicted to make room for the newest.

    Parameters
    ----------
    pool_dir:
        Directory where snapshot ``.pt`` files are persisted.  Created on
        first :meth:`add` if it does not already exist.
    max_size:
        Maximum number of snapshots to retain (default 10).
    """

    def __init__(self, pool_dir: str | Path, max_size: int = 10) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.pool_dir = Path(pool_dir)
        self.max_size = int(max_size)
        self._snapshots: List[Path] = []
        self._rng: np.random.Generator = np.random.default_rng()
        self._reload_from_disk()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, policy: "MAPPOPolicy", version: int) -> Path:
        """Save *policy* as a new snapshot and add it to the pool.

        The snapshot stores both the ``state_dict`` and the constructor
        kwargs so the policy can be fully reconstructed later.

        Parameters
        ----------
        policy:
            The :class:`~models.mappo_policy.MAPPOPolicy` to snapshot.
        version:
            Monotonically increasing version number embedded in the file
            name for traceability.

        Returns
        -------
        Path
            Path of the newly saved snapshot file.
        """
        import torch  # local import to avoid hard dependency at module load

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.pool_dir / f"team_snapshot_v{version:06d}.pt"

        torch.save(
            {
                "state_dict": policy.state_dict(),
                "kwargs": {
                    "obs_dim": policy.obs_dim,
                    "action_dim": policy.action_dim,
                    "state_dim": policy.state_dim,
                    "n_agents": policy.n_agents,
                    "share_parameters": policy.share_parameters,
                    "actor_hidden_sizes": policy.actor_hidden_sizes,
                    "critic_hidden_sizes": policy.critic_hidden_sizes,
                },
            },
            snapshot_path,
        )

        # Evict oldest if at capacity.
        while len(self._snapshots) >= self.max_size:
            oldest = self._snapshots.pop(0)
            try:
                oldest.unlink(missing_ok=True)
                log.debug("TeamOpponentPool: evicted snapshot %s", oldest)
            except OSError as exc:
                log.warning("TeamOpponentPool: failed to evict %s: %s", oldest, exc)

        self._snapshots.append(snapshot_path)
        log.info(
            "TeamOpponentPool: saved snapshot %s (pool=%d/%d)",
            snapshot_path,
            len(self._snapshots),
            self.max_size,
        )
        return snapshot_path

    def sample(self, rng: Optional[np.random.Generator] = None, device: Optional[str] = None) -> Optional["MAPPOPolicy"]:
        """Load and return a uniformly sampled snapshot.

        Parameters
        ----------
        rng:
            Optional NumPy random generator for reproducible sampling.
            When ``None``, the pool's internal RNG is used.
        device:
            Optional PyTorch device string (e.g. ``"cuda:0"``).  When
            provided the loaded policy is moved to *device* before being
            returned.  When ``None`` the policy stays on CPU.

        Returns
        -------
        MAPPOPolicy or None
            A loaded policy in evaluation mode, or ``None`` if the pool is
            empty or loading fails.
        """
        if not self._snapshots:
            return None
        _rng = rng if rng is not None else self._rng
        idx = int(_rng.integers(0, len(self._snapshots)))
        return self._load_snapshot(self._snapshots[idx], device=device)

    def sample_latest(self, device: Optional[str] = None) -> Optional["MAPPOPolicy"]:
        """Load and return the most recently added snapshot.

        Parameters
        ----------
        device:
            Optional PyTorch device string.  When provided the loaded
            policy is moved to *device* before being returned.

        Returns
        -------
        MAPPOPolicy or None
            The latest policy, or ``None`` if the pool is empty.
        """
        if not self._snapshots:
            return None
        return self._load_snapshot(self._snapshots[-1], device=device)

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

    def _load_snapshot(self, path: Path, device: Optional[str] = None) -> Optional["MAPPOPolicy"]:
        """Load a :class:`~models.mappo_policy.MAPPOPolicy` from *path*.

        Parameters
        ----------
        path:
            Path to the ``.pt`` snapshot file.
        device:
            Optional PyTorch device string.  When provided the loaded
            policy is moved to *device* after loading (which always uses
            CPU as the intermediate map_location for safety).
        """
        import torch  # local import
        from models.mappo_policy import MAPPOPolicy

        try:
            # weights_only=True restricts deserialization to safe tensor/primitive
            # types, mitigating arbitrary-code-execution risk from tampered files.
            data = torch.load(str(path), map_location="cpu", weights_only=True)
            policy = MAPPOPolicy(**data["kwargs"])
            policy.load_state_dict(data["state_dict"])
            if device is not None:
                policy = policy.to(device)
            policy.eval()
            actual_device = next(policy.parameters()).device
            log.debug("TeamOpponentPool: loaded snapshot %s (device=%s)", path, actual_device)
            return policy
        except Exception as exc:
            log.warning("TeamOpponentPool: failed to load %s: %s", path, exc)
            return None

    def _reload_from_disk(self) -> None:
        """Populate *_snapshots* from existing files in *pool_dir*."""
        if not self.pool_dir.exists():
            return
        existing = sorted(self.pool_dir.glob("team_snapshot_v*.pt"))
        excess = existing[: max(0, len(existing) - self.max_size)]
        for path in excess:
            try:
                path.unlink(missing_ok=True)
                log.debug("TeamOpponentPool._reload_from_disk: deleted excess %s", path)
            except OSError as exc:
                log.warning(
                    "TeamOpponentPool._reload_from_disk: failed to delete %s: %s", path, exc
                )
        self._snapshots = existing[max(0, len(existing) - self.max_size) :]
        if self._snapshots:
            log.info(
                "TeamOpponentPool: restored %d snapshot(s) from %s",
                len(self._snapshots),
                self.pool_dir,
            )


# ---------------------------------------------------------------------------
# evaluate_team_vs_pool
# ---------------------------------------------------------------------------


def evaluate_team_vs_pool(
    policy: "MAPPOPolicy",
    opponent: "MAPPOPolicy",
    n_blue: int = 2,
    n_red: int = 2,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict] = None,
) -> float:
    """Evaluate a MAPPO *policy* (Blue) against *opponent* (Red) in self-play.

    Runs *n_episodes* episodes of
    :class:`~envs.multi_battalion_env.MultiBattalionEnv` where Blue is
    driven by *policy* and Red is driven by *opponent*.

    For symmetric self-play (``n_blue == n_red``) the opponent is used
    directly as a Red policy.  When team sizes differ, the opponent's
    shared actor is applied to each Red agent in turn.

    Parameters
    ----------
    policy:
        The Blue :class:`~models.mappo_policy.MAPPOPolicy` under evaluation.
    opponent:
        The frozen Red opponent policy.
    n_blue, n_red:
        Team sizes (must match the training configuration).
    n_episodes:
        Number of evaluation episodes (default 20).
    deterministic:
        Blue acts deterministically when ``True``; Red always acts
        stochastically to simulate a diverse pool.
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.
    env_kwargs:
        Extra keyword arguments forwarded to
        :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

    Returns
    -------
    float
        Blue win rate in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *n_episodes* < 1.
    """
    import torch  # local import
    from envs.multi_battalion_env import MultiBattalionEnv

    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    # Derive evaluation device from policy parameters; fall back to CPU if the
    # policy has no parameters (edge case / mocked policies in tests).
    try:
        eval_device = next(policy.parameters()).device
    except StopIteration:
        eval_device = torch.device("cpu")

    # Move opponent to the same device so all tensor operations are consistent.
    opponent = opponent.to(eval_device)
    opponent.eval()

    _env_kwargs: dict = env_kwargs or {}
    env = MultiBattalionEnv(n_blue=n_blue, n_red=n_red, **_env_kwargs)
    act_low = env._act_space.low
    act_high = env._act_space.high
    obs_dim = env._obs_dim

    wins = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        blue_won = False

        while env.agents:
            action_dict: dict[str, np.ndarray] = {}

            # Blue actions (controlled by *policy*)
            for i in range(n_blue):
                agent_id = f"blue_{i}"
                if agent_id in env.agents:
                    agent_obs = obs.get(agent_id, np.zeros(obs_dim, dtype=np.float32))
                    obs_t = torch.as_tensor(agent_obs, device=eval_device).unsqueeze(0)
                    with torch.no_grad():
                        acts_t, _ = policy.act(obs_t, agent_idx=i, deterministic=deterministic)
                    action_dict[agent_id] = np.clip(
                        acts_t[0].cpu().numpy(), act_low, act_high
                    )

            # Red actions (controlled by *opponent*)
            for i in range(n_red):
                agent_id = f"red_{i}"
                if agent_id in env.agents:
                    agent_obs = obs.get(agent_id, np.zeros(obs_dim, dtype=np.float32))
                    obs_t = torch.as_tensor(agent_obs, device=eval_device).unsqueeze(0)
                    with torch.no_grad():
                        acts_t, _ = opponent.act(
                            obs_t,
                            agent_idx=i % opponent.n_agents,
                            deterministic=False,
                        )
                    action_dict[agent_id] = np.clip(
                        acts_t[0].cpu().numpy(), act_low, act_high
                    )

            obs, _, _, _, _ = env.step(action_dict)

            # Win condition: Red fully eliminated while at least one Blue alive
            red_alive = any(a.startswith("red_") for a in env.agents)
            blue_alive = any(a.startswith("blue_") for a in env.agents)
            if not red_alive and blue_alive and not blue_won:
                blue_won = True

        if blue_won:
            wins += 1

    env.close()
    return wins / n_episodes


# ---------------------------------------------------------------------------
# nash_exploitability_proxy
# ---------------------------------------------------------------------------


def nash_exploitability_proxy(
    policy: "MAPPOPolicy",
    pool: TeamOpponentPool,
    n_blue: int = 2,
    n_red: int = 2,
    n_episodes_per_opponent: int = 10,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict] = None,
) -> float:
    """Estimate Nash exploitability using the current self-play pool.

    Evaluates *policy* (as Blue) against **every** snapshot in *pool* and
    returns the *nemesis gap*:

    .. math::

        \\text{ExplProxy} = \\max_i (1 - \\text{wr}_i) - \\text{mean}_i (1 - \\text{wr}_i)

    where :math:`\\text{wr}_i` is the Blue win rate against opponent *i*.

    Interpretation:

    * ``0.0`` — policy performs equally well (or poorly) against every pool
      member; hard to exploit from the pool.
    * High value — one pool member significantly outperforms the average
      against the policy; the policy has an exploitable weakness.

    Parameters
    ----------
    policy:
        The Blue :class:`~models.mappo_policy.MAPPOPolicy` to evaluate.
    pool:
        :class:`TeamOpponentPool` of frozen opponent snapshots.
    n_blue, n_red:
        Team sizes.
    n_episodes_per_opponent:
        Episodes per pool member (default 10).  Smaller than the regular
        evaluation budget to keep the cost tractable.
    seed:
        Base random seed.
    env_kwargs:
        Extra kwargs forwarded to
        :class:`~envs.multi_battalion_env.MultiBattalionEnv`.

    Returns
    -------
    float
        Exploitability proxy in ``[0, 1]``.  Returns ``0.0`` if the pool
        is empty or all snapshots fail to load.
    """
    if pool.size == 0:
        return 0.0

    opp_win_rates: list[float] = []
    for path in pool.snapshot_paths:
        opponent = pool._load_snapshot(path)
        if opponent is None:
            continue
        blue_wr = evaluate_team_vs_pool(
            policy=policy,
            opponent=opponent,
            n_blue=n_blue,
            n_red=n_red,
            n_episodes=n_episodes_per_opponent,
            deterministic=True,
            seed=seed,
            env_kwargs=env_kwargs,
        )
        opp_win_rates.append(1.0 - blue_wr)

    if not opp_win_rates:
        return 0.0

    mean_opp = sum(opp_win_rates) / len(opp_win_rates)
    max_opp = max(opp_win_rates)
    return max_opp - mean_opp


# ---------------------------------------------------------------------------
# Internal helpers (team)
# ---------------------------------------------------------------------------


def _max_team_version_in_pool(pool: TeamOpponentPool) -> int:
    """Return the highest version number in *pool*'s team snapshot files.

    Snapshot file names follow the pattern
    ``team_snapshot_v{version:06d}.pt``.  Returns ``0`` if the pool is
    empty or no version can be parsed.
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
