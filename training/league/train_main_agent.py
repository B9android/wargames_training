# training/league/train_main_agent.py
"""Main agent training loop for league training (E4.2).

Trains a MAPPO main agent against the full league pool using PFSP
(Prioritized Fictitious Self-Play) opponent sampling.  The main agent
periodically snapshots its current policy into the pool, and per-matchup
win rates plus an Elo rating are tracked and logged to W&B.

Classes
-------
MainAgentTrainer
    Core training loop combining :class:`~training.train_mappo.MAPPOTrainer`
    with the league infrastructure.

Functions
---------
make_pfsp_weight_fn
    Factory for parameterised PFSP weight functions (configurable temperature).
main
    Hydra entry-point.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra

from envs.multi_battalion_env import MultiBattalionEnv
from models.mappo_policy import MAPPOPolicy
from training.elo import EloRegistry
from training.league.agent_pool import AgentPool, AgentType
from training.league.diversity import DiversityTracker, TrajectoryBatch
from training.league.match_database import MatchDatabase
from training.league.matchmaker import LeagueMatchmaker
from training.league.nash import build_payoff_matrix, compute_nash_distribution, nash_entropy
from training.train_mappo import MAPPOTrainer

log = logging.getLogger(__name__)

__all__ = [
    "MainAgentTrainer",
    "make_pfsp_weight_fn",
]

# ---------------------------------------------------------------------------
# PFSP weight function factory
# ---------------------------------------------------------------------------


def make_pfsp_weight_fn(temperature: float = 1.0) -> Callable[[float], float]:
    """Return a PFSP weight function parameterised by *temperature*.

    The weight function is::

        f_T(w) = (1 - w) ** (1 / T)

    where *w* is the focal agent's win rate against a candidate opponent.

    * ``T = 1`` — standard hard-first weighting (``f(w) = 1 - w``).
    * ``T > 1`` — softer bias; approaches uniform sampling as ``T → ∞``.
    * ``T → 0`` — maximally hard; concentrates mass on the toughest opponent.

    Parameters
    ----------
    temperature:
        Temperature parameter ``T > 0``.

    Returns
    -------
    Callable[[float], float]
        PFSP weight function.

    Raises
    ------
    ValueError
        If *temperature* is not strictly positive.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"PFSP temperature must be > 0, got {temperature!r}"
        )
    _T = float(temperature)

    def _weight_fn(win_rate: float) -> float:
        return max((1.0 - float(win_rate)) ** (1.0 / _T), 0.0)

    return _weight_fn


# ---------------------------------------------------------------------------
# MainAgentTrainer
# ---------------------------------------------------------------------------


class MainAgentTrainer:
    """MAPPO training loop for league main agents with PFSP opponent sampling.

    The main agent trains against the full league pool.  Opponents are
    selected using PFSP — sampling probability is proportional to
    ``pfsp_weight_fn(win_rate(agent, opponent))``, which biases training
    towards historically difficult opponents.

    At configurable intervals the main agent's current policy is snapshotted
    and added back to the pool so that future agents must contend with it.

    Per-matchup win rates and an Elo rating are tracked and logged to W&B
    after each evaluation interval.

    Parameters
    ----------
    trainer:
        A pre-built :class:`~training.train_mappo.MAPPOTrainer` that provides
        the MAPPO rollout and policy-update mechanics.
    agent_pool:
        :class:`~training.league.agent_pool.AgentPool` containing the league
        snapshot registry.
    match_database:
        :class:`~training.league.match_database.MatchDatabase` for storing
        historical match outcomes.
    matchmaker:
        :class:`~training.league.matchmaker.LeagueMatchmaker` that selects
        PFSP-weighted opponents from the pool.
    elo_registry:
        :class:`~training.elo.EloRegistry` for tracking the main agent's
        Elo rating.
    agent_id:
        Unique identifier for the main agent being trained.
    snapshot_dir:
        Directory in which to write policy snapshot ``.pt`` files.
    snapshot_freq:
        Environment steps between consecutive policy snapshots (default
        100 000).  Each snapshot is added to *agent_pool* as a new record.
    eval_freq:
        Steps between evaluation rounds (default 50 000).  At each round
        the main agent is evaluated against the current PFSP-selected
        opponent and results are recorded.
    n_eval_episodes:
        Episodes per evaluation round (default 20).
    pfsp_temperature:
        Temperature ``T`` for the PFSP weight function (default ``1.0``,
        i.e. standard hard-first).
    log_interval:
        Steps between W&B training-metric logs (default 2 000).
    checkpoint_dir:
        Directory for saving trainer checkpoints (``None`` disables).
    checkpoint_freq:
        Steps between trainer checkpoints (default 100 000).
    """

    def __init__(
        self,
        trainer: MAPPOTrainer,
        agent_pool: AgentPool,
        match_database: MatchDatabase,
        matchmaker: LeagueMatchmaker,
        elo_registry: EloRegistry,
        agent_id: str,
        snapshot_dir: Path,
        snapshot_freq: int = 100_000,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        pfsp_temperature: float = 1.0,
        log_interval: int = 2_000,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 100_000,
    ) -> None:
        self._trainer = trainer
        self._agent_pool = agent_pool
        self._match_db = match_database
        self._matchmaker = matchmaker
        self._elo = elo_registry

        self.agent_id = agent_id
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_freq = int(snapshot_freq)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.pfsp_temperature = float(pfsp_temperature)
        self.log_interval = int(log_interval)
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_freq = int(checkpoint_freq)

        # Replace the matchmaker's weight function with the parameterised one.
        self._matchmaker.set_weight_function(make_pfsp_weight_fn(pfsp_temperature))

        # Internal state
        self._snapshot_version: int = 0
        self._current_opponent_id: Optional[str] = None
        # per-matchup win-rate accumulator {opponent_id: [outcomes]}
        self._matchup_outcomes: Dict[str, List[float]] = {}

        # Diversity tracker: accumulates per-agent behavioral embeddings and
        # computes pool-wide diversity scores for W&B logging (E4.6).
        self._diversity_tracker: DiversityTracker = DiversityTracker()

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> None:
        """Run the main agent training loop for *total_timesteps* steps.

        The loop:
        1. Optionally registers the main agent in the pool on the first run.
        2. Selects a PFSP opponent before each rollout.
        3. Collects a rollout and updates the policy.
        4. Every ``snapshot_freq`` steps: saves a snapshot to the pool.
        5. Every ``eval_freq`` steps: evaluates vs the current opponent,
           records the result, and updates Elo + W&B logs.
        6. Every ``log_interval`` steps: logs MAPPO training metrics.
        7. Every ``checkpoint_freq`` steps: saves a trainer checkpoint.

        Parameters
        ----------
        total_timesteps:
            Total environment steps to train for.
        """
        if total_timesteps < 1:
            raise ValueError(f"total_timesteps must be >= 1, got {total_timesteps!r}")

        # Ensure the main agent has an initial entry in the pool so PFSP
        # can reference it as an opponent for other agents.
        self._ensure_initial_snapshot()

        last_log = self._trainer._total_steps
        last_ckpt = self._trainer._total_steps
        last_snapshot = self._trainer._total_steps
        last_eval = self._trainer._total_steps

        log.info(
            "MainAgentTrainer: start | agent_id=%s | total_timesteps=%d"
            " | snapshot_freq=%d | eval_freq=%d | pfsp_temperature=%.2f",
            self.agent_id,
            total_timesteps,
            self.snapshot_freq,
            self.eval_freq,
            self.pfsp_temperature,
        )

        while self._trainer._total_steps < total_timesteps:
            # ------------------------------------------------------------------
            # PFSP opponent selection
            # ------------------------------------------------------------------
            self._refresh_opponent()

            # ------------------------------------------------------------------
            # Rollout + policy update
            # ------------------------------------------------------------------
            self._trainer.collect_rollout()
            losses = self._trainer.update_policy()
            total_steps = self._trainer._total_steps

            # ------------------------------------------------------------------
            # Periodic snapshot: save policy + register in pool
            # ------------------------------------------------------------------
            if (
                total_steps > last_snapshot
                and total_steps - last_snapshot >= self.snapshot_freq
            ):
                self._snapshot_version += 1
                snap_path = self._save_snapshot(self._snapshot_version)
                new_id = f"{self.agent_id}_v{self._snapshot_version:06d}"
                self._agent_pool.add(
                    snap_path,
                    AgentType.MAIN_AGENT,
                    agent_id=new_id,
                    version=self._snapshot_version,
                    metadata={"parent_agent_id": self.agent_id, "step": total_steps},
                    force=True,
                )
                log.info(
                    "MainAgentTrainer: snapshot v%d saved → pool size=%d",
                    self._snapshot_version,
                    self._agent_pool.size,
                )
                last_snapshot = total_steps

            # ------------------------------------------------------------------
            # Periodic evaluation: win rate, match recording, Elo, W&B
            # ------------------------------------------------------------------
            if (
                total_steps > last_eval
                and total_steps - last_eval >= self.eval_freq
            ):
                self._run_evaluation(total_steps)
                last_eval = total_steps

            # ------------------------------------------------------------------
            # Training metric logging
            # ------------------------------------------------------------------
            if total_steps - last_log >= self.log_interval:
                self._log_training(losses, total_steps)
                last_log = total_steps

            # ------------------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------------------
            if (
                self._checkpoint_dir is not None
                and total_steps > last_ckpt
                and total_steps - last_ckpt >= self.checkpoint_freq
            ):
                self._trainer._save_checkpoint(self._checkpoint_dir)
                last_ckpt = total_steps

        # ------------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------------
        if self._checkpoint_dir is not None:
            self._trainer._save_checkpoint(self._checkpoint_dir, suffix="_final")

        log.info(
            "MainAgentTrainer: training complete | steps=%d | snapshots=%d",
            self._trainer._total_steps,
            self._snapshot_version,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_initial_snapshot(self) -> None:
        """Register the main agent in the pool if not already present."""
        if self.agent_id not in self._agent_pool:
            snap_path = self._save_snapshot(version=0)
            self._agent_pool.add(
                snap_path,
                AgentType.MAIN_AGENT,
                agent_id=self.agent_id,
                version=0,
                metadata={"parent_agent_id": self.agent_id, "step": 0},
            )
            log.info(
                "MainAgentTrainer: registered initial snapshot for %s", self.agent_id
            )

    def _save_snapshot(self, version: int) -> Path:
        """Save the current policy weights to disk and return the path."""
        snap_path = self.snapshot_dir / f"main_agent_{self.agent_id}_v{version:06d}.pt"
        policy = self._trainer.policy
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
            snap_path,
        )
        log.debug("MainAgentTrainer: saved snapshot → %s", snap_path)
        return snap_path

    def _load_opponent(self, snapshot_path: Path) -> Optional[MAPPOPolicy]:
        """Load a frozen opponent policy from *snapshot_path*.

        Returns ``None`` if loading fails (e.g. file missing / corrupt).
        """
        try:
            data = torch.load(
                str(snapshot_path),
                map_location="cpu",
                weights_only=True,
            )
            policy = MAPPOPolicy(**data["kwargs"])
            policy.load_state_dict(data["state_dict"])
            policy = policy.to(self._trainer.device)
            policy.eval()
            return policy
        except Exception as exc:
            log.warning(
                "MainAgentTrainer: failed to load opponent from %s: %s",
                snapshot_path,
                exc,
            )
            return None

    def _refresh_opponent(self) -> None:
        """Select a PFSP opponent and set it as the Red policy."""
        if self._agent_pool.size <= 1:
            # Pool only contains the main agent itself → no valid opponent yet.
            # Fall back to the trainer's default Red behaviour.
            if self._current_opponent_id is not None:
                self._trainer.set_red_policy(None)
                self._current_opponent_id = None
            return

        opponent_record = self._matchmaker.select_opponent(
            self.agent_id,
            rng=self._trainer._rng,
        )
        if opponent_record is None:
            return

        if opponent_record.agent_id == self._current_opponent_id:
            # Same opponent as last rollout — no need to reload.
            return

        opponent_policy = self._load_opponent(opponent_record.snapshot_path)
        if opponent_policy is not None:
            self._trainer.set_red_policy(opponent_policy)
            self._current_opponent_id = opponent_record.agent_id
            if log.isEnabledFor(logging.DEBUG):
                win_rates = self._match_db.win_rates_for(self.agent_id)
                log.debug(
                    "MainAgentTrainer: opponent changed → %s (win_rate=%.3f)",
                    opponent_record.agent_id,
                    win_rates.get(opponent_record.agent_id, float("nan")),
                )

    def _evaluate_vs_opponent(self, opponent_id: str, snapshot_path: Path) -> float:
        """Evaluate the main agent against *opponent_id* and return win rate.

        Parameters
        ----------
        opponent_id:
            ID of the opponent (for logging only).
        snapshot_path:
            Path to the opponent's frozen policy snapshot.

        Returns
        -------
        float
            Blue win rate in ``[0, 1]``.
        """
        from training.self_play import evaluate_team_vs_pool

        opponent_policy = self._load_opponent(snapshot_path)
        if opponent_policy is None:
            log.warning(
                "MainAgentTrainer: skipping eval vs %s (load failed)", opponent_id
            )
            return 0.5  # unknown result — neutral

        win_rate = evaluate_team_vs_pool(
            policy=self._trainer.policy,
            opponent=opponent_policy,
            n_blue=self._trainer.n_blue,
            n_red=self._trainer.n_red,
            n_episodes=self.n_eval_episodes,
            deterministic=True,
            seed=self._trainer.seed,
        )
        return win_rate

    def _collect_trajectory(
        self,
        agent_id: str,
        snapshot_path: Optional[Path],
        n_steps: int = 200,
    ) -> Optional[TrajectoryBatch]:
        """Run a single episode with a frozen snapshot and collect a trajectory.

        The episode is run using the trainer's environment.  The policy from
        *snapshot_path* controls the Blue team; a scripted (default) Red
        policy is used.  Observations, actions, and normalised (x, y)
        positions are collected for the first Blue agent on each step.

        Parameters
        ----------
        agent_id:
            Identifier used to label the returned :class:`TrajectoryBatch`.
        snapshot_path:
            Path to the frozen ``.pt`` snapshot.  If ``None``, the trainer's
            *current* (live) policy is used.
        n_steps:
            Maximum number of environment steps to collect (default 200).

        Returns
        -------
        TrajectoryBatch | None
            Collected trajectory, or ``None`` on failure.
        """
        try:
            if snapshot_path is not None:
                policy = self._load_opponent(snapshot_path)
                if policy is None:
                    return None
            else:
                policy = self._trainer.policy

            env = self._trainer.env  # type: ignore[attr-defined]
            obs_dict, _ = env.reset()

            action_list: List[np.ndarray] = []
            pos_list: List[np.ndarray] = []

            # Collect up to n_steps transitions.
            for _ in range(n_steps):
                # Build observation tensor for the Blue team.
                blue_ids = [a for a in env.possible_agents if a.startswith("blue_")]
                if not blue_ids:
                    break

                obs_arr = np.stack(
                    [obs_dict.get(a, np.zeros(env.observation_space(a).shape)) for a in blue_ids],
                    axis=0,
                )  # (n_blue, obs_dim)

                obs_t = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0)  # (1, n_blue, obs_dim)
                state_t = obs_t.view(1, -1)  # flat global state proxy

                with torch.no_grad():
                    actions_t, _ = policy.act(obs_t, state_t, deterministic=True)
                actions_np = actions_t.squeeze(0).cpu().numpy()  # (n_blue, action_dim)

                # Aggregate all Blue agents' actions into one row per step.
                # Mean aggregation captures the team's overall action tendency
                # while keeping the embedding dimension independent of team size.
                mean_action = actions_np.mean(axis=0)  # (action_dim,)

                # Extract (x, y) position from the first Blue agent's obs.
                first_obs = obs_arr[0]  # (obs_dim,)
                pos_xy = first_obs[:2].astype(np.float32)  # normalised (x, y)

                action_list.append(mean_action.astype(np.float32))
                pos_list.append(pos_xy)

                # Build action dict for env.step.
                action_dict = {
                    a: actions_np[i] for i, a in enumerate(blue_ids)
                }
                # Default action for Red agents.
                red_ids = [a for a in env.possible_agents if a.startswith("red_")]
                for r in red_ids:
                    action_dict[r] = env.action_space(r).sample() * 0.0  # idle

                obs_dict, _, terminated, truncated, _ = env.step(action_dict)
                done = all(terminated.values()) or all(truncated.values())
                if done:
                    break

            if not action_list:
                return None

            return TrajectoryBatch(
                actions=np.stack(action_list, axis=0),
                positions=np.stack(pos_list, axis=0),
                agent_id=agent_id,
            )
        except Exception as exc:
            log.debug(
                "MainAgentTrainer: trajectory collection failed for %s: %s",
                agent_id,
                exc,
            )
            return None

    def _run_evaluation(self, total_steps: int) -> None:
        """Evaluate vs the current opponent, update records, and log to W&B."""
        if self._current_opponent_id is None:
            log.debug("MainAgentTrainer: eval skipped — no opponent yet")
            return

        try:
            opponent_record = self._agent_pool.get(self._current_opponent_id)
        except KeyError:
            log.debug(
                "MainAgentTrainer: eval skipped — opponent %s not in pool",
                self._current_opponent_id,
            )
            return

        win_rate = self._evaluate_vs_opponent(
            self._current_opponent_id, opponent_record.snapshot_path
        )

        # Accumulate per-matchup win rates for W&B logging.
        opp_id = self._current_opponent_id
        if opp_id not in self._matchup_outcomes:
            self._matchup_outcomes[opp_id] = []
        self._matchup_outcomes[opp_id].append(win_rate)

        # Record in MatchDatabase.
        self._match_db.record(
            agent_id=self.agent_id,
            opponent_id=opp_id,
            outcome=win_rate,
            metadata={"step": total_steps},
        )

        # Update Elo.
        try:
            elo_delta = self._elo.update(
                agent=self.agent_id,
                opponent=opp_id,
                outcome=win_rate,
                n_games=self.n_eval_episodes,
            )
            elo_rating = self._elo.get_rating(self.agent_id)
        except ValueError as exc:
            # Opponent may not be a valid Elo target (e.g. is itself a baseline).
            log.warning("MainAgentTrainer: Elo update skipped: %s", exc)
            elo_delta = 0.0
            elo_rating = self._elo.get_rating(self.agent_id)

        log.info(
            "MainAgentTrainer: eval | opponent=%s | win_rate=%.3f"
            " | elo=%.1f (delta=%+.1f) | step=%d",
            opp_id,
            win_rate,
            elo_rating,
            elo_delta,
            total_steps,
        )

        # W&B: per-matchup win rate + Elo.
        metrics: dict = {
            f"matchup/win_rate/{opp_id}": win_rate,
            "elo/main_agent": elo_rating,
            "elo/delta": elo_delta,
        }
        # Also log aggregate win rates across all matchups seen so far.
        for mid, outcomes in self._matchup_outcomes.items():
            if outcomes:
                metrics[f"matchup/mean_win_rate/{mid}"] = float(
                    sum(outcomes) / len(outcomes)
                )
        # Nash distribution entropy over the current league pool.
        # Build the payoff matrix from pre-cached per-agent win-rate dicts
        # (one O(num_matches) pass per agent) to avoid the O(N^2 * num_matches)
        # cost of calling win_rate() separately for each matrix cell.
        all_records = self._agent_pool.list()
        if len(all_records) >= 2:
            agent_ids = [r.agent_id for r in all_records]
            win_rates_cache = {aid: self._match_db.win_rates_for(aid) for aid in agent_ids}
            payoff = build_payoff_matrix(
                agent_ids,
                lambda ai, aj: win_rates_cache.get(ai, {}).get(aj),
                self._matchmaker.unknown_win_rate,
            )
            nash_dist = compute_nash_distribution(payoff)
            metrics["league/nash_entropy"] = nash_entropy(nash_dist)

        # Collect behavioral trajectories and update diversity tracker (E4.6).
        # Sample the main agent's current behaviour and the opponent's behaviour,
        # then compute the pool-wide diversity score for W&B logging.
        self._update_diversity_tracker(opp_id, opponent_record.snapshot_path)
        if self._diversity_tracker.pool_size >= 2:
            div_score = self._diversity_tracker.diversity_score()
            metrics["league/diversity_score"] = div_score
            log.info(
                "MainAgentTrainer: diversity_score=%.4f (pool=%d agents tracked)",
                div_score,
                self._diversity_tracker.pool_size,
            )

        metrics["eval/step"] = total_steps
        if wandb.run is not None:
            wandb.log(metrics, step=total_steps)

        # Persist Elo ratings.
        try:
            self._elo.save()
        except ValueError:
            pass  # No path configured — in-memory only.

    def _update_diversity_tracker(
        self,
        opponent_id: str,
        opponent_snapshot_path: Path,
    ) -> None:
        """Collect trajectories for the main agent and opponent; update tracker.

        Runs a short rollout episode for each party to obtain behavioral
        embeddings.  Silently skips if trajectory collection fails (e.g. the
        environment is unavailable or the snapshot is missing).

        Parameters
        ----------
        opponent_id:
            ID of the opponent agent.
        opponent_snapshot_path:
            Path to the opponent's frozen snapshot ``.pt`` file.
        """
        # Main agent — use live policy (snapshot_path=None).
        main_traj = self._collect_trajectory(self.agent_id, snapshot_path=None)
        if main_traj is not None:
            self._diversity_tracker.update(self.agent_id, main_traj)

        # Opponent — use frozen snapshot.
        opp_traj = self._collect_trajectory(
            opponent_id, snapshot_path=opponent_snapshot_path
        )
        if opp_traj is not None:
            self._diversity_tracker.update(opponent_id, opp_traj)

    def _log_training(self, losses: dict, total_steps: int) -> None:
        """Log MAPPO training losses and league state to W&B."""
        metrics: dict = {
            "train/policy_loss": losses.get("policy_loss", float("nan")),
            "train/value_loss": losses.get("value_loss", float("nan")),
            "train/entropy": losses.get("entropy", float("nan")),
            "train/total_loss": losses.get("total_loss", float("nan")),
            "league/pool_size": self._agent_pool.size,
            "league/snapshot_version": self._snapshot_version,
            "train/step": total_steps,
        }
        if self._current_opponent_id is not None:
            win_rates = self._match_db.win_rates_for(self.agent_id)
            opp_wr = win_rates.get(self._current_opponent_id, float("nan"))
            metrics["train/current_opponent_win_rate"] = opp_wr

        if wandb.run is not None:
            wandb.log(metrics, step=total_steps)

        log.info(
            "[%d steps] policy_loss=%.4f value_loss=%.4f entropy=%.4f pool_size=%d",
            total_steps,
            losses.get("policy_loss", float("nan")),
            losses.get("value_loss", float("nan")),
            losses.get("entropy", float("nan")),
            self._agent_pool.size,
        )


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="league/main_agent",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run a league main-agent training session from a Hydra configuration.

    Parameters
    ----------
    cfg:
        Hydra-merged configuration (see ``configs/league/main_agent.yaml``).
    """
    logging.basicConfig(level=getattr(logging, cfg.logging.level, logging.INFO))

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        reinit=True,
    )
    log.info("W&B run: %s", run.url if run else "offline")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = MultiBattalionEnv(
        n_blue=int(cfg.env.n_blue),
        n_red=int(cfg.env.n_red),
        map_width=float(cfg.env.map_width),
        map_height=float(cfg.env.map_height),
        max_steps=int(cfg.env.max_steps),
        randomize_terrain=bool(cfg.env.randomize_terrain),
        hill_speed_factor=float(cfg.env.hill_speed_factor),
        visibility_radius=float(cfg.env.visibility_radius),
    )
    obs_dim: int = env._obs_dim
    action_dim: int = env._act_space.shape[0]
    state_dim: int = env._state_dim

    log.info(
        "Env: n_blue=%d n_red=%d obs_dim=%d action_dim=%d state_dim=%d",
        env.n_blue, env.n_red, obs_dim, action_dim, state_dim,
    )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    share_params = bool(
        OmegaConf.select(cfg, "training.share_parameters", default=True)
    )
    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=env.n_blue,
        share_parameters=share_params,
        actor_hidden_sizes=tuple(cfg.training.actor_hidden_sizes),
        critic_hidden_sizes=tuple(cfg.training.critic_hidden_sizes),
    )

    # ------------------------------------------------------------------
    # MAPPO Trainer
    # ------------------------------------------------------------------
    trainer = MAPPOTrainer(
        policy=policy,
        env=env,
        n_steps=int(cfg.training.n_steps),
        n_epochs=int(cfg.training.n_epochs),
        batch_size=int(cfg.training.batch_size),
        lr=float(cfg.training.lr),
        gamma=float(cfg.training.gamma),
        gae_lambda=float(cfg.training.gae_lambda),
        clip_range=float(cfg.training.clip_range),
        vf_coef=float(cfg.training.vf_coef),
        ent_coef=float(cfg.training.ent_coef),
        max_grad_norm=float(cfg.training.max_grad_norm),
        device=str(OmegaConf.select(cfg, "training.device", default="cpu")),
        seed=int(cfg.training.seed),
    )

    # ------------------------------------------------------------------
    # League infrastructure
    # ------------------------------------------------------------------
    league_cfg = cfg.league
    pool_manifest = _PROJECT_ROOT / str(league_cfg.pool_manifest)
    match_db_path = _PROJECT_ROOT / str(league_cfg.match_db_path)
    elo_path = _PROJECT_ROOT / str(league_cfg.elo_registry_path)
    snapshot_dir = _PROJECT_ROOT / str(league_cfg.snapshot_dir)

    agent_pool = AgentPool(
        pool_manifest=pool_manifest,
        max_size=int(OmegaConf.select(league_cfg, "pool_max_size", default=200)),
    )
    match_db = MatchDatabase(db_path=match_db_path)
    matchmaker = LeagueMatchmaker(
        agent_pool=agent_pool,
        match_database=match_db,
        unknown_win_rate=float(
            OmegaConf.select(league_cfg, "unknown_win_rate", default=0.5)
        ),
    )
    elo_registry = EloRegistry(path=elo_path)

    checkpoint_dir = (
        _PROJECT_ROOT / str(league_cfg.checkpoint_dir)
        if OmegaConf.select(league_cfg, "checkpoint_dir", default=None) is not None
        else None
    )

    # ------------------------------------------------------------------
    # MainAgentTrainer
    # ------------------------------------------------------------------
    league_trainer = MainAgentTrainer(
        trainer=trainer,
        agent_pool=agent_pool,
        match_database=match_db,
        matchmaker=matchmaker,
        elo_registry=elo_registry,
        agent_id=str(league_cfg.agent_id),
        snapshot_dir=snapshot_dir,
        snapshot_freq=int(league_cfg.snapshot_freq),
        eval_freq=int(league_cfg.eval_freq),
        n_eval_episodes=int(league_cfg.n_eval_episodes),
        pfsp_temperature=float(
            OmegaConf.select(league_cfg, "pfsp_temperature", default=1.0)
        ),
        log_interval=int(cfg.wandb.log_freq),
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=int(
            OmegaConf.select(league_cfg, "checkpoint_freq", default=100_000)
        ),
    )

    league_trainer.train(total_timesteps=int(cfg.training.total_timesteps))

    env.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
