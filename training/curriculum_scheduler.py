# training/curriculum_scheduler.py
"""Curriculum scheduler for the E2.3 staged 2v2 training pipeline.

Implements a three-stage curriculum that bootstraps cooperative skills by
progressing from 1v1 (using a frozen v1 checkpoint) through an asymmetric 2v1
scenario to the full 2v2 cooperative challenge:

    Stage STAGE_1V1  →  1v1  (frozen v1 policy seeds initial weights)
    Stage STAGE_2V1  →  2v1  (Blue 2-agent advantage; Red stationary)
    Stage STAGE_2V2  →  2v2  (symmetric; Blue must cooperate to win)

Promotion logic
---------------
A rolling win-rate window of the most recent ``win_rate_window`` episodes is
tracked.  When the mean win rate reaches ``promote_threshold`` the scheduler
advances to the next stage and swaps out the :class:`~envs.multi_battalion_env.
MultiBattalionEnv` for the new scenario.  All stage transitions are logged as
W&B events.

Typical usage::

    from training.curriculum_scheduler import CurriculumScheduler, CurriculumStage

    scheduler = CurriculumScheduler(
        promote_threshold=0.7,
        win_rate_window=50,
    )
    scheduler.record_episode(win=True)
    if scheduler.should_promote():
        new_stage = scheduler.promote()

Checkpoint transfer
-------------------
The :func:`load_v1_weights_into_mappo` helper copies the shared-trunk weights
from a SB3 PPO v1 ``.zip`` checkpoint into the actor trunk of a
:class:`~models.mappo_policy.MAPPOPolicy`, enabling warm-start from
single-agent training.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum
from pathlib import Path
from typing import Deque, Optional

import torch

log = logging.getLogger(__name__)

__all__ = [
    "CurriculumStage",
    "CurriculumScheduler",
    "load_v1_weights_into_mappo",
]


# ---------------------------------------------------------------------------
# Stage enum
# ---------------------------------------------------------------------------


class CurriculumStage(IntEnum):
    """Ordered curriculum stages.

    The integer value doubles as the stage index used when indexing into the
    ``STAGE_ENV_KWARGS`` mapping defined in :data:`STAGE_ENV_KWARGS`.
    """

    STAGE_1V1 = 0  #: Bootstrap stage — 1 Blue vs 1 Red (frozen v1 checkpoint)
    STAGE_2V1 = 1  #: Asymmetric advantage — 2 Blue vs 1 Red
    STAGE_2V2 = 2  #: Full cooperative challenge — 2 Blue vs 2 Red


#: Default environment keyword arguments for each curriculum stage.
#: These match the scenario YAML files in ``configs/scenarios/``.
STAGE_ENV_KWARGS: dict[CurriculumStage, dict] = {
    CurriculumStage.STAGE_1V1: {"n_blue": 1, "n_red": 1},
    CurriculumStage.STAGE_2V1: {"n_blue": 2, "n_red": 1},
    CurriculumStage.STAGE_2V2: {"n_blue": 2, "n_red": 2},
}

#: Human-readable labels for W&B logging.
STAGE_LABELS: dict[CurriculumStage, str] = {
    CurriculumStage.STAGE_1V1: "1v1",
    CurriculumStage.STAGE_2V1: "2v1",
    CurriculumStage.STAGE_2V2: "2v2",
}


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Tracks episode outcomes and decides when to promote the curriculum stage.

    Parameters
    ----------
    promote_threshold:
        Rolling win rate (in ``[0, 1]``) that must be reached to advance to
        the next stage.  Defaults to ``0.70`` (70 % win rate).
    win_rate_window:
        Number of most-recent episodes used to compute the rolling win rate.
        Defaults to ``50``.
    initial_stage:
        The curriculum stage to begin from.  Defaults to
        :attr:`CurriculumStage.STAGE_1V1`.
    """

    def __init__(
        self,
        promote_threshold: float = 0.70,
        win_rate_window: int = 50,
        initial_stage: CurriculumStage = CurriculumStage.STAGE_1V1,
    ) -> None:
        if not (0.0 < promote_threshold <= 1.0):
            raise ValueError(
                f"promote_threshold must be in (0, 1], got {promote_threshold}"
            )
        if win_rate_window < 1:
            raise ValueError(
                f"win_rate_window must be >= 1, got {win_rate_window}"
            )

        self.promote_threshold = float(promote_threshold)
        self.win_rate_window = int(win_rate_window)
        self._stage: CurriculumStage = initial_stage

        # Rolling window of episode outcomes (True=win, False=loss/draw)
        self._outcomes: Deque[bool] = deque(maxlen=win_rate_window)

        # Cumulative episode counter
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stage(self) -> CurriculumStage:
        """The current curriculum stage."""
        return self._stage

    @property
    def stage_label(self) -> str:
        """Human-readable label for the current stage (e.g. ``"2v1"``)."""
        return STAGE_LABELS[self._stage]

    @property
    def is_final_stage(self) -> bool:
        """``True`` when the scheduler is already at the last stage (2v2)."""
        return self._stage == CurriculumStage.STAGE_2V2

    @property
    def total_episodes(self) -> int:
        """Total episodes recorded since creation."""
        return self._total_episodes

    # ------------------------------------------------------------------
    # Episode tracking
    # ------------------------------------------------------------------

    def record_episode(self, win: bool) -> None:
        """Record the outcome of a completed episode.

        Parameters
        ----------
        win:
            ``True`` if the Blue team won the episode, ``False`` otherwise.
        """
        self._outcomes.append(bool(win))
        self._total_episodes += 1

    def win_rate(self) -> float:
        """Return the rolling win rate over the last ``win_rate_window`` episodes.

        Returns ``0.0`` if no episodes have been recorded yet.
        """
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    # ------------------------------------------------------------------
    # Promotion logic
    # ------------------------------------------------------------------

    def should_promote(self) -> bool:
        """Return ``True`` if promotion criteria are met.

        Criteria:
        * At least ``win_rate_window`` episodes have been recorded since the
          last promotion (or since creation).
        * The rolling win rate meets or exceeds ``promote_threshold``.
        * The current stage is not already the final stage.
        """
        if self.is_final_stage:
            return False
        if len(self._outcomes) < self.win_rate_window:
            return False
        return self.win_rate() >= self.promote_threshold

    def promote(self) -> CurriculumStage:
        """Advance to the next curriculum stage and reset the outcome window.

        Returns
        -------
        The new :class:`CurriculumStage` after promotion.

        Raises
        ------
        RuntimeError
            If called when already at the final stage.
        """
        if self.is_final_stage:
            raise RuntimeError(
                "Cannot promote past the final curriculum stage "
                f"({self.stage_label})."
            )

        old_stage = self._stage
        self._stage = CurriculumStage(int(self._stage) + 1)
        # Reset rolling window so the new stage's win rate is measured fresh.
        self._outcomes.clear()

        log.info(
            "Curriculum promoted: %s → %s after %d total episodes",
            STAGE_LABELS[old_stage],
            STAGE_LABELS[self._stage],
            self._total_episodes,
        )
        return self._stage

    def env_kwargs(self) -> dict:
        """Return the environment kwargs for the current stage.

        These match the ``n_blue``/``n_red`` values in the scenario YAML files
        under ``configs/scenarios/``.
        """
        return dict(STAGE_ENV_KWARGS[self._stage])

    def wandb_metrics(self) -> dict:
        """Return a dict of W&B metrics for the current state.

        Keys:
        * ``curriculum/stage`` — integer stage index
        * ``curriculum/stage_label`` — e.g. ``"2v1"``
        * ``curriculum/win_rate`` — rolling win rate in ``[0, 1]``
        * ``curriculum/total_episodes`` — cumulative episode count
        """
        return {
            "curriculum/stage": int(self._stage),
            "curriculum/stage_label": self.stage_label,
            "curriculum/win_rate": self.win_rate(),
            "curriculum/total_episodes": self._total_episodes,
        }

    def log_promotion_event(self, total_steps: int, wandb_run: object = None) -> None:
        """Log a curriculum stage-promotion event to W&B.

        Parameters
        ----------
        total_steps:
            Current total environment step count (used as the W&B x-axis).
        wandb_run:
            An active ``wandb.run`` object.  When ``None`` the event is only
            written to the Python logger.
        """
        metrics = self.wandb_metrics()
        metrics["curriculum/promotion_step"] = total_steps
        log.info(
            "Curriculum stage transition → %s at step %d (win_rate=%.3f, "
            "total_episodes=%d)",
            self.stage_label,
            total_steps,
            metrics["curriculum/win_rate"],
            self._total_episodes,
        )
        if wandb_run is not None:
            try:
                import wandb  # local import to keep module importable without wandb

                wandb.log(metrics, step=total_steps)
            except Exception as exc:  # pragma: no cover
                log.warning("W&B logging failed during promotion: %s", exc)


# ---------------------------------------------------------------------------
# Checkpoint transfer helper
# ---------------------------------------------------------------------------


def load_v1_weights_into_mappo(
    v1_checkpoint_path: str | Path,
    mappo_policy: torch.nn.Module,
    *,
    strict: bool = False,
) -> dict:
    """Copy shared-trunk weights from a SB3 PPO v1 checkpoint into a MAPPO actor.

    The v1 SB3 checkpoint stores the policy network under the key
    ``policy`` inside the zip.  The features extractor (``mlp_extractor``) and
    action network (``action_net``) weights are mapped onto the MAPPO actor's
    ``trunk`` and ``action_mean`` layers where shapes match.

    Parameters
    ----------
    v1_checkpoint_path:
        Path to the SB3 ``.zip`` checkpoint produced by ``training/train.py``.
    mappo_policy:
        A :class:`~models.mappo_policy.MAPPOPolicy` instance whose actor
        trunk will be warm-started.
    strict:
        When ``True``, raises on any shape or key mismatch.  When ``False``
        (default) mismatches are logged as warnings and skipped, allowing
        partial weight transfer when the 1v1 obs-dim differs from 2v2.

    Returns
    -------
    A dict with keys ``"loaded"`` (list of transferred layer names) and
    ``"skipped"`` (list of skipped layer names).
    """
    import zipfile
    import io

    v1_path = Path(v1_checkpoint_path)
    if not v1_path.exists():
        raise FileNotFoundError(f"v1 checkpoint not found: {v1_path}")

    # SB3 saves the PyTorch state dict as "policy.pth" inside the zip.
    with zipfile.ZipFile(v1_path, "r") as zf:
        if "policy.pth" not in zf.namelist():
            raise ValueError(
                f"Expected 'policy.pth' inside {v1_path}; "
                f"found: {zf.namelist()}"
            )
        with zf.open("policy.pth") as f:
            v1_state: dict = torch.load(io.BytesIO(f.read()), map_location="cpu")

    # Build a name-mapping from v1 feature-extractor keys to MAPPO actor keys.
    # SB3 BattalionMlpPolicy stores weights under mlp_extractor.* and action_net.*
    # The MAPPO actor stores them under actor.trunk.* and actor.action_mean.*
    # (or actors.0.trunk.* etc. when share_parameters=False).
    transferred: list[str] = []
    skipped: list[str] = []

    mappo_state = mappo_policy.state_dict()
    new_state: dict = {}

    # We prefix-map:  "mlp_extractor." → "actor.trunk." (shared actor)
    #                 "action_net."    → "actor.action_mean."
    # Only transfer layers whose shapes match.
    _prefix_map = [
        ("mlp_extractor.", "actor.trunk."),
        ("action_net.", "actor.action_mean."),
    ]

    for v1_key, v1_tensor in v1_state.items():
        mapped = False
        for v1_prefix, mappo_prefix in _prefix_map:
            if v1_key.startswith(v1_prefix):
                mappo_key = mappo_prefix + v1_key[len(v1_prefix):]
                if mappo_key in mappo_state:
                    if mappo_state[mappo_key].shape == v1_tensor.shape:
                        new_state[mappo_key] = v1_tensor
                        transferred.append(mappo_key)
                    else:
                        msg = (
                            f"Shape mismatch for {mappo_key}: "
                            f"v1={v1_tensor.shape} mappo={mappo_state[mappo_key].shape}"
                        )
                        if strict:
                            raise ValueError(msg)
                        log.warning("load_v1_weights_into_mappo: skipping %s", msg)
                        skipped.append(mappo_key)
                else:
                    skipped.append(mappo_key)
                mapped = True
                break
        if not mapped:
            skipped.append(v1_key)

    if new_state:
        mappo_state.update(new_state)
        mappo_policy.load_state_dict(mappo_state, strict=False)
        log.info(
            "load_v1_weights_into_mappo: transferred %d layers, skipped %d layers",
            len(transferred),
            len(skipped),
        )
    else:
        log.warning(
            "load_v1_weights_into_mappo: no layers transferred from %s", v1_path
        )

    return {"loaded": transferred, "skipped": skipped}
