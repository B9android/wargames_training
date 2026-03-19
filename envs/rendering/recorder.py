# envs/rendering/recorder.py
"""Episode recorder and replayer for BattalionEnv.

Typical usage — recording::

    from envs.rendering.recorder import EpisodeRecorder

    recorder = EpisodeRecorder()
    obs, _ = env.reset()
    recorder.record_step(0, env.blue, env.red)
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(env._step_count, env.blue, env.red, reward, info)
        done = terminated or truncated
    save_path = recorder.save("replays/my_episode.json")

Typical usage — replaying::

    from envs.rendering.recorder import EpisodeReplayer
    from envs.rendering.renderer import BattalionRenderer

    replayer = EpisodeReplayer.from_file("replays/my_episode.json")
    renderer = BattalionRenderer(map_width=1000.0, map_height=1000.0, fps=10)
    replayer.replay(renderer)
    renderer.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from envs.rendering.renderer import BattalionRenderer
    from envs.sim.battalion import Battalion
    from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# EpisodeRecorder
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Collects per-step snapshots of an episode for later JSON export.

    Each frame stores the full state of both battalions, the reward, and
    a JSON-serialisable subset of the ``info`` dict.
    """

    def __init__(self) -> None:
        self._frames: list[dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        blue: "Battalion",
        red: "Battalion",
        reward: float = 0.0,
        info: Optional[dict] = None,
    ) -> None:
        """Append one step's state to the recording.

        Parameters
        ----------
        step:
            Zero-based step index.
        blue, red:
            Current battalion states.
        reward:
            Scalar reward received this step.
        info:
            Info dict from :meth:`BattalionEnv.step`.  Non-serialisable
            values are silently dropped.
        """
        serialisable_info = {
            k: v for k, v in (info or {}).items() if _is_json_serialisable(v)
        }
        self._frames.append(
            {
                "step": int(step),
                "blue": _battalion_to_dict(blue),
                "red": _battalion_to_dict(red),
                "reward": float(reward),
                "info": serialisable_info,
            }
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise the recording to *path* as a JSON file.

        Parent directories are created automatically.  Returns the
        resolved :class:`~pathlib.Path` that was written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump({"frames": self._frames}, fh)
        return out.resolve()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    def frames(self) -> list[dict]:
        """Return a shallow copy of the list of recorded frame dicts."""
        return list(self._frames)


# ---------------------------------------------------------------------------
# EpisodeReplayer
# ---------------------------------------------------------------------------


class EpisodeReplayer:
    """Replays a previously-recorded episode through a :class:`BattalionRenderer`.

    Parameters
    ----------
    frames:
        List of frame dicts as produced by :class:`EpisodeRecorder`.
    """

    def __init__(self, frames: list[dict]) -> None:
        self._frames = list(frames)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "EpisodeReplayer":
        """Load a recording from a JSON file written by :meth:`EpisodeRecorder.save`.

        Parameters
        ----------
        path:
            Path to the ``.json`` file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If the JSON does not contain a top-level ``"frames"`` key.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["frames"])

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        renderer: "BattalionRenderer",
        terrain: Optional["TerrainMap"] = None,
    ) -> None:
        """Step through all recorded frames and render each one.

        The replay stops early if the renderer's window is closed by the
        user (i.e. :meth:`BattalionRenderer.render_frame` returns
        ``False``).

        Parameters
        ----------
        renderer:
            A live :class:`BattalionRenderer` instance.
        terrain:
            Optional terrain to use as the background.  When ``None`` no
            terrain overlay is drawn (plain background colour).
        """
        for frame in self._frames:
            blue = _dict_to_battalion(frame["blue"])
            red = _dict_to_battalion(frame["red"])
            alive = renderer.render_frame(
                blue,
                red,
                terrain=terrain,
                step=frame["step"],
                info=frame.get("info"),
            )
            if not alive:
                break

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Total number of frames in this replay."""
        return len(self._frames)


# ---------------------------------------------------------------------------
# Private serialisation helpers
# ---------------------------------------------------------------------------


def _battalion_to_dict(b: "Battalion") -> dict:
    """Serialise a :class:`~envs.sim.battalion.Battalion` to a plain dict."""
    return {
        "x": float(b.x),
        "y": float(b.y),
        "theta": float(b.theta),
        "strength": float(b.strength),
        "morale": float(b.morale),
        "team": int(b.team),
        "routed": bool(b.routed),
    }


def _dict_to_battalion(d: dict) -> "Battalion":
    """Reconstruct a :class:`~envs.sim.battalion.Battalion` from a dict."""
    from envs.sim.battalion import Battalion  # noqa: PLC0415 — avoid circular import

    b = Battalion(
        x=d["x"],
        y=d["y"],
        theta=d["theta"],
        strength=d["strength"],
        team=d["team"],
    )
    b.morale = float(d.get("morale", 1.0))
    b.routed = bool(d.get("routed", False))
    return b


def _is_json_serialisable(value: Any) -> bool:
    """Return ``True`` if *value* can be serialised to JSON without error."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False
