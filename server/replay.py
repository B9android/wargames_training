# server/replay.py
"""Replay recording, serialisation, and playback for E9.1.

A :class:`ReplayRecorder` is attached to a game session and records every
rendered frame as a JSON-serialisable dict.  At the end of an episode the
full replay can be exported to JSON for download / re-loading.

A :class:`ReplayPlayer` accepts a previously-exported replay dict and yields
frames one at a time so the frontend can step through them.

Wire protocol (server → client)
--------------------------------
``{"type": "replay_frame", "frame": {...}, "index": 0, "total": N}``
    One game-state frame emitted by :meth:`ReplayPlayer.step`.

``{"type": "replay_export", "replay": {...}}``
    Full replay blob ready for the client to download as ``.json``.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ReplayMetadata:
    """Metadata stored at the top of every replay export."""

    scenario: str
    difficulty: int
    map_width: float
    map_height: float
    #: ISO-8601 UTC timestamp when the recording started.
    recorded_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    #: Final outcome: ``"blue_wins"``, ``"red_wins"``, or ``"draw"``.
    outcome: str = "unknown"
    total_steps: int = 0


@dataclass
class Replay:
    """Complete replay exported from a finished episode."""

    metadata: ReplayMetadata
    #: List of frame dicts produced by :class:`~envs.rendering.web_renderer.WebRenderer`.
    frames: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the replay to a plain JSON-serialisable dict."""
        return {
            "metadata": asdict(self.metadata),
            "frames": list(self.frames),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialise the replay to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Replay":
        """Deserialise a replay from a plain dict.

        Parameters
        ----------
        data:
            Dict previously produced by :meth:`to_dict`.

        Raises
        ------
        KeyError
            If required keys are missing.
        """
        meta_raw = data["metadata"]
        meta = ReplayMetadata(
            scenario=meta_raw["scenario"],
            difficulty=int(meta_raw["difficulty"]),
            map_width=float(meta_raw["map_width"]),
            map_height=float(meta_raw["map_height"]),
            recorded_at=meta_raw.get("recorded_at", ""),
            outcome=meta_raw.get("outcome", "unknown"),
            total_steps=int(meta_raw.get("total_steps", 0)),
        )
        return cls(metadata=meta, frames=list(data.get("frames", [])))

    @classmethod
    def from_json(cls, text: str) -> "Replay":
        """Deserialise a replay from a JSON string."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class ReplayRecorder:
    """Records game frames during a live session.

    Attach one recorder per :class:`~server.game_server.GameSession`.  Call
    :meth:`record` once per game step with the frame dict from
    :class:`~envs.rendering.web_renderer.WebRenderer`.  When the episode
    ends, call :meth:`finish` to get the exportable :class:`Replay`.

    Parameters
    ----------
    scenario:
        Name of the active scenario (e.g. ``"open_field"``).
    difficulty:
        AI difficulty level (1–5).
    map_width, map_height:
        Map dimensions in metres.
    """

    def __init__(
        self,
        scenario: str,
        difficulty: int,
        map_width: float,
        map_height: float,
    ) -> None:
        self._metadata = ReplayMetadata(
            scenario=scenario,
            difficulty=difficulty,
            map_width=map_width,
            map_height=map_height,
        )
        self._frames: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, frame: dict[str, Any]) -> None:
        """Append one rendered frame to the recording.

        Parameters
        ----------
        frame:
            JSON-serialisable dict as returned by
            :meth:`~envs.rendering.web_renderer.WebRenderer.render_frame`.
        """
        self._frames.append(frame)

    def finish(self, outcome: str = "unknown") -> Replay:
        """Finalise the recording and return a :class:`Replay`.

        Parameters
        ----------
        outcome:
            Episode outcome: ``"blue_wins"``, ``"red_wins"``, or ``"draw"``.
        """
        meta = copy.copy(self._metadata)
        meta.outcome = outcome
        meta.total_steps = len(self._frames)
        return Replay(metadata=meta, frames=list(self._frames))

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


class ReplayPlayer:
    """Steps through a previously-recorded :class:`Replay`.

    Yields one frame dict per :meth:`step` call.  Supports seeking to an
    arbitrary frame index via :meth:`seek`.

    Parameters
    ----------
    replay:
        The replay to play back.
    """

    def __init__(self, replay: Replay) -> None:
        self._replay = replay
        self._index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_index(self) -> int:
        """Index of the frame that will be returned by the next :meth:`step`."""
        return self._index

    @property
    def total_frames(self) -> int:
        """Total number of frames in the replay."""
        return len(self._replay.frames)

    @property
    def metadata(self) -> ReplayMetadata:
        """Replay metadata."""
        return self._replay.metadata

    @property
    def done(self) -> bool:
        """``True`` when all frames have been yielded."""
        return self._index >= self.total_frames

    def seek(self, index: int) -> None:
        """Jump to a specific frame index.

        Parameters
        ----------
        index:
            Target frame index (0-based).  Clamped to ``[0, total_frames]``.
        """
        self._index = max(0, min(index, self.total_frames))

    def step(self) -> Optional[dict[str, Any]]:
        """Return the current frame and advance the index.

        Returns
        -------
        dict or None
            The frame dict, or ``None`` when the replay is finished.
        """
        if self.done:
            return None
        frame = self._replay.frames[self._index]
        self._index += 1
        return frame

    def iter_frames(self) -> Iterator[dict[str, Any]]:
        """Yield all remaining frames from the current index."""
        while not self.done:
            frame = self.step()
            if frame is not None:
                yield frame
