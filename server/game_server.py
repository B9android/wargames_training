# server/game_server.py
"""Asyncio WebSocket game server for E9.1 interactive wargame interface.

The server bridges the browser frontend, the BattalionEnv simulation, and the
AI policy (either scripted or via the ONNX policy server).

Architecture
------------
::

    Browser  ←──WebSocket──→  GameServer  ←──HTTP──→  PolicyServer (ONNX)
                                   │
                                BattalionEnv (Python sim)

Each browser tab opens a *game session* (:class:`GameSession`).  The session
owns one :class:`~envs.human_env.HumanEnv` instance and one
:class:`~envs.rendering.web_renderer.WebRenderer`.

WebSocket message protocol
--------------------------

Client → Server:

``{"type": "start", "scenario": "open_field", "difficulty": 5}``
    Begin a new episode.

``{"type": "action", "move": 0.5, "rotate": 0.0, "fire": 1.0}``
    Submit one human action (floats, clamped to ``[-1, 1]`` / ``[0, 1]``).
    If omitted, the action defaults to zero.

``{"type": "reset"}``
    Restart the current episode without changing scenario.

``{"type": "load_scenario", "scenario": "open_field", "difficulty": 3,
    "units": [...], "weather": {...}}``
    Load a custom scenario from the scenario editor.

``{"type": "export_replay"}``
    Request the recorded replay for the current (or last) episode.

``{"type": "load_replay", "replay": {...}}``
    Load a previously exported replay for playback.

``{"type": "replay_step"}``
    Advance the replay by one frame.

``{"type": "replay_seek", "index": 42}``
    Jump the replay cursor to a specific frame.

Server → Client:

``{"type": "frame", ...frame fields...}``
    Current game-state frame (from :class:`~envs.rendering.web_renderer.WebRenderer`).

``{"type": "episode_end", "outcome": "blue_wins", "step": 420}``
    Episode finished.

``{"type": "replay_frame", "frame": {...}, "index": 0, "total": N}``
    One frame from the loaded replay.

``{"type": "replay_export", "replay": {...}}``
    Full replay blob.

``{"type": "error", "message": "..."}``
    Something went wrong.

Usage::

    python -m server.game_server --host 0.0.0.0 --port 8765

or from code::

    from server.game_server import GameServer
    import asyncio

    server = GameServer(host="0.0.0.0", port=8765)
    asyncio.run(server.serve_forever())
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional websockets import (graceful at module level for tests).
# ---------------------------------------------------------------------------

try:
    import websockets  # type: ignore[import-untyped]
    _WS_AVAILABLE = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]
    _WS_AVAILABLE = False

# Type alias for the WebSocket server protocol (varies by websockets version).
WebSocketServerProtocol = object


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUTCOMES = frozenset({"blue_wins", "red_wins", "draw", "unknown"})

_DEFAULT_SCENARIO = "open_field"
_DEFAULT_DIFFICULTY = 5

# How many seconds the server waits between game-loop ticks when there is no
# pending human action.  Corresponds to ~10 game steps per second.
_TICK_INTERVAL: float = 0.1


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _determine_outcome(info: dict[str, Any]) -> str:
    """Map the BattalionEnv step-info dict to an outcome string."""
    blue_alive = info.get("blue_strength", 1.0) > 0.0
    red_alive = info.get("red_strength", 1.0) > 0.0
    if blue_alive and not red_alive:
        return "blue_wins"
    if red_alive and not blue_alive:
        return "red_wins"
    return "draw"


# ---------------------------------------------------------------------------
# AI policy client
# ---------------------------------------------------------------------------


class PolicyClient:
    """HTTP client for the ONNX policy server.

    Falls back to a zero-action (all zeros) if the policy server is
    unavailable, so the game loop is never blocked.

    Parameters
    ----------
    url:
        Base URL of the policy server, e.g. ``"http://localhost:8080"``.
    timeout:
        Request timeout in seconds.  Sub-10 ms responses are expected on the
        local loopback; a generous 0.5 s timeout guards against cold starts.
    """

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 0.5) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._obs_dim: int = 0  # inferred lazily on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass through the remote policy.

        Parameters
        ----------
        obs:
            Observation vector (1-D float32 array).

        Returns
        -------
        np.ndarray
            Action vector returned by the policy, or zeros on failure.
        """
        payload = json.dumps({"obs": [obs.tolist()]}).encode()
        req = urllib.request.Request(
            f"{self._url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
            output = body["output"][0]
            return np.array(output, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — never crash the game loop
            logger.warning("PolicyClient.predict failed: %s", exc)
            return np.zeros(3, dtype=np.float32)

    def is_available(self) -> bool:
        """Return ``True`` if the policy server is reachable."""
        try:
            with urllib.request.urlopen(f"{self._url}/health", timeout=self._timeout):
                return True
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Game session
# ---------------------------------------------------------------------------


class GameSession:
    """Manages a single player's game instance.

    One :class:`GameSession` is created per WebSocket connection.  It owns:

    * A :class:`~envs.human_env.HumanEnv` environment.
    * A :class:`~envs.rendering.web_renderer.WebRenderer` for JSON frames.
    * A :class:`~server.replay.ReplayRecorder` that logs every frame.
    * An optional :class:`PolicyClient` for ONNX-backed AI.

    Parameters
    ----------
    scenario:
        Built-in scenario name (key in :data:`~envs.human_env.SCENARIOS`).
    difficulty:
        Red opponent difficulty level (1–5).
    policy_url:
        Optional base URL of the ONNX policy server.  When provided and the
        server is reachable, the AI policy will be driven by ONNX; otherwise
        the scripted heuristic is used.
    """

    def __init__(
        self,
        scenario: str = _DEFAULT_SCENARIO,
        difficulty: int = _DEFAULT_DIFFICULTY,
        policy_url: Optional[str] = None,
    ) -> None:
        from envs.human_env import HumanEnv, SCENARIOS
        from envs.rendering.web_renderer import WebRenderer
        from server.replay import ReplayRecorder

        if scenario not in SCENARIOS:
            scenario = _DEFAULT_SCENARIO
        self._scenario = scenario
        self._difficulty = int(difficulty)

        self._env = HumanEnv(scenario=scenario, difficulty=self._difficulty)
        self._renderer = WebRenderer(
            map_width=self._env.map_width,
            map_height=self._env.map_height,
        )
        self._recorder = ReplayRecorder(
            scenario=scenario,
            difficulty=self._difficulty,
            map_width=self._env.map_width,
            map_height=self._env.map_height,
        )

        self._policy_client: Optional[PolicyClient] = None
        if policy_url:
            self._policy_client = PolicyClient(url=policy_url)

        # Latest human action (updated by handle_message).
        self._human_action: np.ndarray = np.zeros(3, dtype=np.float32)

        # Current observations (set on reset).
        self._obs: Optional[np.ndarray] = None
        self._step: int = 0
        self._done: bool = False
        self._last_replay: Optional[Any] = None  # server.replay.Replay

        # Replay player for playback mode.
        self._replay_player: Optional[Any] = None  # server.replay.ReplayPlayer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset the episode and return the initial frame dict."""
        obs, info = self._env.reset(seed=seed)
        self._obs = obs
        self._step = 0
        self._done = False
        self._human_action = np.zeros(3, dtype=np.float32)

        from server.replay import ReplayRecorder

        self._recorder = ReplayRecorder(
            scenario=self._scenario,
            difficulty=self._difficulty,
            map_width=self._env.map_width,
            map_height=self._env.map_height,
        )

        frame = self._make_frame(info)
        self._recorder.record(frame)
        return frame

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> tuple[dict[str, Any], bool]:
        """Advance the simulation one step.

        The human action (``self._human_action``) is consumed; it is reset to
        zero after each step to avoid repeated-input artefacts.

        Returns
        -------
        frame : dict
            JSON-serialisable game-state dict.
        done : bool
            ``True`` when the episode is finished.
        """
        if self._done or self._obs is None:
            return self._make_frame({}), True

        action = np.copy(self._human_action)
        self._human_action = np.zeros(3, dtype=np.float32)

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._obs = obs
        self._step += 1
        self._done = bool(terminated or truncated)

        frame = self._make_frame(info)
        self._recorder.record(frame)

        if self._done:
            outcome = _determine_outcome(info)
            self._last_replay = self._recorder.finish(outcome=outcome)

        return frame, self._done

    # ------------------------------------------------------------------
    # Human input
    # ------------------------------------------------------------------

    def apply_action(self, move: float, rotate: float, fire: float) -> None:
        """Store the latest human action for consumption on the next :meth:`step`.

        Parameters
        ----------
        move:
            Forward/backward in ``[-1, 1]``.
        rotate:
            CCW/CW rotation in ``[-1, 1]``.
        fire:
            Fire intensity in ``[0, 1]``.
        """
        self._human_action = np.array(
            [_clamp(move, -1.0, 1.0), _clamp(rotate, -1.0, 1.0), _clamp(fire, 0.0, 1.0)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def export_replay(self) -> Optional[dict[str, Any]]:
        """Return the last finished episode as a plain dict, or ``None``."""
        if self._last_replay is None:
            return None
        return self._last_replay.to_dict()

    def load_replay(self, replay_data: dict[str, Any]) -> bool:
        """Load a replay dict for playback.

        Parameters
        ----------
        replay_data:
            Dict previously produced by :meth:`export_replay`.

        Returns
        -------
        bool
            ``True`` on success.
        """
        from server.replay import Replay, ReplayPlayer

        try:
            replay = Replay.from_dict(replay_data)
            self._replay_player = ReplayPlayer(replay)
            return True
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("load_replay failed: %s", exc)
            return False

    def replay_step(self) -> Optional[dict[str, Any]]:
        """Advance the replay player one frame.

        Returns
        -------
        dict or None
            Message dict ready to send to the client, or ``None`` when the
            replay player is not loaded or the replay is finished.
        """
        if self._replay_player is None:
            return None
        frame = self._replay_player.step()
        if frame is None:
            return None
        return {
            "type": "replay_frame",
            "frame": frame,
            "index": self._replay_player.current_index - 1,
            "total": self._replay_player.total_frames,
        }

    def replay_seek(self, index: int) -> Optional[dict[str, Any]]:
        """Seek the replay player to *index* and return that frame.

        Returns
        -------
        dict or None
            Message dict, or ``None`` when the replay player is not loaded.
        """
        if self._replay_player is None:
            return None
        self._replay_player.seek(index)
        frame = self._replay_player.step()
        if frame is None:
            return None
        return {
            "type": "replay_frame",
            "frame": frame,
            "index": self._replay_player.current_index - 1,
            "total": self._replay_player.total_frames,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_frame(self, info: dict[str, Any]) -> dict[str, Any]:
        """Produce a ``"frame"``-type message from the current env state."""
        env = self._env._env  # inner BattalionEnv
        blue = env.blue
        red = env.red
        if blue is None or red is None:
            return {"type": "frame", "step": self._step, "map": {}, "blue": {}, "red": {}, "info": info}
        terrain = getattr(env, "terrain", None)
        frame = self._renderer.render_frame(
            blue=blue,
            red=red,
            terrain=terrain,
            step=self._step,
            info=dict(info),
        )
        frame["type"] = "frame"
        return frame


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------


class GameServer:
    """Asyncio WebSocket server that manages concurrent game sessions.

    Parameters
    ----------
    host:
        Bind address.
    port:
        Bind port.
    policy_url:
        Base URL of the ONNX policy server, forwarded to each
        :class:`GameSession`.  ``None`` disables the ONNX client.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        policy_url: Optional[str] = None,
    ) -> None:
        if not _WS_AVAILABLE:
            raise ImportError(
                "websockets is required.  Install it with: pip install websockets"
            )
        self._host = host
        self._port = port
        self._policy_url = policy_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def serve_forever(self) -> None:
        """Start listening for connections and serve until cancelled."""
        logger.info("GameServer starting on ws://%s:%d", self._host, self._port)
        async with websockets.serve(self._handle_connection, self._host, self._port):
            await asyncio.Future()  # run forever

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(self, ws: Any) -> None:
        """Manage one client connection (one :class:`GameSession`)."""
        session: Optional[GameSession] = None
        logger.info("Client connected: %s", ws.remote_address)

        try:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, ValueError) as exc:
                    await self._send_error(ws, f"Invalid JSON: {exc}")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(ws, "Message must be a JSON object.")
                    continue

                session = await self._dispatch(ws, msg, session)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in connection handler: %s", exc)
        finally:
            logger.info("Client disconnected: %s", ws.remote_address)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        ws: Any,
        msg: dict[str, Any],
        session: Optional[GameSession],
    ) -> Optional[GameSession]:
        """Route an incoming client message to the appropriate handler.

        Returns the (possibly updated) session.
        """
        msg_type = msg.get("type", "")

        if msg_type == "start":
            scenario = str(msg.get("scenario", _DEFAULT_SCENARIO))
            difficulty = int(msg.get("difficulty", _DEFAULT_DIFFICULTY))
            session = GameSession(
                scenario=scenario,
                difficulty=difficulty,
                policy_url=self._policy_url,
            )
            frame = session.reset()
            await ws.send(json.dumps(frame))
            # Kick off the game loop as a background task.
            asyncio.create_task(self._game_loop(ws, session))

        elif msg_type == "action":
            if session is None:
                await self._send_error(ws, "No active session. Send 'start' first.")
            else:
                session.apply_action(
                    move=float(msg.get("move", 0.0)),
                    rotate=float(msg.get("rotate", 0.0)),
                    fire=float(msg.get("fire", 0.0)),
                )

        elif msg_type == "reset":
            if session is None:
                await self._send_error(ws, "No active session. Send 'start' first.")
            else:
                frame = session.reset()
                await ws.send(json.dumps(frame))

        elif msg_type == "load_scenario":
            # Re-create the session with the requested scenario / difficulty.
            scenario = str(msg.get("scenario", _DEFAULT_SCENARIO))
            difficulty = int(msg.get("difficulty", _DEFAULT_DIFFICULTY))
            session = GameSession(
                scenario=scenario,
                difficulty=difficulty,
                policy_url=self._policy_url,
            )
            frame = session.reset()
            await ws.send(json.dumps(frame))
            asyncio.create_task(self._game_loop(ws, session))

        elif msg_type == "export_replay":
            if session is None:
                await self._send_error(ws, "No active session.")
            else:
                replay = session.export_replay()
                if replay is None:
                    await self._send_error(ws, "No completed episode to export.")
                else:
                    await ws.send(json.dumps({"type": "replay_export", "replay": replay}))

        elif msg_type == "load_replay":
            replay_data = msg.get("replay")
            if not isinstance(replay_data, dict):
                await self._send_error(ws, "Missing 'replay' field.")
            else:
                if session is None:
                    session = GameSession(policy_url=self._policy_url)
                ok = session.load_replay(replay_data)
                if not ok:
                    await self._send_error(ws, "Failed to load replay (invalid format).")
                else:
                    await ws.send(json.dumps({"type": "replay_loaded", "total": session._replay_player.total_frames if session._replay_player else 0}))

        elif msg_type == "replay_step":
            if session is None:
                await self._send_error(ws, "No active session.")
            else:
                result = session.replay_step()
                if result is None:
                    await ws.send(json.dumps({"type": "replay_done"}))
                else:
                    await ws.send(json.dumps(result))

        elif msg_type == "replay_seek":
            if session is None:
                await self._send_error(ws, "No active session.")
            else:
                index = int(msg.get("index", 0))
                result = session.replay_seek(index)
                if result is None:
                    await self._send_error(ws, "No replay loaded or index out of range.")
                else:
                    await ws.send(json.dumps(result))

        else:
            await self._send_error(ws, f"Unknown message type: {msg_type!r}")

        return session

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    async def _game_loop(self, ws: Any, session: GameSession) -> None:
        """Tick the simulation at :data:`_TICK_INTERVAL` seconds per step.

        Each tick:

        1. Steps the environment with the latest human action.
        2. Sends the resulting frame to the client.
        3. On episode end, sends an ``episode_end`` message.
        """
        try:
            while not session._done:
                frame, done = session.step()
                await ws.send(json.dumps(frame))
                if done:
                    outcome = (
                        session._last_replay.metadata.outcome
                        if session._last_replay
                        else "unknown"
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "episode_end",
                                "outcome": outcome,
                                "step": session._step,
                            }
                        )
                    )
                    break
                await asyncio.sleep(_TICK_INTERVAL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Game loop ended: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_error(ws: Any, message: str) -> None:
        """Send an ``error`` message to the client."""
        await ws.send(json.dumps({"type": "error", "message": message}))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E9.1 WebSocket game server — browser ↔ BattalionEnv ↔ AI policy"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--policy-url",
        default=None,
        help="Base URL of the ONNX policy server (e.g. http://localhost:8080). "
        "When omitted, the scripted heuristic drives the AI.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, _args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    server = GameServer(host=_args.host, port=_args.port, policy_url=_args.policy_url)
    asyncio.run(server.serve_forever())
