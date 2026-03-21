# api/coa_endpoint.py
"""Flask REST API for the Course of Action (COA) generator (Epic E5.2).

Exposes the COA generator as a lightweight HTTP service so that external
tools (e.g., planning dashboards, integration tests, human-in-the-loop
tools) can request scored COA lists for a given scenario without needing
a direct Python dependency on the training environment.

Endpoints
---------
``GET  /health``
    Returns ``{"status": "ok"}`` — useful as a liveness probe.

``POST /coas``
    Generate and return a ranked list of COAs.

    **Request body** (JSON, all fields optional):

    .. code-block:: json

        {
            "n_rollouts": 20,
            "n_coas":     5,
            "seed":       42,
            "strategies": ["aggressive", "defensive"],
            "env_kwargs": {
                "curriculum_level": 3,
                "randomize_terrain": false
            }
        }

    **Response** (JSON):

    .. code-block:: json

        {
            "coas": [
                {
                    "label": "aggressive",
                    "rank": 1,
                    "score": {
                        "win_rate": 0.65,
                        "draw_rate": 0.10,
                        "loss_rate": 0.25,
                        "blue_casualties": 0.12,
                        "red_casualties": 0.45,
                        "terrain_control": 0.60,
                        "composite": 0.55,
                        "n_rollouts": 20
                    },
                    "action_summary": { ... },
                    "seed": 37
                }
            ],
            "n_coas": 5
        }

Running the server::

    # Development
    python api/coa_endpoint.py

    # Via Flask CLI
    flask --app api.coa_endpoint run --port 5000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when this file is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Flask is required to run the COA API endpoint.  "
        "Install it with: pip install flask"
    ) from exc

from analysis.coa_generator import generate_coas, STRATEGY_LABELS

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int(value: Any, name: str, min_val: int = 1) -> int:
    """Parse and validate an integer parameter from the request body."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer, got {value!r}.")
    if v < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}, got {v}.")
    return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Liveness probe — always returns HTTP 200 with ``{"status": "ok"}``."""
    return jsonify({"status": "ok"}), 200


@app.route("/coas", methods=["POST"])
def coas() -> Any:
    """Generate and return a ranked list of COAs for the requested scenario.

    See module docstring for the full request/response schema.
    """
    body_raw = request.get_json(force=True, silent=True)
    if not isinstance(body_raw, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    body: dict = body_raw

    # ── Parse parameters ────────────────────────────────────────────────────
    try:
        n_rollouts: int = _parse_int(body.get("n_rollouts", 20), "n_rollouts")
        n_coas: int     = _parse_int(body.get("n_coas", 5),     "n_coas")
        seed_raw        = body.get("seed", None)
        seed: int | None = int(seed_raw) if seed_raw is not None else None
        strategies_raw  = body.get("strategies", None)
        env_kwargs_raw  = body.get("env_kwargs", {})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    if not isinstance(env_kwargs_raw, dict):
        return jsonify({"error": "'env_kwargs' must be a JSON object."}), 400
    env_kwargs: dict = env_kwargs_raw

    # Validate n_coas against the number of available archetypes
    if n_coas > len(STRATEGY_LABELS):
        return jsonify({
            "error": (
                f"n_coas ({n_coas}) exceeds the number of built-in strategy "
                f"archetypes ({len(STRATEGY_LABELS)}).  "
                f"Valid strategies: {list(STRATEGY_LABELS)}"
            )
        }), 400

    # Validate strategy labels when supplied
    if strategies_raw is not None:
        if not isinstance(strategies_raw, list):
            return jsonify({"error": "'strategies' must be a JSON array."}), 400
        invalid = [s for s in strategies_raw if s not in STRATEGY_LABELS]
        if invalid:
            return jsonify({
                "error": (
                    f"Unknown strategy labels: {invalid}.  "
                    f"Valid: {list(STRATEGY_LABELS)}"
                )
            }), 400
        if len(strategies_raw) < n_coas:
            return jsonify({
                "error": (
                    f"'strategies' has {len(strategies_raw)} entries but "
                    f"n_coas={n_coas} COAs were requested."
                )
            }), 400

    # Validate env_kwargs keys to avoid passing unexpected arguments.
    _allowed_env_keys = {
        "map_width", "map_height", "max_steps", "randomize_terrain",
        "hill_speed_factor", "curriculum_level",
    }
    unknown_keys = set(env_kwargs) - _allowed_env_keys
    if unknown_keys:
        return jsonify({
            "error": (
                f"Unknown env_kwargs keys: {sorted(unknown_keys)}.  "
                f"Allowed: {sorted(_allowed_env_keys)}"
            )
        }), 400

    # ── Run generator ───────────────────────────────────────────────────────
    try:
        result_coas = generate_coas(
            n_rollouts=n_rollouts,
            n_coas=n_coas,
            seed=seed,
            strategies=strategies_raw,
            env_kwargs=env_kwargs,
        )
    except (ValueError, TypeError) as exc:
        # Input-related issues raised during COA generation (e.g., invalid
        # env_kwargs values such as negative map_width or bad curriculum_level)
        # are treated as client errors.
        return jsonify({"error": f"Invalid request parameters: {exc}"}), 400
    except RuntimeError as exc:
        # Unexpected internal failures remain 500 errors.
        return jsonify({"error": f"COA generation failed: {exc}"}), 500

    return jsonify({
        "coas": [c.as_dict() for c in result_coas],
        "n_coas": len(result_coas),
    }), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=False)
