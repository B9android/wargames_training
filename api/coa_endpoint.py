# api/coa_endpoint.py
"""Flask REST API for the Course of Action (COA) generator (Epics E5.2 / E9.2).

Exposes the COA generator as a lightweight HTTP service so that external
tools (e.g., planning dashboards, integration tests, human-in-the-loop
tools) can request scored COA lists for a given scenario without needing
a direct Python dependency on the training environment.

Endpoints
---------
``GET  /health``
    Returns ``{"status": "ok"}`` — useful as a liveness probe.

``POST /coas``
    Generate and return a ranked list of battalion-level COAs.

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

``POST /corps/coas``   (E9.2)
    Generate up to 10 ranked corps-level COAs.

    **Request body** (JSON, all fields optional):

    .. code-block:: json

        {
            "n_rollouts": 10,
            "n_coas":     10,
            "seed":       42,
            "strategies": ["full_advance", "pincer_attack"],
            "explain":    false,
            "env_kwargs": {"n_divisions": 3}
        }

``POST /corps/coas/modify``   (E9.2)
    Modify an existing COA and re-simulate it.

    **Request body** (JSON):

    .. code-block:: json

        {
            "coa": { ... },
            "modification": {
                "strategy_override": "pincer_attack",
                "n_rollouts": 5,
                "division_command_overrides": {"0": 2}
            },
            "env_kwargs": {"n_divisions": 3}
        }

``POST /corps/coas/explain``   (E9.2)
    Explain a COA's key decisions.

    **Request body** (JSON):

    .. code-block:: json

        {
            "coa": { ... },
            "env_kwargs": {"n_divisions": 3}
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

from analysis.coa_generator import (
    generate_coas,
    STRATEGY_LABELS,
    CorpsCOAGenerator,
    CorpsCOAScore,
    CorpsCourseOfAction,
    COAExplanation,
    COAModification,
    CORPS_STRATEGY_LABELS,
)

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


# ===========================================================================
# Corps-level endpoints (E9.2)
# ===========================================================================

_ALLOWED_CORPS_ENV_KEYS = {
    "n_divisions", "n_brigades_per_division", "n_red_divisions",
    "max_steps", "map_width", "map_height",
}


def _parse_corps_env_kwargs(raw: Any) -> dict:
    """Validate and return corps env_kwargs or raise ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("'env_kwargs' must be a JSON object.")
    unknown = set(raw) - _ALLOWED_CORPS_ENV_KEYS
    if unknown:
        raise ValueError(
            f"Unknown env_kwargs keys: {sorted(unknown)}.  "
            f"Allowed: {sorted(_ALLOWED_CORPS_ENV_KEYS)}"
        )
    return raw


@app.route("/corps/coas", methods=["POST"])
def corps_coas() -> Any:
    """Generate and return up to 10 ranked corps-level COAs (E9.2).

    Request body (all optional):

    .. code-block:: json

        {
            "n_rollouts": 10,
            "n_coas": 10,
            "seed": 42,
            "strategies": ["full_advance", "pincer_attack"],
            "explain": false,
            "env_kwargs": {"n_divisions": 3}
        }
    """
    from analysis.coa_generator import generate_corps_coas

    body_raw = request.get_json(force=True, silent=True)
    if not isinstance(body_raw, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    body: dict = body_raw

    try:
        n_rollouts: int = _parse_int(body.get("n_rollouts", 10), "n_rollouts")
        n_coas: int     = _parse_int(body.get("n_coas", 10),     "n_coas")
        seed_raw        = body.get("seed", None)
        if seed_raw is not None:
            try:
                seed: int | None = int(seed_raw)
            except (TypeError, ValueError):
                raise ValueError(f"'seed' must be an integer, got {seed_raw!r}.")
        else:
            seed = None
        strategies_raw  = body.get("strategies", None)
        explain: bool   = bool(body.get("explain", False))
        env_kwargs      = _parse_corps_env_kwargs(body.get("env_kwargs", {}))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    if n_coas > len(CORPS_STRATEGY_LABELS):
        return jsonify({
            "error": (
                f"n_coas ({n_coas}) exceeds available corps strategy archetypes "
                f"({len(CORPS_STRATEGY_LABELS)}).  Valid: {list(CORPS_STRATEGY_LABELS)}"
            )
        }), 400

    if strategies_raw is not None:
        if not isinstance(strategies_raw, list):
            return jsonify({"error": "'strategies' must be a JSON array."}), 400
        invalid = [s for s in strategies_raw if s not in CORPS_STRATEGY_LABELS]
        if invalid:
            return jsonify({
                "error": (
                    f"Unknown corps strategy labels: {invalid}.  "
                    f"Valid: {list(CORPS_STRATEGY_LABELS)}"
                )
            }), 400
        if len(strategies_raw) < n_coas:
            return jsonify({
                "error": (
                    f"'strategies' has {len(strategies_raw)} entries but "
                    f"n_coas={n_coas} COAs were requested."
                )
            }), 400

    try:
        result_coas = generate_corps_coas(
            n_rollouts=n_rollouts,
            n_coas=n_coas,
            seed=seed,
            strategies=strategies_raw,
            env_kwargs=env_kwargs,
            explain=explain,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid request parameters: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"error": f"COA generation failed: {exc}"}), 500

    return jsonify({
        "coas": [c.as_dict() for c in result_coas],
        "n_coas": len(result_coas),
    }), 200


@app.route("/corps/coas/modify", methods=["POST"])
def corps_coas_modify() -> Any:
    """Modify a COA and re-simulate it (E9.2).

    Request body:

    .. code-block:: json

        {
            "coa": { "label": "full_advance", "rank": 1, "score": {...},
                     "action_summary": {}, "seed": 37 },
            "modification": {
                "strategy_override": "pincer_attack",
                "n_rollouts": 5,
                "division_command_overrides": {"0": 2}
            },
            "env_kwargs": {"n_divisions": 3}
        }
    """
    from envs.corps_env import CorpsEnv
    from analysis.coa_generator import (
        CorpsCOAGenerator, COAModification, CorpsCourseOfAction, CorpsCOAScore,
    )

    body_raw = request.get_json(force=True, silent=True)
    if not isinstance(body_raw, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    body: dict = body_raw

    coa_dict = body.get("coa")
    mod_dict  = body.get("modification", {})
    if not isinstance(coa_dict, dict):
        return jsonify({"error": "'coa' must be a JSON object."}), 400
    if not isinstance(mod_dict, dict):
        return jsonify({"error": "'modification' must be a JSON object."}), 400

    try:
        env_kwargs = _parse_corps_env_kwargs(body.get("env_kwargs", {}))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Reconstruct a CorpsCourseOfAction from the dict.
    try:
        score_d = coa_dict.get("score", {})
        coa_score = CorpsCOAScore(
            win_rate=float(score_d.get("win_rate", 0.0)),
            draw_rate=float(score_d.get("draw_rate", 0.0)),
            loss_rate=float(score_d.get("loss_rate", 0.0)),
            blue_casualties=float(score_d.get("blue_casualties", 0.0)),
            red_casualties=float(score_d.get("red_casualties", 0.0)),
            objective_completion=float(score_d.get("objective_completion", 0.0)),
            supply_efficiency=float(score_d.get("supply_efficiency", 0.0)),
            composite=float(score_d.get("composite", 0.0)),
            n_rollouts=int(score_d.get("n_rollouts", 1)),
        )
        coa = CorpsCourseOfAction(
            label=str(coa_dict.get("label", "")),
            rank=int(coa_dict.get("rank", 0)),
            score=coa_score,
            action_summary=coa_dict.get("action_summary", {}),
            seed=int(coa_dict.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid 'coa' structure: {exc}"}), 400

    # Build COAModification.
    div_overrides_raw = mod_dict.get("division_command_overrides", None)
    div_overrides: dict | None = None
    if div_overrides_raw is not None:
        try:
            div_overrides = {int(k): int(v) for k, v in div_overrides_raw.items()}
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid division_command_overrides: {exc}"}), 400

    n_rollouts_mod_raw = mod_dict.get("n_rollouts", None)
    try:
        n_rollouts_mod = int(n_rollouts_mod_raw) if n_rollouts_mod_raw is not None else None
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid modification n_rollouts: {exc}"}), 400

    modification = COAModification(
        strategy_override=mod_dict.get("strategy_override", None),
        n_rollouts=n_rollouts_mod,
        division_command_overrides=div_overrides,
    )

    try:
        env = CorpsEnv(**(env_kwargs or {}))
        generator = CorpsCOAGenerator(env=env, n_rollouts=10, n_coas=1)
        result = generator.modify_and_evaluate(coa, modification)
        env.close()
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid request parameters: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"error": f"COA modification failed: {exc}"}), 500

    return jsonify({"coa": result.as_dict()}), 200


@app.route("/corps/coas/explain", methods=["POST"])
def corps_coas_explain() -> Any:
    """Explain the key decisions driving a COA's outcome (E9.2).

    Request body:

    .. code-block:: json

        {
            "coa": { "label": "full_advance", "rank": 1, "score": {...},
                     "action_summary": {}, "seed": 37 },
            "n_rollouts": 10,
            "env_kwargs": {"n_divisions": 3}
        }
    """
    from envs.corps_env import CorpsEnv
    from analysis.coa_generator import (
        CorpsCOAGenerator, CorpsCourseOfAction, CorpsCOAScore,
    )

    body_raw = request.get_json(force=True, silent=True)
    if not isinstance(body_raw, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    body: dict = body_raw

    coa_dict = body.get("coa")
    if not isinstance(coa_dict, dict):
        return jsonify({"error": "'coa' must be a JSON object."}), 400

    try:
        env_kwargs = _parse_corps_env_kwargs(body.get("env_kwargs", {}))
        n_rollouts_raw = body.get("n_rollouts", 5)
        n_rollouts = _parse_int(n_rollouts_raw, "n_rollouts")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    # Reconstruct the CorpsCourseOfAction.
    try:
        score_d = coa_dict.get("score", {})
        coa_score = CorpsCOAScore(
            win_rate=float(score_d.get("win_rate", 0.0)),
            draw_rate=float(score_d.get("draw_rate", 0.0)),
            loss_rate=float(score_d.get("loss_rate", 0.0)),
            blue_casualties=float(score_d.get("blue_casualties", 0.0)),
            red_casualties=float(score_d.get("red_casualties", 0.0)),
            objective_completion=float(score_d.get("objective_completion", 0.0)),
            supply_efficiency=float(score_d.get("supply_efficiency", 0.0)),
            composite=float(score_d.get("composite", 0.0)),
            n_rollouts=int(score_d.get("n_rollouts", 1)),
        )
        coa = CorpsCourseOfAction(
            label=str(coa_dict.get("label", "")),
            rank=int(coa_dict.get("rank", 0)),
            score=coa_score,
            action_summary=coa_dict.get("action_summary", {}),
            seed=int(coa_dict.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid 'coa' structure: {exc}"}), 400

    try:
        env = CorpsEnv(**(env_kwargs or {}))
        # We need fresh rollouts for explain if none are cached.
        generator = CorpsCOAGenerator(
            env=env, n_rollouts=n_rollouts, n_coas=1,
            strategies=[coa.label], seed=coa.seed,
        )
        generator.generate()  # populate _last_rollout_results
        explanation = generator.explain_coa(coa)
        env.close()
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid request parameters: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"error": f"COA explanation failed: {exc}"}), 500

    return jsonify({"explanation": explanation.as_dict()}), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=False)
