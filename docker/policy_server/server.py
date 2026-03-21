"""Policy inference server for exported policies (Epic E5.5).

Serves ONNX and TorchScript policies via a lightweight Flask REST API.
Designed to run inside the ``policy_server`` Docker container but can
also be launched locally for development.

Endpoints
---------
``GET  /health``
    Liveness probe.  Returns ``{"status": "ok"}``.

``GET  /info``
    Returns metadata about the loaded model
    (format, input/output shapes, model path).

``POST /predict``
    Run a single forward pass.

    **Request** (JSON)::

        {"obs": [[0.1, 0.2, ..., 0.22]]}   # shape (batch, obs_dim)

    **Response** (JSON)::

        {"output": [[0.03, -0.12, 0.55]]}  # shape (batch, out_dim)

Environment variables
---------------------
``POLICY_PATH``     Path to the exported model file (required).
``POLICY_FORMAT``   ``"onnx"`` (default) or ``"torchscript"``.
``HOST``            Bind address (default ``"0.0.0.0"``).
``PORT``            Bind port (default ``"8080"``).

Running locally::

    POLICY_PATH=exports/actor.onnx python docker/policy_server/server.py

Running via Docker::

    docker run -e POLICY_PATH=/models/actor.onnx \\
               -v $(pwd)/exports:/models \\
               -p 8080:8080 \\
               policy_server
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Flask is required.  Install it with: pip install flask"
    ) from exc

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Global model state — populated in _load_model_on_startup().
_model: Any = None
_policy_format: str = "onnx"
_policy_path: str = ""
_input_name: str = "obs"
_input_shape: list[int] = []
_output_shape: list[int] = []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_onnx(path: str) -> Any:
    """Load an ONNX model via ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX inference.  "
            "Install it with: pip install onnxruntime"
        ) from exc

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_opts=sess_opts, providers=["CPUExecutionProvider"])
    return sess


def _load_torchscript(path: str) -> Any:
    """Load a TorchScript model."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for TorchScript inference.  "
            "Install it with: pip install torch"
        ) from exc

    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def _load_model_on_startup() -> None:
    """Called once at startup to load the policy into memory."""
    global _model, _policy_format, _policy_path, _input_name, _input_shape, _output_shape

    _policy_path = os.environ.get("POLICY_PATH", "")
    if not _policy_path:
        raise RuntimeError(
            "POLICY_PATH environment variable is not set.  "
            "Set it to the path of an exported .onnx or .torchscript.pt file."
        )

    if not Path(_policy_path).exists():
        raise FileNotFoundError(f"Policy file not found: {_policy_path!r}")

    _policy_format = os.environ.get("POLICY_FORMAT", "").lower()
    if not _policy_format:
        # Auto-detect from file extension.
        if _policy_path.endswith(".onnx"):
            _policy_format = "onnx"
        elif _policy_path.endswith(".pt") or _policy_path.endswith(".pth"):
            _policy_format = "torchscript"
        else:
            _policy_format = "onnx"

    if _policy_format == "onnx":
        _model = _load_onnx(_policy_path)
        inp = _model.get_inputs()[0]
        _input_name = inp.name
        _input_shape = [d if isinstance(d, int) else -1 for d in inp.shape]
        out = _model.get_outputs()[0]
        _output_shape = [d if isinstance(d, int) else -1 for d in out.shape]
    elif _policy_format == "torchscript":
        _model = _load_torchscript(_policy_path)
        _input_name = "obs"
        _input_shape = [-1, -1]
        _output_shape = [-1, -1]
    else:
        raise ValueError(
            f"Unknown POLICY_FORMAT={_policy_format!r}.  Use 'onnx' or 'torchscript'."
        )

    print(
        f"[server] Loaded {_policy_format} policy from {_policy_path!r}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/info", methods=["GET"])
def info() -> Any:
    """Return metadata about the loaded model."""
    return jsonify(
        {
            "policy_path": _policy_path,
            "policy_format": _policy_format,
            "input_name": _input_name,
            "input_shape": _input_shape,
            "output_shape": _output_shape,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """Run a forward pass and return model output.

    Request body (JSON)::

        {"obs": [[float, ...], ...]}   # shape (batch, obs_dim)

    Response (JSON)::

        {"output": [[float, ...], ...], "latency_ms": float}
    """
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    obs_raw = body.get("obs")
    if obs_raw is None:
        return jsonify({"error": "Missing required field 'obs'."}), 400

    try:
        if _policy_format == "onnx":
            import numpy as np

            obs_np = np.array(obs_raw, dtype=np.float32)
            if obs_np.ndim == 1:
                obs_np = obs_np[None, :]  # add batch dim
            t0 = time.perf_counter()
            result = _model.run(None, {_input_name: obs_np})
            latency_ms = (time.perf_counter() - t0) * 1e3
            output = result[0].tolist()

        elif _policy_format == "torchscript":
            import torch

            obs_t = torch.tensor(obs_raw, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                out_t = _model(obs_t)
            latency_ms = (time.perf_counter() - t0) * 1e3
            output = out_t.tolist()

        else:
            return jsonify({"error": f"Unsupported format: {_policy_format!r}"}), 500

    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"error": f"Inference error: {exc}"}), 500

    return jsonify({"output": output, "latency_ms": round(latency_ms, 4)}), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_model_on_startup()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host=host, port=port, debug=False)
