"""Export trained policies to ONNX and TorchScript formats (Epic E5.5).

Supports three policy types shipped with this project:

* :class:`~models.mappo_policy.MAPPOActor` — actor-only export for deployment
  (most common inference use-case: given obs → action mean & std).
* :class:`~models.mappo_policy.MAPPOCritic` — centralized critic export.
* ``BattalionMlpPolicy`` (SB3 ActorCriticPolicy) — wraps the mlp_extractor
  + action_net in a thin :class:`_SB3ActorWrapper` for tracing.

Usage::

    # Export a MAPPOActor checkpoint to ONNX and TorchScript
    python scripts/export_policy.py \\
        --checkpoint path/to/actor.pt \\
        --model-type mappo_actor \\
        --obs-dim 22 \\
        --action-dim 3 \\
        --output-dir exports/

    # Export an SB3 PPO checkpoint
    python scripts/export_policy.py \\
        --checkpoint path/to/ppo.zip \\
        --model-type sb3_mlp \\
        --obs-dim 12 \\
        --action-dim 3 \\
        --output-dir exports/

    # ONNX only
    python scripts/export_policy.py \\
        --checkpoint actor.pt --model-type mappo_actor \\
        --obs-dim 22 --action-dim 3 \\
        --formats onnx

Output files
------------
``<output-dir>/<stem>.onnx``          — ONNX model (opset 17)
``<output-dir>/<stem>.torchscript.pt`` — TorchScript (torch.jit.trace)

Exit codes
----------
0 — all requested exports succeeded
1 — export failed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import models / training modules.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = [
    "export_to_onnx",
    "export_to_torchscript",
    "export_policy",
    "benchmark_inference",
]

# ---------------------------------------------------------------------------
# SB3 wrapper
# ---------------------------------------------------------------------------


class _SB3ActorWrapper(nn.Module):
    """Thin wrapper around an SB3 ActorCriticPolicy for tracing.

    Only the *actor* path (mlp_extractor → action_net → action mean) is
    exported.  This is the part used at inference time.

    Parameters
    ----------
    sb3_policy:
        The ``policy`` attribute of a ``stable_baselines3.PPO`` model, i.e.
        an instance of :class:`~stable_baselines3.common.policies.ActorCriticPolicy`.
    """

    def __init__(self, sb3_policy: nn.Module) -> None:
        super().__init__()
        self.mlp_extractor = sb3_policy.mlp_extractor  # type: ignore[attr-defined]
        self.action_net = sb3_policy.action_net  # type: ignore[attr-defined]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action-distribution *mean* for a batch of observations.

        Parameters
        ----------
        obs:
            Float32 tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        action_mean : torch.Tensor — shape ``(batch, action_dim)``
        """
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def _load_model(
    checkpoint: Path,
    model_type: str,
    obs_dim: int,
    action_dim: int,
    state_dim: int,
) -> nn.Module:
    """Load a model from *checkpoint* according to *model_type*.

    Parameters
    ----------
    checkpoint:
        Path to a ``.pt`` / ``.pth`` state-dict file **or** an SB3 ``.zip``
        file.
    model_type:
        One of ``"mappo_actor"``, ``"mappo_critic"``, ``"sb3_mlp"``.
    obs_dim:
        Observation dimensionality (used to construct fresh model before
        loading state dict).
    action_dim:
        Action dimensionality (only relevant for actor models).
    state_dim:
        Global state dimensionality (only relevant for critic models).

    Returns
    -------
    nn.Module
        The loaded model in ``eval()`` mode.

    Raises
    ------
    ValueError
        For unknown *model_type*.
    RuntimeError
        If loading fails.
    """
    if model_type == "mappo_actor":
        from models.mappo_policy import MAPPOActor

        model: nn.Module = MAPPOActor(obs_dim=obs_dim, action_dim=action_dim)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    elif model_type == "mappo_critic":
        from models.mappo_policy import MAPPOCritic

        model = MAPPOCritic(state_dim=state_dim)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    elif model_type == "sb3_mlp":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for sb3_mlp export.  "
                "Install it with: pip install stable-baselines3"
            ) from exc

        ppo = PPO.load(str(checkpoint), device="cpu")
        model = _SB3ActorWrapper(ppo.policy)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'.  "
            "Valid choices: mappo_actor, mappo_critic, sb3_mlp."
        )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    *,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_axes: dict | None = None,
) -> Path:
    """Export *model* to ONNX at *output_path*.

    Parameters
    ----------
    model:
        ``nn.Module`` in eval mode.
    dummy_input:
        Representative input tensor used for tracing.
    output_path:
        Destination ``.onnx`` file.
    opset_version:
        ONNX opset.  Defaults to 17.
    input_names:
        Names for the ONNX input nodes.
    output_names:
        Names for the ONNX output nodes.
    dynamic_axes:
        Dynamic-axes dict forwarded to :func:`torch.onnx.export`.
        Defaults to making the batch dimension dynamic.

    Returns
    -------
    Path
        *output_path* (for chaining / logging).

    Raises
    ------
    ImportError
        If the ``onnx`` package is not installed.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required for ONNX export.  "
            "Install it with: pip install onnx"
        ) from exc

    if input_names is None:
        input_names = ["obs"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
        )

    # Validate the produced model graph.
    import onnx as _onnx

    _onnx.checker.check_model(str(output_path))

    return output_path


def export_to_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> Path:
    """Trace *model* with TorchScript and save to *output_path*.

    Parameters
    ----------
    model:
        ``nn.Module`` in eval mode.
    dummy_input:
        Representative input tensor for tracing.
    output_path:
        Destination ``.pt`` file.

    Returns
    -------
    Path
        *output_path* (for chaining / logging).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        scripted = torch.jit.trace(model, dummy_input, strict=True)

    torch.jit.save(scripted, str(output_path))
    return output_path


def benchmark_inference(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark PyTorch CPU inference latency for *model*.

    Parameters
    ----------
    model:
        ``nn.Module`` in eval mode.
    dummy_input:
        Representative input tensor (will be moved to CPU).
    n_warmup:
        Number of warm-up forward passes (not timed).
    n_runs:
        Number of timed forward passes.

    Returns
    -------
    dict
        ``mean_ms``, ``min_ms``, ``max_ms`` latency statistics.
    """
    dummy_input = dummy_input.cpu()
    model = model.cpu()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_input)
            times.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_onnx_inference(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    *,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark ONNX Runtime CPU inference latency.

    Parameters
    ----------
    onnx_path:
        Path to the exported ``.onnx`` file.
    dummy_input:
        Representative input tensor.
    n_warmup:
        Number of warm-up passes.
    n_runs:
        Number of timed passes.

    Returns
    -------
    dict
        ``mean_ms``, ``min_ms``, ``max_ms`` latency statistics.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "The 'onnxruntime' package is required for ONNX benchmarking.  "
            "Install it with: pip install onnxruntime"
        ) from exc

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    np_input = dummy_input.numpy()

    for _ in range(n_warmup):
        sess.run(None, {input_name: np_input})

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: np_input})
        times.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


# ---------------------------------------------------------------------------
# High-level export orchestrator
# ---------------------------------------------------------------------------


def export_policy(
    checkpoint: Path | str,
    model_type: str,
    output_dir: Path | str,
    *,
    obs_dim: int,
    action_dim: int = 3,
    state_dim: int = 25,
    formats: Sequence[str] = ("onnx", "torchscript"),
    batch_size: int = 1,
    run_benchmark: bool = False,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export a trained policy to the requested portable formats.

    This is the primary programmatic entry-point.  The CLI wraps this
    function.

    Parameters
    ----------
    checkpoint:
        Path to the source checkpoint (``.pt`` state-dict or SB3 ``.zip``).
    model_type:
        ``"mappo_actor"``, ``"mappo_critic"``, or ``"sb3_mlp"``.
    output_dir:
        Directory where exported files will be written.
    obs_dim:
        Observation dimensionality (input size).
    action_dim:
        Action dimensionality (used for actor models).
    state_dim:
        Global state dimensionality (used for critic models).
    formats:
        Subset of ``{"onnx", "torchscript"}`` to export.
    batch_size:
        Dummy-input batch size used for tracing.
    run_benchmark:
        When ``True``, print latency statistics for PyTorch and ONNX.
    opset_version:
        ONNX opset version.

    Returns
    -------
    dict[str, Path]
        Mapping from format name to the written file path.

    Raises
    ------
    ValueError
        For unrecognised *model_type* or *formats* entries.
    """
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)

    valid_formats = {"onnx", "torchscript"}
    unknown = set(formats) - valid_formats
    if unknown:
        raise ValueError(
            f"Unknown export formats: {sorted(unknown)}.  "
            f"Valid: {sorted(valid_formats)}"
        )

    # ── Build dummy input ────────────────────────────────────────────────────
    if model_type == "mappo_critic":
        input_dim = state_dim
    else:
        input_dim = obs_dim

    dummy_input = torch.zeros(batch_size, input_dim, dtype=torch.float32)

    # ── Load model ───────────────────────────────────────────────────────────
    model = _load_model(checkpoint, model_type, obs_dim, action_dim, state_dim)

    # ── Determine output file stem ───────────────────────────────────────────
    stem = checkpoint.stem

    exported: dict[str, Path] = {}

    if "torchscript" in formats:
        ts_path = output_dir / f"{stem}.torchscript.pt"
        export_to_torchscript(model, dummy_input, ts_path)
        print(f"[export] TorchScript → {ts_path}")
        exported["torchscript"] = ts_path

    if "onnx" in formats:
        onnx_path = output_dir / f"{stem}.onnx"
        # Critic takes a different input; label accordingly.
        input_name = "state" if model_type == "mappo_critic" else "obs"
        output_name = "value" if model_type == "mappo_critic" else "output"
        export_to_onnx(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset_version,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        )
        print(f"[export] ONNX       → {onnx_path}")
        exported["onnx"] = onnx_path

    # ── Optional benchmark ───────────────────────────────────────────────────
    if run_benchmark:
        print("\n[benchmark] PyTorch CPU inference:")
        pt_stats = benchmark_inference(model, dummy_input)
        print(
            f"  mean={pt_stats['mean_ms']:.3f} ms  "
            f"min={pt_stats['min_ms']:.3f} ms  "
            f"max={pt_stats['max_ms']:.3f} ms"
        )

        if "onnx" in exported:
            try:
                print("[benchmark] ONNX Runtime CPU inference:")
                ort_stats = benchmark_onnx_inference(exported["onnx"], dummy_input)
                print(
                    f"  mean={ort_stats['mean_ms']:.3f} ms  "
                    f"min={ort_stats['min_ms']:.3f} ms  "
                    f"max={ort_stats['max_ms']:.3f} ms"
                )
                speedup = pt_stats["mean_ms"] / max(ort_stats["mean_ms"], 1e-9)
                print(f"  ONNX speedup vs PyTorch: {speedup:.2f}x")
            except ImportError as exc:
                print(f"  [skip] {exc}")

        if "torchscript" in exported:
            print("[benchmark] TorchScript CPU inference:")
            ts_model = torch.jit.load(str(exported["torchscript"]))
            ts_model.eval()
            ts_stats = benchmark_inference(ts_model, dummy_input)
            print(
                f"  mean={ts_stats['mean_ms']:.3f} ms  "
                f"min={ts_stats['min_ms']:.3f} ms  "
                f"max={ts_stats['max_ms']:.3f} ms"
            )

    return exported


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a trained policy to ONNX and/or TorchScript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to source checkpoint (.pt state-dict or SB3 .zip).",
    )
    p.add_argument(
        "--model-type",
        required=True,
        choices=["mappo_actor", "mappo_critic", "sb3_mlp"],
        help="Architecture type to load.",
    )
    p.add_argument(
        "--obs-dim",
        required=True,
        type=int,
        help="Observation dimensionality (actor input size).",
    )
    p.add_argument(
        "--action-dim",
        type=int,
        default=3,
        help="Action dimensionality (actor output size).",
    )
    p.add_argument(
        "--state-dim",
        type=int,
        default=25,
        help="Global state dimensionality (critic input size).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory for exported files.",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["onnx", "torchscript"],
        default=["onnx", "torchscript"],
        help="Export format(s).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy-input batch size used for tracing.",
    )
    p.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Print inference latency statistics after export.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        exported = export_policy(
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            output_dir=args.output_dir,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            formats=args.formats,
            batch_size=args.batch_size,
            opset_version=args.opset_version,
            run_benchmark=args.benchmark,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"\n[done] {len(exported)} file(s) written:")
    for fmt, path in exported.items():
        print(f"  {fmt}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
