# Main training entry point
"""Train a PPO agent on :class:`~envs.battalion_env.BattalionEnv`.

Loads hyperparameters from ``configs/default.yaml`` (overridden by an
optional ``--config`` file), initialises a W&B run, trains a
:class:`~models.mlp_policy.BattalionMlpPolicy` agent with SB3 PPO, and
saves checkpoints.

Quick start::

    python training/train.py
    python training/train.py --config configs/experiment_1.yaml
    python training/train.py --total-timesteps 50000 --seed 7
    python training/train.py --no-wandb   # disable W&B (offline / CI)

Acceptance criteria (E1.4):
* Runs to completion without errors on CPU.
* W&B run appears with reward curves and episode-length metrics.
* Checkpoint saved to ``checkpoints/<run_id>/final.zip``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path bootstrap (allow running from repo root without installing the package)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from envs.battalion_env import BattalionEnv  # noqa: E402
from models.mlp_policy import BattalionMlpPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(default_path: Path, override_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``default.yaml`` and optionally merge an experiment override file."""
    cfg = _load_yaml(default_path)
    if override_path is not None:
        cfg = _deep_merge(cfg, _load_yaml(override_path))
    return cfg


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# SB3 callbacks
# ---------------------------------------------------------------------------


def _make_eval_callback(eval_env: BattalionEnv, cfg: Dict) -> Any:
    """Build a SB3 EvalCallback with checkpoint saving."""
    from stable_baselines3.common.callbacks import EvalCallback

    run_id = _get_run_id()
    checkpoint_dir = Path(cfg["eval"]["checkpoint_dir"]) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=max(1, cfg["eval"]["eval_freq"]),
        n_eval_episodes=cfg["eval"]["n_eval_episodes"],
        deterministic=True,
        render=False,
    )


def _make_checkpoint_callback(cfg: Dict) -> Any:
    """Build a SB3 CheckpointCallback for periodic saves."""
    from stable_baselines3.common.callbacks import CheckpointCallback

    run_id = _get_run_id()
    checkpoint_dir = Path(cfg["eval"]["checkpoint_dir"]) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return CheckpointCallback(
        save_freq=max(1, cfg["eval"]["checkpoint_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_battalion",
        verbose=1,
    )


# Module-level run_id so that callbacks and main() use the same value.
_RUN_ID: Optional[str] = None


def _get_run_id() -> str:
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = f"run_{int(time.time())}"
    return _RUN_ID


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(cfg: Dict, use_wandb: bool = True) -> Path:
    """Run the full PPO training loop.

    Parameters
    ----------
    cfg:
        Merged configuration dictionary (see :func:`load_config`).
    use_wandb:
        If ``False`` training proceeds without W&B (useful for CI/offline).

    Returns
    -------
    Path
        Path to the saved final model checkpoint.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    training_cfg = cfg.get("training", {})
    env_cfg = cfg.get("env", {})
    wandb_cfg = cfg.get("wandb", {})

    seed: int = training_cfg.get("seed", 42)
    total_timesteps: int = training_cfg.get("total_timesteps", 1_000_000)

    # ------------------------------------------------------------------
    # W&B initialisation
    # ------------------------------------------------------------------
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "wargames_training"),
                entity=wandb_cfg.get("entity") or None,
                tags=wandb_cfg.get("tags", ["v1", "ppo"]),
                config={
                    **cfg,
                    "git_sha": _git_sha(),
                },
                settings=wandb.Settings(start_method="thread"),
            )
            _RUN_ID_setter(wandb_run.id)
            print(f"[train] W&B run: {wandb_run.get_url()}")
        except Exception as exc:  # W&B not configured / no network
            print(f"[train] W&B init skipped: {exc}")
            use_wandb = False

    run_id = _get_run_id()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    def _make_env(seed_offset: int = 0):
        def _init():
            env = BattalionEnv(
                map_width=env_cfg.get("map_width", 1000.0),
                map_height=env_cfg.get("map_height", 1000.0),
                max_steps=env_cfg.get("max_steps", 500),
            )
            env.reset(seed=seed + seed_offset)
            return env
        return _init

    train_env = DummyVecEnv([_make_env(i) for i in range(1)])
    eval_env = BattalionEnv(
        map_width=env_cfg.get("map_width", 1000.0),
        map_height=env_cfg.get("map_height", 1000.0),
        max_steps=env_cfg.get("max_steps", 500),
    )

    # ------------------------------------------------------------------
    # W&B callback (optional)
    # ------------------------------------------------------------------
    callbacks = []
    if use_wandb and wandb_run is not None:
        try:
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(
                WandbCallback(
                    gradient_save_freq=0,
                    log="parameters",
                    verbose=0,
                )
            )
        except ImportError:
            print("[train] wandb.integration.sb3 not available; skipping WandbCallback")

    callbacks.append(_make_eval_callback(eval_env, cfg))
    callbacks.append(_make_checkpoint_callback(cfg))

    # ------------------------------------------------------------------
    # PPO model
    # ------------------------------------------------------------------
    # TensorBoard logging is optional; skip if package is not installed.
    try:
        import tensorboard  # noqa: F401
        tb_log = str(Path("logs") / run_id)
    except ImportError:
        tb_log = None

    model = PPO(
        BattalionMlpPolicy,
        train_env,
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        n_steps=training_cfg.get("n_steps", 2048),
        batch_size=training_cfg.get("batch_size", 64),
        n_epochs=training_cfg.get("n_epochs", 10),
        gamma=training_cfg.get("gamma", 0.99),
        gae_lambda=training_cfg.get("gae_lambda", 0.95),
        clip_range=training_cfg.get("clip_range", 0.2),
        ent_coef=training_cfg.get("ent_coef", 0.01),
        vf_coef=training_cfg.get("vf_coef", 0.5),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.5),
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"[train] Starting PPO training for {total_timesteps} timesteps (run_id={run_id})")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=True,
    )

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    checkpoint_dir = Path(cfg["eval"]["checkpoint_dir"]) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "final"
    model.save(str(final_path))
    print(f"[train] Final model saved → {final_path}.zip")

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    train_env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()

    return final_path


def _RUN_ID_setter(new_id: str) -> None:
    global _RUN_ID
    _RUN_ID = new_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on BattalionEnv"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML experiment override file (merged over default.yaml).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (useful for CI or offline runs).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    default_cfg_path = _PROJECT_ROOT / "configs" / "default.yaml"
    cfg = load_config(default_cfg_path, args.config)

    # CLI overrides
    if args.total_timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    use_wandb = not args.no_wandb
    train(cfg, use_wandb=use_wandb)


if __name__ == "__main__":
    main()

