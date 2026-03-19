"""Checkpoint naming and manifest helpers for training artifacts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional


_STEP_PATTERN = re.compile(r"_(\d+)_steps\.zip$")


def checkpoint_name_prefix(*, seed: int, curriculum_level: int, enable_v2: bool) -> str:
    """Return periodic checkpoint prefix for the active naming mode."""
    if not enable_v2:
        return "ppo_battalion"
    return f"ppo_battalion_s{int(seed)}_c{int(curriculum_level)}"


def checkpoint_final_stem(*, seed: int, curriculum_level: int, enable_v2: bool) -> str:
    """Return final checkpoint stem (without .zip) for the active naming mode."""
    if not enable_v2:
        return "ppo_battalion_final"
    return f"ppo_battalion_s{int(seed)}_c{int(curriculum_level)}_final"


def checkpoint_best_filename(*, seed: int, curriculum_level: int, enable_v2: bool) -> str:
    """Return best checkpoint filename for the active naming mode."""
    if not enable_v2:
        return "best_model.zip"
    return f"ppo_battalion_s{int(seed)}_c{int(curriculum_level)}_best.zip"


def parse_step_from_checkpoint_name(path: Path) -> Optional[int]:
    """Extract timesteps from a periodic checkpoint file name."""
    match = _STEP_PATTERN.search(path.name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class CheckpointManifest:
    """Append-only JSONL checkpoint manifest for local artifact indexing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read_rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def known_paths(self) -> set[str]:
        """Return all path strings already present in the manifest."""
        return {str(row.get("path", "")) for row in self._read_rows()}

    def append(self, row: dict) -> None:
        """Append a single JSON object row to manifest storage."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def register(
        self,
        artifact_path: Path,
        *,
        artifact_type: str,
        seed: int,
        curriculum_level: int,
        run_id: Optional[str],
        config_hash: str,
        step: Optional[int],
    ) -> bool:
        """Register one artifact path if it is not already indexed."""
        path_value = str(artifact_path)
        if path_value in self.known_paths():
            return False
        self.append(
            {
                "timestamp": int(time.time()),
                "type": str(artifact_type),
                "path": path_value,
                "seed": int(seed),
                "curriculum_level": int(curriculum_level),
                "run_id": run_id or None,
                "config_hash": str(config_hash),
                "step": int(step) if step is not None else None,
            }
        )
        return True

    def latest_periodic(self, checkpoint_dir: Path, prefix: str) -> Optional[Path]:
        """Return latest periodic checkpoint from manifest, if available."""
        rows = self._read_rows()
        best_step = -1
        best_path: Optional[Path] = None
        for row in rows:
            if row.get("type") != "periodic":
                continue
            path_value = str(row.get("path", ""))
            if not path_value:
                continue
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = checkpoint_dir / candidate
            if not candidate.name.startswith(prefix + "_"):
                continue
            step_value = row.get("step")
            if not isinstance(step_value, int):
                continue
            if not candidate.exists():
                continue
            if step_value > best_step:
                best_step = step_value
                best_path = candidate
        return best_path
