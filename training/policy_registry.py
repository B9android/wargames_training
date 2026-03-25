# SPDX-License-Identifier: MIT
# training/policy_registry.py
"""Versioned policy registry for the multi-echelon HRL stack (E3.6).

Stores and loads policies by echelon (battalion / brigade / division) and a
caller-assigned version string.  The registry is backed by a JSON manifest so
that it survives across training runs and evaluation sessions.

Supported echelons
------------------
* ``battalion`` — MAPPO ``.pt`` checkpoint
  (loaded via :func:`training.utils.freeze_policy.load_and_freeze_mappo`)
* ``brigade`` — SB3 PPO ``.zip`` checkpoint
  (loaded via :func:`training.utils.freeze_policy.load_and_freeze_sb3`)
* ``division`` — SB3 PPO ``.zip`` checkpoint
  (loaded via :func:`training.utils.freeze_policy.load_and_freeze_sb3`)

Usage::

    from training.policy_registry import PolicyRegistry, Echelon

    reg = PolicyRegistry("checkpoints/policy_registry.json")

    # Register a checkpoint
    reg.register(
        echelon=Echelon.BATTALION,
        version="v2_final",
        path="checkpoints/mappo_2v2/mappo_policy_final.pt",
        run_id="wandb-run-abc123",
    )
    reg.save()

    # List all registered policies
    for entry in reg.list():
        print(entry)

    # Load a frozen policy back into memory
    policy = reg.load(Echelon.BATTALION, version="v2_final")

CLI::

    # List all registered policies
    python -m training.policy_registry list

    # Register a new entry
    python -m training.policy_registry register \\
        --echelon battalion \\
        --version v2_final \\
        --path checkpoints/mappo_2v2/mappo_policy_final.pt \\
        --run-id wandb-run-abc123
"""

from __future__ import annotations

import argparse
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Union

log = logging.getLogger(__name__)

__all__ = [
    "Echelon",
    "PolicyEntry",
    "PolicyRegistry",
]

# ---------------------------------------------------------------------------
# Echelon enum
# ---------------------------------------------------------------------------


class Echelon(str, Enum):
    """Supported HRL echelon levels."""

    BATTALION = "battalion"
    BRIGADE = "brigade"
    DIVISION = "division"

    @classmethod
    def from_str(cls, value: str) -> "Echelon":
        """Case-insensitive lookup from string.

        Parameters
        ----------
        value:
            Echelon name, e.g. ``"battalion"``, ``"Brigade"``, or an
            :class:`Echelon` enum value.

        Returns
        -------
        Echelon

        Raises
        ------
        ValueError
            If *value* is not a valid echelon name.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(e.value for e in cls)
            raise ValueError(
                f"Invalid echelon '{value}'. Must be one of: {valid}."
            )



# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _echelon_str(echelon: "Union[Echelon, str]") -> str:
    """Normalise *echelon* to its lowercase string value."""
    if isinstance(echelon, Echelon):
        return echelon.value
    return Echelon.from_str(str(echelon)).value


# ---------------------------------------------------------------------------
# PolicyEntry
# ---------------------------------------------------------------------------


class PolicyEntry(NamedTuple):
    """Metadata record for a single registered policy checkpoint.

    Attributes
    ----------
    echelon:
        The HRL echelon this checkpoint belongs to.
    version:
        Caller-assigned version string, e.g. ``"v2_final"`` or ``"step_500k"``.
    path:
        File-system path to the checkpoint file.
    run_id:
        Optional W&B run ID associated with this checkpoint.
    """

    echelon: str
    version: str
    path: str
    run_id: Optional[str]

    def __str__(self) -> str:
        run = self.run_id or "—"
        return (
            f"{self.echelon:<10s}  {self.version:<20s}  "
            f"{self.path:<50s}  run_id={run}"
        )


# ---------------------------------------------------------------------------
# PolicyRegistry
# ---------------------------------------------------------------------------


class PolicyRegistry:
    """Versioned policy registry backed by a JSON manifest.

    Parameters
    ----------
    path:
        Path to the JSON manifest file.  The parent directory is created on
        :meth:`save`.  Pass ``None`` for an in-memory-only registry.
    """

    def __init__(
        self,
        path: Union[str, Path, None] = "checkpoints/policy_registry.json",
    ) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._entries: list[PolicyEntry] = []
        if self._path is not None and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def can_save(self) -> bool:
        """``True`` when the registry has a backing file path."""
        return self._path is not None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        echelon: Union[Echelon, str],
        version: str,
        path: Union[str, Path],
        run_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> PolicyEntry:
        """Register a new policy checkpoint.

        Parameters
        ----------
        echelon:
            The HRL echelon: ``"battalion"``, ``"brigade"``, or
            ``"division"`` (or the :class:`Echelon` enum value).
        version:
            Caller-assigned version string, e.g. ``"v2_final"``.
        path:
            File-system path to the checkpoint.
        run_id:
            Optional W&B run ID linked to this checkpoint.
        overwrite:
            When ``True`` an existing entry with the same echelon+version
            is replaced.  When ``False`` (default) a :exc:`ValueError` is
            raised if the entry already exists.

        Returns
        -------
        PolicyEntry
            The newly created entry.

        Raises
        ------
        ValueError
            If an entry with the same echelon+version already exists and
            *overwrite* is ``False``, or if *echelon* is invalid.
        """
        echelon_str = _echelon_str(echelon)
        path_str = str(path)

        existing_idx = self._find_index(echelon_str, version)
        if existing_idx is not None:
            if not overwrite:
                raise ValueError(
                    f"Entry already exists for echelon='{echelon_str}' "
                    f"version='{version}'. Pass overwrite=True to replace it."
                )
            self._entries.pop(existing_idx)

        entry = PolicyEntry(
            echelon=echelon_str,
            version=version,
            path=path_str,
            run_id=run_id,
        )
        self._entries.append(entry)
        log.info(
            "PolicyRegistry: registered %s/%s → %s (run_id=%s)",
            echelon_str,
            version,
            path_str,
            run_id,
        )
        return entry

    def get(self, echelon: Union[Echelon, str], version: str) -> PolicyEntry:
        """Return the :class:`PolicyEntry` for *echelon* / *version*.

        Parameters
        ----------
        echelon:
            Echelon name or :class:`Echelon` enum.
        version:
            Version string.

        Returns
        -------
        PolicyEntry

        Raises
        ------
        KeyError
            If no matching entry is found.
        """
        echelon_str = _echelon_str(echelon)
        idx = self._find_index(echelon_str, version)
        if idx is None:
            raise KeyError(
                f"No policy registered for echelon='{echelon_str}' "
                f"version='{version}'."
            )
        return self._entries[idx]

    def remove(self, echelon: Union[Echelon, str], version: str) -> None:
        """Remove an entry from the registry.

        Parameters
        ----------
        echelon:
            Echelon name or :class:`Echelon` enum.
        version:
            Version string.

        Raises
        ------
        KeyError
            If no matching entry is found.
        """
        echelon_str = _echelon_str(echelon)
        idx = self._find_index(echelon_str, version)
        if idx is None:
            raise KeyError(
                f"No policy registered for echelon='{echelon_str}' "
                f"version='{version}'."
            )
        self._entries.pop(idx)
        log.info(
            "PolicyRegistry: removed %s/%s", echelon_str, version
        )

    def list(
        self, echelon: Union[Echelon, str, None] = None
    ) -> List[PolicyEntry]:
        """Return all registered entries, optionally filtered by echelon.

        Parameters
        ----------
        echelon:
            When provided, only entries for this echelon are returned.

        Returns
        -------
        list[PolicyEntry]
            A fresh list; mutating it does not affect the registry.
        """
        if echelon is None:
            return list(self._entries)
        echelon_str = _echelon_str(echelon)
        return [e for e in self._entries if e.echelon == echelon_str]

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load(
        self,
        echelon: Union[Echelon, str],
        version: str,
        device: str = "cpu",
        **mappo_kwargs: Any,
    ) -> Any:
        """Load and return a frozen policy for the given echelon / version.

        The checkpoint format is inferred from the echelon:

        * ``battalion`` → MAPPO ``.pt`` checkpoint loaded via
          :func:`~training.utils.freeze_policy.load_and_freeze_mappo`.
          *mappo_kwargs* (``obs_dim``, ``action_dim``, ``state_dim``,
          ``n_agents``) are forwarded to that function and are **required**
          for battalion policies.
        * ``brigade`` / ``division`` → SB3 PPO ``.zip`` checkpoint loaded
          via :func:`~training.utils.freeze_policy.load_and_freeze_sb3`.

        Parameters
        ----------
        echelon:
            Echelon name or :class:`Echelon` enum.
        version:
            Version string.
        device:
            PyTorch device string (default ``"cpu"``).
        **mappo_kwargs:
            Extra keyword arguments forwarded to
            :func:`~training.utils.freeze_policy.load_and_freeze_mappo` when
            *echelon* is ``"battalion"``.  Must include ``obs_dim``,
            ``action_dim``, ``state_dim``, and optionally ``n_agents``.

        Returns
        -------
        Any
            A frozen :class:`~models.mappo_policy.MAPPOPolicy` (battalion)
            or a frozen SB3 ``PPO`` model (brigade / division).

        Raises
        ------
        KeyError
            If no entry is registered for this echelon+version.
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        entry = self.get(echelon, version)
        echelon_enum = Echelon.from_str(entry.echelon)

        checkpoint_path = Path(entry.path)

        if echelon_enum == Echelon.BATTALION:
            required = ("obs_dim", "action_dim", "state_dim")
            missing = [k for k in required if k not in mappo_kwargs]
            if missing:
                raise ValueError(
                    f"Loading a battalion policy requires keyword arguments: "
                    f"{', '.join(missing)}. "
                    "Pass them as keyword arguments to load()."
                )

        from training.utils.freeze_policy import (
            load_and_freeze_mappo,
            load_and_freeze_sb3,
        )

        if echelon_enum == Echelon.BATTALION:
            return load_and_freeze_mappo(
                checkpoint_path=checkpoint_path,
                device=device,
                **mappo_kwargs,
            )

        # brigade or division → SB3 PPO
        return load_and_freeze_sb3(checkpoint_path=checkpoint_path, device=device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the registry to its JSON manifest file.

        Raises
        ------
        ValueError
            If the registry was created without a file path.
        """
        if self._path is None:
            raise ValueError(
                "Cannot save: this PolicyRegistry was created without a "
                "file path.  Pass a path to the constructor."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "entries": [
                {
                    "echelon": e.echelon,
                    "version": e.version,
                    "path": e.path,
                    "run_id": e.run_id,
                }
                for e in self._entries
            ]
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        log.info("PolicyRegistry: saved %d entries to %s", len(self._entries), self._path)

    def _load(self) -> None:
        """Load entries from the existing JSON manifest."""
        assert self._path is not None
        with open(self._path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"PolicyRegistry: failed to parse JSON from "
                    f"'{self._path}': {exc}"
                ) from exc
        raw_entries = data.get("entries", [])
        for item in raw_entries:
            try:
                entry = PolicyEntry(
                    echelon=str(item["echelon"]),
                    version=str(item["version"]),
                    path=str(item["path"]),
                    run_id=item.get("run_id"),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"PolicyRegistry: malformed entry in '{self._path}': "
                    f"{item!r} — {exc}"
                ) from exc
            self._entries.append(entry)
        log.info(
            "PolicyRegistry: loaded %d entries from %s",
            len(self._entries),
            self._path,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PolicyRegistry(path={self._path!r}, "
            f"n_entries={len(self._entries)})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_index(self, echelon: str, version: str) -> Optional[int]:
        """Return the index of the matching entry or ``None``."""
        for i, e in enumerate(self._entries):
            if e.echelon == echelon and e.version == version:
                return i
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.policy_registry",
        description="Manage the versioned multi-echelon policy registry.",
    )
    parser.add_argument(
        "--registry",
        default="checkpoints/policy_registry.json",
        help="Path to the JSON registry manifest (default: checkpoints/policy_registry.json).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── list ──────────────────────────────────────────────────────────
    list_p = sub.add_parser("list", help="List all registered policies.")
    list_p.add_argument(
        "--echelon",
        default=None,
        choices=[e.value for e in Echelon],
        help="Filter by echelon.",
    )

    # ── register ──────────────────────────────────────────────────────
    reg_p = sub.add_parser("register", help="Register a new policy checkpoint.")
    reg_p.add_argument(
        "--echelon",
        required=True,
        choices=[e.value for e in Echelon],
        help="Echelon: battalion, brigade, or division.",
    )
    reg_p.add_argument("--version", required=True, help="Version string, e.g. v2_final.")
    reg_p.add_argument("--path", required=True, help="Path to the checkpoint file.")
    reg_p.add_argument("--run-id", default=None, help="W&B run ID (optional).")
    reg_p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing entry with the same echelon+version.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the policy registry."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    registry = PolicyRegistry(path=args.registry)

    if args.command == "list":
        entries = registry.list(echelon=args.echelon)
        if not entries:
            echelon_hint = f" for echelon='{args.echelon}'" if args.echelon else ""
            print(f"No policies registered{echelon_hint}.")
            return
        header = f"{'Echelon':<10s}  {'Version':<20s}  {'Path':<50s}  W&B Run ID"
        print(header)
        print("-" * len(header))
        for entry in entries:
            print(str(entry))

    elif args.command == "register":
        entry = registry.register(
            echelon=args.echelon,
            version=args.version,
            path=args.path,
            run_id=args.run_id,
            overwrite=args.overwrite,
        )
        registry.save()
        print(f"Registered: {entry}")


if __name__ == "__main__":
    main()
