"""Batch scenario importer for E11.1 — Historical Battle Database.

Reads 50+ Napoleonic battle records from JSON or CSV files and converts
them to :class:`~envs.scenarios.historical.HistoricalScenario` objects
ready for use with the simulation engine.

Supported input formats
-----------------------
* **JSON** — An array of battle-record objects (see ``data/historical/battles.json``).
* **CSV** — Flat CSV with at least the columns ``id``, ``name``, ``date``,
  ``winner``, ``blue_casualties``, ``red_casualties``, ``duration_steps``.
  Terrain and unit positions are set to sensible defaults when absent.

Typical usage::

    from envs.scenarios.importer import BatchScenarioImporter

    importer = BatchScenarioImporter("data/historical/battles.json")
    scenarios = importer.load_all()
    print(f"Loaded {len(scenarios)} scenarios")

    # Or filter by source
    napoleon_battles = importer.load_by_source("Napoleon's Battles")
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from envs.scenarios.historical import (
    HistoricalOutcome,
    HistoricalScenario,
    ScenarioUnit,
    TerrainConfig,
)


# ---------------------------------------------------------------------------
# Public dataclass for a raw battle record (before conversion)
# ---------------------------------------------------------------------------


@dataclass
class BattleRecord:
    """Raw battle record parsed from JSON/CSV.

    This is an intermediate representation — use
    :meth:`BatchScenarioImporter.load_all` to obtain fully-constructed
    :class:`~envs.scenarios.historical.HistoricalScenario` objects.

    Attributes
    ----------
    battle_id:
        Unique snake_case identifier (e.g. ``"waterloo_1815"``).
    name:
        Human-readable battle name.
    date:
        ISO-8601 date string.
    location:
        Geographic location.
    description:
        Narrative summary.
    source:
        Source reference (e.g. ``"Napoleon's Battles"``,
        ``"Corsican Ogre"``, ``"Nafziger OOBs"``).
    factions:
        Mapping with keys ``"blue"`` and ``"red"``.
    terrain:
        Raw terrain dict (mirrors :class:`~envs.scenarios.historical.TerrainConfig`
        fields).
    blue_units, red_units:
        Lists of raw unit dicts with keys ``id``, ``x``, ``y``,
        ``theta``, ``strength``.
    historical_outcome:
        Raw outcome dict with keys ``winner``, ``blue_casualties``,
        ``red_casualties``, ``duration_steps``, ``description``.
    """

    battle_id: str
    name: str
    date: str
    location: str = ""
    description: str = ""
    source: str = ""
    factions: Dict[str, str] = field(default_factory=dict)
    terrain: Dict = field(default_factory=dict)
    blue_units: List[Dict] = field(default_factory=list)
    red_units: List[Dict] = field(default_factory=list)
    historical_outcome: Dict = field(default_factory=dict)

    def to_scenario(self) -> HistoricalScenario:
        """Convert to a :class:`~envs.scenarios.historical.HistoricalScenario`."""
        return _record_to_scenario(self)


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class BatchScenarioImporter:
    """Load multiple historical scenarios from a single JSON or CSV file.

    Parameters
    ----------
    path:
        Path to the JSON or CSV file containing battle records.
        The file type is determined from the extension (``.json`` or
        ``.csv``); case is ignored.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not ``.json`` or ``.csv``, or if the
        file content cannot be parsed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_records(self) -> List[BattleRecord]:
        """Parse the file and return raw :class:`BattleRecord` objects.

        Returns
        -------
        List[BattleRecord]
            One record per battle in the input file.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Battle records file not found: {self.path}"
            )
        ext = self.path.suffix.lower()
        if ext == ".json":
            return self._load_json()
        elif ext == ".csv":
            return self._load_csv()
        else:
            raise ValueError(
                f"Unsupported file format {ext!r}. "
                "Expected '.json' or '.csv'."
            )

    def load_all(self) -> List[HistoricalScenario]:
        """Load and convert all records to :class:`HistoricalScenario` objects.

        Returns
        -------
        List[HistoricalScenario]
            Fully-constructed scenario objects, one per battle record.
        """
        return [rec.to_scenario() for rec in self.load_records()]

    def load_by_source(self, source: str) -> List[HistoricalScenario]:
        """Load scenarios whose ``source`` field matches *source* exactly.

        Parameters
        ----------
        source:
            Source string to filter on (case-sensitive), e.g.
            ``"Napoleon's Battles"``.

        Returns
        -------
        List[HistoricalScenario]
        """
        return [
            rec.to_scenario()
            for rec in self.load_records()
            if rec.source == source
        ]

    def load_by_id(self, battle_id: str) -> Optional[HistoricalScenario]:
        """Load a single scenario by its ``battle_id``.

        Parameters
        ----------
        battle_id:
            The ``id`` field value, e.g. ``"waterloo_1815"``.

        Returns
        -------
        HistoricalScenario or None
            ``None`` if no record with this ID exists.
        """
        for rec in self.load_records():
            if rec.battle_id == battle_id:
                return rec.to_scenario()
        return None

    # ------------------------------------------------------------------
    # Internal: JSON loading
    # ------------------------------------------------------------------

    def _load_json(self) -> List[BattleRecord]:
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a JSON array in {self.path}, got {type(raw).__name__}."
            )
        records: List[BattleRecord] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Entry {i} in {self.path} is not a JSON object."
                )
            records.append(_parse_json_entry(entry))
        return records

    # ------------------------------------------------------------------
    # Internal: CSV loading
    # ------------------------------------------------------------------

    def _load_csv(self) -> List[BattleRecord]:
        records: List[BattleRecord] = []
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(_parse_csv_row(row))
        return records


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_entry(raw: dict) -> BattleRecord:
    units_raw = raw.get("units", {})
    return BattleRecord(
        battle_id=str(raw.get("id", raw.get("battle_id", "unknown"))),
        name=str(raw.get("name", "Unknown Battle")),
        date=str(raw.get("date", "")),
        location=str(raw.get("location", "")),
        description=str(raw.get("description", "")),
        source=str(raw.get("source", "")),
        factions=dict(raw.get("factions", {})),
        terrain=dict(raw.get("terrain", {})),
        blue_units=list(units_raw.get("blue", [])),
        red_units=list(units_raw.get("red", [])),
        historical_outcome=dict(raw.get("historical_outcome", {})),
    )


def _parse_csv_row(row: Dict[str, str]) -> BattleRecord:
    """Convert a flat CSV row to a :class:`BattleRecord`.

    Core columns used when present: ``id``, ``name``, ``date``, ``winner``,
    ``blue_casualties``, ``red_casualties``, ``duration_steps``.
    Missing or invalid values are replaced with permissive defaults
    (e.g. casualties → 0.0, duration_steps → 500) so that the scenario
    can still run.  Unit positions and terrain are set to sensible defaults
    when the corresponding optional columns are absent.
    """
    def _float(val: str, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _int(val: str, default: int = 0) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    raw_winner = row.get("winner", "")
    if raw_winner.lower() in ("null", "draw", "none", ""):
        winner_val = None
    else:
        try:
            winner_val = int(raw_winner)
        except ValueError:
            winner_val = None

    historical_outcome = {
        "winner": winner_val,
        "blue_casualties": _float(row.get("blue_casualties", "0.0")),
        "red_casualties": _float(row.get("red_casualties", "0.0")),
        "duration_steps": _int(row.get("duration_steps", "500")),
        "description": row.get("description", ""),
    }

    # Generate minimal default units if not embedded in CSV
    blue_units = [
        {"id": "blue_0", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0},
        {"id": "blue_1", "x": 400.0, "y": 500.0, "theta": 0.0, "strength": 0.90},
    ]
    red_units = [
        {"id": "red_0", "x": 700.0, "y": 500.0, "theta": 3.1416, "strength": 1.0},
        {"id": "red_1", "x": 600.0, "y": 500.0, "theta": 3.1416, "strength": 0.90},
    ]

    battle_id = row.get("id", row.get("battle_id", "unknown"))
    terrain = {
        "type": row.get("terrain_type", "flat"),
        "width": _float(row.get("terrain_width", "1000.0"), 1000.0),
        "height": _float(row.get("terrain_height", "1000.0"), 1000.0),
        "rows": _int(row.get("terrain_rows", "20"), 20),
        "cols": _int(row.get("terrain_cols", "20"), 20),
        "seed": _int(row.get("terrain_seed", "0"), 0),
        "n_hills": _int(row.get("n_hills", "3"), 3),
        "n_forests": _int(row.get("n_forests", "2"), 2),
    }

    return BattleRecord(
        battle_id=str(battle_id),
        name=str(row.get("name", "Unknown Battle")),
        date=str(row.get("date", "")),
        location=str(row.get("location", "")),
        description=str(row.get("description", "")),
        source=str(row.get("source", "")),
        factions={
            "blue": row.get("faction_blue", "Blue"),
            "red": row.get("faction_red", "Red"),
        },
        terrain=terrain,
        blue_units=blue_units,
        red_units=red_units,
        historical_outcome=historical_outcome,
    )


def _record_to_scenario(rec: BattleRecord) -> HistoricalScenario:
    """Convert a :class:`BattleRecord` to a :class:`HistoricalScenario`."""
    # Terrain
    t = rec.terrain
    terrain_cfg = TerrainConfig(
        terrain_type=str(t.get("type", "flat")),
        width=float(t.get("width", 1000.0)),
        height=float(t.get("height", 1000.0)),
        rows=int(t.get("rows", 20)),
        cols=int(t.get("cols", 20)),
        seed=int(t.get("seed", 0)),
        n_hills=int(t.get("n_hills", 3)),
        n_forests=int(t.get("n_forests", 2)),
    )

    # Historical outcome
    o = rec.historical_outcome
    raw_winner = o.get("winner")
    if raw_winner is None or (
        isinstance(raw_winner, str)
        and raw_winner.lower() in ("null", "draw", "none", "")
    ):
        winner: Optional[int] = None
    else:
        try:
            winner_int = int(raw_winner)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid historical outcome winner {raw_winner!r}; "
                "expected 0, 1, or a null/draw sentinel."
            ) from exc
        if winner_int not in (0, 1):
            raise ValueError(
                f"Invalid historical outcome winner {winner_int!r}; "
                "expected 0 or 1."
            )
        winner = winner_int

    outcome = HistoricalOutcome(
        winner=winner,
        blue_casualties=float(o.get("blue_casualties", 0.0)),
        red_casualties=float(o.get("red_casualties", 0.0)),
        duration_steps=int(o.get("duration_steps", 500)),
        description=str(o.get("description", "")),
    )

    # Units
    blue_units = [_parse_unit(u, team=0) for u in rec.blue_units]
    red_units = [_parse_unit(u, team=1) for u in rec.red_units]

    # Factions
    factions = rec.factions
    faction_blue = str(factions.get("blue", "Blue"))
    faction_red = str(factions.get("red", "Red"))

    return HistoricalScenario(
        name=rec.name,
        date=rec.date,
        description=rec.description,
        faction_blue=faction_blue,
        faction_red=faction_red,
        blue_units=blue_units,
        red_units=red_units,
        terrain_config=terrain_cfg,
        historical_outcome=outcome,
        source_path=None,  # no YAML path; loaded from JSON/CSV
    )


_NAMED_ANGLES: Dict[str, float] = {
    "east": 0.0,
    "west": math.pi,
    "north": math.pi / 2,
    "south": -math.pi / 2,
}


def _parse_unit(raw: dict, team: int) -> ScenarioUnit:
    theta_raw = raw.get("theta", 0.0)
    if isinstance(theta_raw, str):
        key = theta_raw.lower()
        theta = _NAMED_ANGLES.get(key)
        if theta is None:
            allowed = ", ".join(sorted(_NAMED_ANGLES.keys()))
            raise ValueError(
                f"Unknown theta direction {theta_raw!r}; "
                f"expected one of: {allowed}"
            )
    else:
        theta = float(theta_raw)

    return ScenarioUnit(
        unit_id=str(raw.get("id", f"unit_{team}")),
        x=float(raw.get("x", 500.0)),
        y=float(raw.get("y", 500.0)),
        theta=theta,
        strength=float(raw.get("strength", 1.0)),
        team=team,
    )
