"""Historical scenario loader and outcome comparator (E5.4).

This module provides:

* :class:`ScenarioUnit` — initial-condition descriptor for a single battalion.
* :class:`HistoricalOutcome` — documented historical result used as a baseline.
* :class:`HistoricalScenario` — complete scenario specification loaded from YAML.
* :class:`ScenarioLoader` — converts a YAML file into a :class:`HistoricalScenario`
  together with live :class:`~envs.sim.battalion.Battalion` and
  :class:`~envs.sim.terrain.TerrainMap` objects ready for the simulation engine.
* :class:`OutcomeComparator` — measures how closely a simulated
  :class:`~envs.sim.engine.EpisodeResult` reproduces the
  :class:`HistoricalOutcome`.
* :class:`ComparisonResult` — structured result of a single comparison.

Typical usage::

    from envs.scenarios import load_scenario

    scenario, blue_battalions, red_battalions, terrain = load_scenario(
        "configs/scenarios/historical/waterloo.yaml"
    )
    print(scenario.name, len(blue_battalions), len(red_battalions))

    from envs.sim.engine import SimEngine
    from envs.scenarios import OutcomeComparator

    result = SimEngine(blue_battalions[0], red_battalions[0], terrain=terrain).run()
    cmp = OutcomeComparator(scenario.historical_outcome).compare(result)
    print(cmp.winner_matches, cmp.casualty_delta_blue, cmp.casualty_delta_red)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from envs.sim.battalion import Battalion
from envs.sim.engine import EpisodeResult
from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class ScenarioUnit:
    """Initial conditions for a single battalion in a scenario.

    Attributes
    ----------
    unit_id:
        Human-readable identifier (e.g. ``"old_guard"``).
    x, y:
        World-space position in metres.
    theta:
        Facing angle in radians.  ``0`` = facing east, ``π`` = facing west.
    strength:
        Initial relative strength in ``[0, 1]``.
    team:
        ``0`` for blue (player / trained policy), ``1`` for red (opponent).
    """

    unit_id: str
    x: float
    y: float
    theta: float
    strength: float
    team: int

    def to_battalion(self) -> Battalion:
        """Construct a :class:`~envs.sim.battalion.Battalion` from this descriptor."""
        battalion = Battalion(
            x=self.x,
            y=self.y,
            theta=self.theta,
            strength=float(np.clip(self.strength, 0.0, 1.0)),
            team=self.team,
        )
        # Preserve the scenario unit identifier on the created battalion for tracing.
        setattr(battalion, "unit_id", self.unit_id)
        return battalion


@dataclass
class HistoricalOutcome:
    """Documented historical result used as the validation baseline.

    Attributes
    ----------
    winner:
        ``0`` if blue (player side) won historically, ``1`` if red won,
        ``None`` if the battle was a draw or inconclusive.
    blue_casualties:
        Fraction of blue force lost ``[0, 1]``.
    red_casualties:
        Fraction of red force lost ``[0, 1]``.
    duration_steps:
        Approximate battle duration expressed in simulation steps.
        Purely indicative — used only for soft comparison.
    description:
        Free-text summary of the historical outcome.
    """

    winner: Optional[int]
    blue_casualties: float
    red_casualties: float
    duration_steps: int
    description: str = ""


@dataclass
class TerrainConfig:
    """Terrain specification from the YAML file.

    Attributes
    ----------
    terrain_type:
        One of ``"flat"``, ``"generated"``, or ``"gis"``.
    width, height:
        Map dimensions in metres.  For GIS terrain these are ignored in
        favour of the site's geographic bounding-box dimensions.
    rows, cols:
        Grid resolution (used for flat / generated / GIS terrain).
    seed:
        Random seed for terrain generation (only used when
        ``terrain_type == "generated"``).
    n_hills:
        Number of elevation blobs for generated terrain.
    n_forests:
        Number of forest/cover blobs for generated terrain.
    gis_site:
        Battle-site identifier for GIS terrain (e.g. ``"waterloo"``).
        Required when ``terrain_type == "gis"``.
    gis_data_dir:
        Optional directory containing ``{site}.tif`` and ``{site}.osm``
        files.  When absent the synthetic GIS fallback is used.
    """

    terrain_type: str = "flat"
    width: float = 1000.0
    height: float = 1000.0
    rows: int = 20
    cols: int = 20
    seed: int = 0
    n_hills: int = 3
    n_forests: int = 2
    gis_site: str = ""
    gis_data_dir: str = ""


@dataclass
class HistoricalScenario:
    """Complete scenario specification.

    Attributes
    ----------
    name:
        Scenario display name (e.g. ``"Battle of Waterloo (1815)"``).
    date:
        ISO-8601 date string (e.g. ``"1815-06-18"``).
    description:
        Narrative context.
    faction_blue, faction_red:
        Names of the two opposing forces.
    blue_units, red_units:
        Ordered lists of :class:`ScenarioUnit` objects for each side.
    terrain_config:
        Terrain parameters.
    historical_outcome:
        The documented historical result.
    source_path:
        Path to the YAML file from which this scenario was loaded
        (``None`` if constructed programmatically).
    """

    name: str
    date: str
    description: str
    faction_blue: str
    faction_red: str
    blue_units: List[ScenarioUnit] = field(default_factory=list)
    red_units: List[ScenarioUnit] = field(default_factory=list)
    terrain_config: TerrainConfig = field(default_factory=TerrainConfig)
    historical_outcome: HistoricalOutcome = field(
        default_factory=lambda: HistoricalOutcome(
            winner=None,
            blue_casualties=0.0,
            red_casualties=0.0,
            duration_steps=500,
        )
    )
    source_path: Optional[Path] = field(default=None, repr=False)

    @property
    def n_blue(self) -> int:
        """Number of blue battalions."""
        return len(self.blue_units)

    @property
    def n_red(self) -> int:
        """Number of red battalions."""
        return len(self.red_units)

    def build_battalions(
        self,
    ) -> Tuple[List[Battalion], List[Battalion]]:
        """Instantiate :class:`~envs.sim.battalion.Battalion` objects.

        Returns
        -------
        blue_battalions, red_battalions:
            Two lists of :class:`Battalion` objects ready for the simulation
            engine.  Each call returns freshly constructed instances, so
            repeated calls do not share mutable state.
        """
        blue = [u.to_battalion() for u in self.blue_units]
        red = [u.to_battalion() for u in self.red_units]
        return blue, red

    def build_terrain(self) -> TerrainMap:
        """Construct the :class:`~envs.sim.terrain.TerrainMap` for this scenario."""
        cfg = self.terrain_config
        if cfg.terrain_type == "generated":
            return TerrainMap.generate_random(
                rng=np.random.default_rng(cfg.seed),
                width=cfg.width,
                height=cfg.height,
                rows=cfg.rows,
                cols=cfg.cols,
                num_hills=cfg.n_hills,
                num_forests=cfg.n_forests,
            )
        if cfg.terrain_type == "gis":
            return self._build_gis_terrain(cfg)
        # Default: flat terrain
        return TerrainMap.flat(
            width=cfg.width,
            height=cfg.height,
            rows=cfg.rows,
            cols=cfg.cols,
        )

    @staticmethod
    def _build_gis_terrain(cfg: "TerrainConfig") -> TerrainMap:
        """Build a GIS-sourced :class:`~envs.sim.terrain.TerrainMap`.

        Imports :mod:`data.gis.terrain_importer` at call time so that the
        heavy-weight dependency is only required when actually using GIS
        terrain — the rest of the module works without it.

        Raises
        ------
        ValueError
            If ``cfg.gis_site`` is empty or not a known battle site.
        """
        from data.gis.terrain_importer import GISTerrainBuilder  # noqa: PLC0415

        site = cfg.gis_site.strip()
        if not site:
            raise ValueError(
                "terrain_type is 'gis' but gis_site is empty. "
                "Set gis_site to one of: waterloo, austerlitz, borodino, salamanca."
            )
        data_dir = cfg.gis_data_dir.strip() or None
        builder = GISTerrainBuilder(
            site=site,
            rows=cfg.rows,
            cols=cfg.cols,
            srtm_path=(Path(data_dir) / f"{site}.tif") if data_dir else None,
            osm_path=(Path(data_dir) / f"{site}.osm") if data_dir else None,
        )
        return builder.build()


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------


class ScenarioLoader:
    """Load a :class:`HistoricalScenario` from a YAML file.

    The YAML schema is documented in ``docs/historical_scenarios.md``.

    Parameters
    ----------
    path:
        Path to the YAML file.  Can be absolute or relative to the
        current working directory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> HistoricalScenario:
        """Parse the YAML file and return a :class:`HistoricalScenario`.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        KeyError / ValueError
            If required fields are missing or have invalid values.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Scenario YAML not found: {self.path}"
            )
        with self.path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        # yaml.safe_load() returns None for empty files or files with only comments.
        # Normalise to a dict and validate the top-level structure so callers
        # consistently see ValueError for invalid content, as documented.
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid scenario YAML structure in {self.path}: "
                "expected a mapping at the top level."
            )
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: dict) -> HistoricalScenario:
        meta = raw.get("scenario", {})
        factions = raw.get("factions", {})
        terrain_raw = raw.get("terrain", {})
        outcome_raw = raw.get("historical_outcome", {})
        units_raw = raw.get("units", {})

        terrain_cfg = self._parse_terrain(terrain_raw)
        outcome = self._parse_outcome(outcome_raw)
        blue_units = [
            self._parse_unit(u, team=0)
            for u in units_raw.get("blue", [])
        ]
        red_units = [
            self._parse_unit(u, team=1)
            for u in units_raw.get("red", [])
        ]

        return HistoricalScenario(
            name=str(meta.get("name", "Unknown Scenario")),
            date=str(meta.get("date", "")),
            description=str(meta.get("description", "")),
            faction_blue=str(factions.get("blue", "Blue")),
            faction_red=str(factions.get("red", "Red")),
            blue_units=blue_units,
            red_units=red_units,
            terrain_config=terrain_cfg,
            historical_outcome=outcome,
            source_path=self.path,
        )

    @staticmethod
    def _parse_terrain(raw: dict) -> TerrainConfig:
        return TerrainConfig(
            terrain_type=str(raw.get("type", "flat")),
            width=float(raw.get("width", 1000.0)),
            height=float(raw.get("height", 1000.0)),
            rows=int(raw.get("rows", 20)),
            cols=int(raw.get("cols", 20)),
            seed=int(raw.get("seed", 0)),
            n_hills=int(raw.get("n_hills", 3)),
            n_forests=int(raw.get("n_forests", 2)),
            gis_site=str(raw.get("gis_site", "")),
            gis_data_dir=str(raw.get("gis_data_dir", "")),
        )

    @staticmethod
    def _parse_outcome(raw: dict) -> HistoricalOutcome:
        raw_winner = raw.get("winner")
        if raw_winner is None or str(raw_winner).lower() in ("null", "draw", "none", ""):
            winner: Optional[int] = None
        else:
            winner = int(raw_winner)
        return HistoricalOutcome(
            winner=winner,
            blue_casualties=float(raw.get("blue_casualties", 0.0)),
            red_casualties=float(raw.get("red_casualties", 0.0)),
            duration_steps=int(raw.get("duration_steps", 500)),
            description=str(raw.get("description", "")),
        )

    @staticmethod
    def _parse_unit(raw: dict, team: int) -> ScenarioUnit:
        theta_raw = raw.get("theta", 0.0)
        # Support named angles for convenience
        _NAMED = {
            "east": 0.0,
            "west": math.pi,
            "north": math.pi / 2,
            "south": -math.pi / 2,
        }
        _ALLOWED_DIRECTIONS = ", ".join(sorted(_NAMED.keys()))
        if isinstance(theta_raw, str):
            key = theta_raw.lower()
            if key in _NAMED:
                theta = _NAMED[key]
            else:
                raise ValueError(
                    f"Unknown theta direction name {theta_raw!r}; "
                    f"expected one of: {_ALLOWED_DIRECTIONS}"
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


# ---------------------------------------------------------------------------
# Outcome comparator
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing a simulated episode to a historical outcome.

    Attributes
    ----------
    winner_matches:
        ``True`` if the simulated winner equals the historical winner
        (both ``None`` counts as a match).
    casualty_delta_blue:
        ``simulated_blue_casualties − historical_blue_casualties``.
        Negative values mean the simulation underestimates losses.
    casualty_delta_red:
        ``simulated_red_casualties − historical_red_casualties``.
    duration_delta:
        ``simulated_steps − historical_duration_steps``.
    fidelity_score:
        Scalar in ``[0, 1]``.  ``1.0`` = perfect match across all metrics;
        ``0.0`` = maximum divergence.  Computed as a weighted mean of
        sub-scores: winner (weight 0.5), casualty accuracy (weight 0.3),
        duration accuracy (weight 0.2).
    historical_outcome:
        The baseline used for this comparison.
    simulated_winner:
        Winner derived from the :class:`~envs.sim.engine.EpisodeResult`.
    simulated_blue_casualties:
        Blue strength lost ``= 1 − result.blue_strength``.
    simulated_red_casualties:
        Red strength lost ``= 1 − result.red_strength``.
    simulated_steps:
        Episode length in steps.
    """

    winner_matches: bool
    casualty_delta_blue: float
    casualty_delta_red: float
    duration_delta: int
    fidelity_score: float
    historical_outcome: HistoricalOutcome
    simulated_winner: Optional[int]
    simulated_blue_casualties: float
    simulated_red_casualties: float
    simulated_steps: int


class OutcomeComparator:
    """Compare a simulated episode result to a historical outcome.

    Parameters
    ----------
    historical_outcome:
        The :class:`HistoricalOutcome` extracted from the scenario YAML.
    """

    def __init__(self, historical_outcome: HistoricalOutcome) -> None:
        self.historical_outcome = historical_outcome

    def compare(self, result: EpisodeResult) -> ComparisonResult:
        """Compare *result* to the stored historical outcome.

        Parameters
        ----------
        result:
            An :class:`~envs.sim.engine.EpisodeResult` produced by
            :class:`~envs.sim.engine.SimEngine`.

        Returns
        -------
        ComparisonResult
        """
        ho = self.historical_outcome
        sim_blue_cas = float(np.clip(1.0 - result.blue_strength, 0.0, 1.0))
        sim_red_cas = float(np.clip(1.0 - result.red_strength, 0.0, 1.0))
        sim_winner = result.winner

        winner_matches = sim_winner == ho.winner

        casualty_delta_blue = sim_blue_cas - ho.blue_casualties
        casualty_delta_red = sim_red_cas - ho.red_casualties
        duration_delta = result.steps - ho.duration_steps

        # --- Fidelity sub-scores ---
        # Winner: binary 0 / 1
        winner_score = 1.0 if winner_matches else 0.0

        # Casualty accuracy: mean of 1 − |delta| for each side, clamped to [0, 1]
        cas_score = float(
            np.clip(
                1.0 - 0.5 * (abs(casualty_delta_blue) + abs(casualty_delta_red)),
                0.0,
                1.0,
            )
        )

        # Duration accuracy: linear score — 1.0 at zero error, 0.0 when the
        # simulated duration differs by 100 % of the historical duration.
        if ho.duration_steps > 0:
            rel_dur_err = abs(duration_delta) / ho.duration_steps
        else:
            rel_dur_err = 0.0
        dur_score = float(np.clip(1.0 - rel_dur_err, 0.0, 1.0))

        fidelity_score = 0.5 * winner_score + 0.3 * cas_score + 0.2 * dur_score

        return ComparisonResult(
            winner_matches=winner_matches,
            casualty_delta_blue=casualty_delta_blue,
            casualty_delta_red=casualty_delta_red,
            duration_delta=duration_delta,
            fidelity_score=float(fidelity_score),
            historical_outcome=ho,
            simulated_winner=sim_winner,
            simulated_blue_casualties=sim_blue_cas,
            simulated_red_casualties=sim_red_cas,
            simulated_steps=result.steps,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def load_scenario(
    path: str | Path,
) -> Tuple[HistoricalScenario, List[Battalion], List[Battalion], TerrainMap]:
    """Load a scenario YAML and return ready-to-use simulation objects.

    Parameters
    ----------
    path:
        Path to the YAML scenario file.

    Returns
    -------
    scenario:
        The parsed :class:`HistoricalScenario`.
    blue_battalions:
        Freshly constructed blue :class:`~envs.sim.battalion.Battalion` list.
    red_battalions:
        Freshly constructed red :class:`~envs.sim.battalion.Battalion` list.
    terrain:
        The :class:`~envs.sim.terrain.TerrainMap` for this scenario.
    """
    scenario = ScenarioLoader(path).load()
    blue, red = scenario.build_battalions()
    terrain = scenario.build_terrain()
    return scenario, blue, red, terrain
