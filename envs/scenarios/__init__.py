"""Historical scenario loading, batch import, and outcome comparison (E5.4, E11.1)."""

from envs.scenarios.historical import (
    HistoricalScenario,
    HistoricalOutcome,
    ScenarioUnit,
    ScenarioLoader,
    OutcomeComparator,
    ComparisonResult,
    load_scenario,
)
from envs.scenarios.importer import (
    BatchScenarioImporter,
    BattleRecord,
)

__all__ = [
    "HistoricalScenario",
    "HistoricalOutcome",
    "ScenarioUnit",
    "ScenarioLoader",
    "OutcomeComparator",
    "ComparisonResult",
    "load_scenario",
    # E11.1 batch importer
    "BatchScenarioImporter",
    "BattleRecord",
]
