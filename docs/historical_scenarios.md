# Historical Scenario Design Methodology

## Overview

This document describes how Napoleonic battle scenarios are designed,
encoded as YAML files, and loaded into the simulation engine for
historical validation experiments (Epic E5.4).

---

## Goals

1. **Calibrate** the simulation engine against documented real-world outcomes.
2. **Stress-test** trained policies by initialising them in historically
   significant tactical configurations.
3. **Discover** whether agents reproduce historically-documented tactics or
   find novel superior alternatives.

---

## Scenario YAML Schema

Each scenario is a single YAML file stored under
`configs/scenarios/historical/`.  The file is parsed by
`envs.scenarios.historical.ScenarioLoader`.

### Top-Level Keys

| Key                  | Required | Description |
|----------------------|----------|-------------|
| `scenario`           | Yes      | Battle metadata (name, date, description) |
| `factions`           | No       | Display names for blue and red sides |
| `terrain`            | No       | Terrain configuration (defaults to flat) |
| `units`              | Yes      | Initial-condition lists for `blue` and `red` |
| `historical_outcome` | Yes      | Documented result used as validation baseline |

### `scenario` Block

```yaml
scenario:
  name: "Battle of Waterloo (1815)"   # Display name
  date: "1815-06-18"                  # ISO-8601
  description: >
    Narrative context (multi-line OK).
```

### `factions` Block

```yaml
factions:
  blue: "Anglo-Allied (Wellington)"
  red:  "French Grande Armée (Napoleon)"
```

### `terrain` Block

```yaml
terrain:
  type: "generated"   # "flat" or "generated"
  width: 1000.0       # map width in metres
  height: 1000.0
  rows: 20            # grid resolution
  cols: 20
  seed: 1815          # RNG seed for generated terrain
  n_hills: 4          # number of Gaussian elevation blobs
  n_forests: 3        # number of Gaussian cover/forest blobs
```

When `type` is `"flat"`, `n_hills` and `n_forests` are ignored.

### `units` Block

```yaml
units:
  blue:
    - id: "1st_division"   # unique identifier (snake_case recommended)
      x: 400.0             # world-space position in metres
      y: 650.0
      theta: "south"       # facing angle (radians or named direction)
      strength: 0.75       # relative strength in [0, 1]
  red:
    - id: "old_guard"
      x: 380.0
      y: 300.0
      theta: "north"
      strength: 0.95
```

**Named directions for `theta`:**

| Name     | Radians | Description         |
|----------|---------|---------------------|
| `east`   | 0       | Facing positive-x   |
| `north`  | π/2     | Facing positive-y   |
| `west`   | π       | Facing negative-x   |
| `south`  | −π/2    | Facing negative-y   |

Numeric radians (e.g. `1.57`) are also accepted.

### `historical_outcome` Block

```yaml
historical_outcome:
  winner: 0              # 0=blue, 1=red, null=draw/inconclusive
  blue_casualties: 0.36  # fraction of blue force lost [0, 1]
  red_casualties: 0.68
  duration_steps: 420    # indicative battle length in simulation steps
  description: >
    Free-text summary of the historical outcome.
```

---

## Simulation Scale Convention

Each YAML unit represents a **brigade-level formation** (~2,000–8,000 men)
rather than a single battalion, to allow manageable scenario sizes.
The mapping rules are:

| Real-world unit | `strength` value | Notes |
|-----------------|------------------|-------|
| Fresh full-strength brigade | 1.00 | Maximum effectiveness |
| Tired / partially engaged   | 0.80–0.95 | Moderate fatigue |
| Already-engaged / worn      | 0.60–0.79 | Significant losses |
| Heavily attrited            | 0.40–0.59 | Combat ineffective soon |
| Depleted remnant            | < 0.40    | Near-routing condition |

Positions are placed on a 1,000 × 1,000 m field.  A rough scale guide:

* 1 m (simulation) ≈ 1–5 m (real-world), depending on the battle frontage.
* For a 5 km frontage, use 1 m (sim) = 5 m (real).

---

## Casualty Calculation

Historical casualty figures are sourced from academic battle histories and
expressed as the **fraction of the engaged force that became casualties**
(killed, wounded, or captured):

```
blue_casualties = total_allied_casualties / total_allied_engaged
```

These are used as the validation baseline in `OutcomeComparator`.  The
comparator measures the absolute difference between simulated and historical
casualty rates for each side, combined with winner accuracy and duration
fidelity into a single **fidelity score** ∈ [0, 1].

---

## Bundled Scenarios

### 1. Battle of Waterloo (1815-06-18)

**File:** `configs/scenarios/historical/waterloo.yaml`

Represents the climactic evening phase: Wellington's exhausted
Anglo-Allied line defends the ridge of Mont-Saint-Jean against Napoleon's
final Imperial Guard assault, with Prussian reinforcements arriving on the
French right flank.

| Metric | Value |
|--------|-------|
| Blue (Allied) units | 4 |
| Red (French) units  | 5 |
| Historical winner   | Blue (Allied) |
| Allied casualties   | ~36 % |
| French casualties   | ~68 % |
| Terrain             | Generated (ridge + orchards) |

### 2. Battle of Austerlitz (1805-12-02)

**File:** `configs/scenarios/historical/austerlitz.yaml`

Napoleon's tactical masterpiece: he feigned weakness on his right to draw
the Allied left wing off the Pratzen Heights, then struck the weakened
centre with Soult's corps.

| Metric | Value |
|--------|-------|
| Blue (French) units  | 4 |
| Red (Allied) units   | 5 |
| Historical winner    | Blue (French) |
| French casualties    | ~12 % |
| Allied casualties    | ~47 % |
| Terrain              | Generated (Pratzen plateau) |

### 3. Battle of Borodino (1812-09-07)

**File:** `configs/scenarios/historical/borodino.yaml`

The bloodiest day of the Napoleonic Wars: Napoleon's frontal assaults on
fortified Russian earthworks produced enormous casualties on both sides
but no decisive strategic result.

| Metric | Value |
|--------|-------|
| Blue (French) units | 4 |
| Red (Russian) units | 5 |
| Historical winner   | Draw (null) |
| French casualties   | ~35 % |
| Russian casualties  | ~40 % |
| Terrain             | Generated (redoubts + forests) |

---

## Adding New Scenarios

1. Create a new YAML file in `configs/scenarios/historical/`.
2. Follow the schema above.  Use the three bundled scenarios as templates.
3. Source casualty and order-of-battle data from peer-reviewed histories or
   established wargame order-of-battle databases (e.g., ORBAT references).
4. Add at least one test in `tests/test_historical_scenarios.py` that:
   - Loads the new file without error.
   - Verifies the expected number of blue and red units.
   - Verifies the historical winner field.
5. Run `python -m pytest tests/test_historical_scenarios.py` to confirm.

---

## Outcome Comparison Methodology

`OutcomeComparator.compare(episode_result)` returns a `ComparisonResult`
with the following sub-scores:

| Sub-score   | Weight | Calculation |
|-------------|--------|-------------|
| **Winner accuracy** | 0.5 | 1.0 if winner matches, else 0.0 |
| **Casualty accuracy** | 0.3 | `1 − 0.5 × (|Δblue| + |Δred|)`, clamped to [0, 1] |
| **Duration accuracy** | 0.2 | `1 − |Δsteps| / duration_steps`, clamped to [0, 1] |

**Fidelity score** = weighted mean of the three sub-scores ∈ [0, 1].

A fidelity score ≥ 0.7 is considered a **good historical match**.
Scores < 0.5 indicate significant divergence from the historical record.

---

## References

* Chandler, D. G. (1966). *The Campaigns of Napoleon*. Macmillan.
* Uffindell, A. (2003). *The Eagle's Last Triumph: Napoleon's Victory at Ligny, June 1815*. Greenhill Books.
* Riehn, R. K. (1990). *1812: Napoleon's Russian Campaign*. McGraw-Hill.
* Duffy, C. (1977). *Austerlitz 1805*. Seeley Service & Co.
