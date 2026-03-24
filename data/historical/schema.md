# Historical Battle Database Schema

This document describes the JSON/CSV schema for historical Napoleonic battle
records stored in `data/historical/`.  These records are loaded by
`envs/scenarios/importer.py` (`BatchScenarioImporter`) and used by the
historical benchmark runner in `training/historical_benchmark.py`.

## Data Sources

Battle records are drawn from three primary sources:

| Source | Description |
|--------|-------------|
| **Napoleon's Battles** | Tabletop wargame rules with detailed OOBs and historical notes |
| **Corsican Ogre** | Boardgame covering Napoleon's Italian and Egyptian campaigns |
| **Nafziger OOBs** | George Nafziger's authoritative Orders of Battle collection |

## JSON Schema

The primary database is `data/historical/battles.json` — a JSON array of
battle objects.  Each object has the following fields:

```json
{
  "id":          "waterloo_1815",
  "name":        "Battle of Waterloo (1815)",
  "date":        "1815-06-18",
  "location":    "Belgium",
  "description": "Narrative summary of the engagement.",
  "source":      "Napoleon's Battles",

  "factions": {
    "blue": "Anglo-Allied (Wellington)",
    "red":  "French Grande Armee (Napoleon)"
  },

  "terrain": {
    "type":      "generated",
    "width":     1000.0,
    "height":    1000.0,
    "rows":      20,
    "cols":      20,
    "seed":      1815,
    "n_hills":   4,
    "n_forests": 3
  },

  "units": {
    "blue": [
      {
        "id":       "1st_division_guard",
        "x":        400.0,
        "y":        650.0,
        "theta":    -1.5708,
        "strength": 0.75
      }
    ],
    "red": [
      {
        "id":       "old_guard_1st",
        "x":        380.0,
        "y":        300.0,
        "theta":    1.5708,
        "strength": 0.95
      }
    ]
  },

  "historical_outcome": {
    "winner":          0,
    "blue_casualties": 0.36,
    "red_casualties":  0.68,
    "duration_steps":  420,
    "description":     "Wellington repulsed the Imperial Guard assault."
  }
}
```

### Field Reference

#### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique snake_case identifier, e.g. `"waterloo_1815"` |
| `name` | string | ✅ | Human-readable battle name |
| `date` | string | ✅ | ISO-8601 date, e.g. `"1815-06-18"` |
| `location` | string | — | Geographic location |
| `description` | string | — | Narrative context |
| `source` | string | — | Primary source reference |
| `factions` | object | — | `{"blue": "…", "red": "…"}` |
| `terrain` | object | — | See terrain fields below |
| `units` | object | — | `{"blue": […], "red": […]}` |
| `historical_outcome` | object | ✅ | See outcome fields below |

#### Terrain fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `"flat"` | `"flat"` or `"generated"` |
| `width` | float | `1000.0` | Map width in metres |
| `height` | float | `1000.0` | Map height in metres |
| `rows` | int | `20` | Grid rows |
| `cols` | int | `20` | Grid columns |
| `seed` | int | `0` | RNG seed (only used for `"generated"`) |
| `n_hills` | int | `3` | Number of elevation blobs |
| `n_forests` | int | `2` | Number of forest/cover blobs |

#### Unit fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto | Human-readable unit ID |
| `x` | float | `500.0` | World-space x position in metres |
| `y` | float | `500.0` | World-space y position in metres |
| `theta` | float or string | `0.0` | Facing angle in radians, or one of `"east"`, `"west"`, `"north"`, `"south"` |
| `strength` | float | `1.0` | Relative strength `[0, 1]` |

#### Historical outcome fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `winner` | int or null | ✅ | `0` = blue won, `1` = red won, `null` = draw |
| `blue_casualties` | float | ✅ | Fraction of blue force lost `[0, 1]` |
| `red_casualties` | float | ✅ | Fraction of red force lost `[0, 1]` |
| `duration_steps` | int | ✅ | Approximate battle duration in simulation steps |
| `description` | string | — | Free-text historical outcome summary |

## CSV Schema

A CSV file can also be used with `BatchScenarioImporter`.  The CSV must
contain at minimum the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique battle ID |
| `name` | Battle name |
| `date` | ISO-8601 date |
| `winner` | `0`, `1`, or `null`/`draw`/empty |
| `blue_casualties` | Float `[0, 1]` |
| `red_casualties` | Float `[0, 1]` |
| `duration_steps` | Integer |

Optional columns for terrain: `terrain_type`, `terrain_width`,
`terrain_height`, `terrain_rows`, `terrain_cols`, `terrain_seed`,
`n_hills`, `n_forests`.

Optional faction columns: `faction_blue`, `faction_red`.

When unit positions are not provided in a CSV row, two default battalions
per side are placed at sensible positions.

## Conventions

- **Positions** are in metres on a `1000 × 1000` map.
- **Simulation scale**: each battalion represents one historical brigade
  or division (~3 000–10 000 men depending on the engagement).
- **Angles**: `theta = 0` faces east, `theta = π/2` faces north,
  `theta = π` faces west, `theta = -π/2` faces south.
- **Blue** is always the player/trained-agent side.
- **Red** is always the opponent/scripted-AI side.
