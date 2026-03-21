# Observation Feature Reference

This document describes each dimension of the 12-dimensional normalised
observation vector returned by `BattalionEnv._get_obs()` (and its Red-
perspective mirror `_get_red_obs()`).  All values are clipped to the declared
`observation_space` bounds before being returned to the agent.

---

## Observation Vector Layout

| Index | Name            | Range     | Description |
|-------|-----------------|-----------|-------------|
| 0     | `blue_x`        | \[0, 1\]  | Blue battalion x-position, normalised by map width (`MAP_WIDTH = 1 000 m`). |
| 1     | `blue_y`        | \[0, 1\]  | Blue battalion y-position, normalised by map height (`MAP_HEIGHT = 1 000 m`). |
| 2     | `blue_cos_θ`    | \[-1, 1\] | Cosine of the Blue heading angle θ (radians). Encodes facing direction without wraparound. |
| 3     | `blue_sin_θ`    | \[-1, 1\] | Sine of the Blue heading angle θ. Combined with `blue_cos_θ` this forms a unit-circle heading representation. |
| 4     | `blue_strength` | \[0, 1\]  | Blue battalion effective fighting strength. Starts at 1.0; decreases as casualties are taken; route/destruction threshold is `DESTROYED_THRESHOLD`. |
| 5     | `blue_morale`   | \[0, 1\]  | Blue morale level. Affects fire effectiveness and routing probability. |
| 6     | `dist_norm`     | \[0, 1\]  | Euclidean distance between Blue and Red, normalised by the map diagonal (`√(MAP_WIDTH² + MAP_HEIGHT²)`). Capped at 1.0. |
| 7     | `cos_bearing`   | \[-1, 1\] | Cosine of the bearing angle from Blue to Red. Encodes relative azimuth without wraparound. |
| 8     | `sin_bearing`   | \[-1, 1\] | Sine of the bearing angle from Blue to Red. Combined with `cos_bearing` this locates Red relative to Blue. |
| 9     | `red_strength`  | \[0, 1\]  | Red battalion strength (same scale as `blue_strength`). |
| 10    | `red_morale`    | \[0, 1\]  | Red morale level (same scale as `blue_morale`). |
| 11    | `step_norm`     | \[0, 1\]  | Current step count normalised by `max_steps`. Provides episode progress context. |

---

## Tactical Significance

### Position & Geometry (indices 0–3, 6–8)
- **`blue_x`, `blue_y`**: Absolute position provides terrain context
  (whether the battalion is near the centre, flanks, or edges of the map).
- **`blue_cos_θ`, `blue_sin_θ`**: Facing direction; critical for fire arc
  and flanking manoeuvre decisions.
- **`dist_norm`**: The single most tactically important scalar — governs
  fire effectiveness, exposure, and movement strategy.
- **`cos_bearing`, `sin_bearing`**: Encodes relative position of the enemy;
  together with `dist_norm` fully specifies the 2-D relative geometry.

### Strength & Morale (indices 4–5, 9–10)
- **`blue_strength`**: Tracks own combat power; low values indicate need for
  disengagement or defensive posture.
- **`blue_morale`**: Low morale increases routing risk; the agent should
  prioritise restoring morale when this is low.
- **`red_strength`**: Drives offensive/defensive trade-offs; a weakened enemy
  suggests aggressive action.
- **`red_morale`**: A demoralised Red battalion is closer to routing; fire
  pressure is more effective.

### Temporal (index 11)
- **`step_norm`**: Late in the episode (`step_norm` → 1) the agent should
  prefer terminal actions (press the attack or lock in a draw) given the
  approaching episode boundary.

---

## Angle Representation Convention

All angles (headings and bearings) are represented as **(cos θ, sin θ)** pairs
rather than raw radians.  This avoids the discontinuity at ±π and gives the
network a continuous, bounded representation of circular quantities.  This
convention is used consistently across all environments in this codebase.

---

## Red-Perspective Observation

`_get_red_obs()` returns the same schema from Red's point of view: indices
0–5 describe Red (not Blue), and indices 9–10 describe Blue (not Red).  This
symmetric formulation allows a single policy trained as Blue to be deployed
unchanged as the Red opponent.

---

## Source

- Environment implementation: `envs/battalion_env.py` (`_get_obs`, `_get_red_obs`)
- Saliency / feature importance: `analysis/saliency.py` (`OBSERVATION_FEATURES`)
- Explainability demo: `notebooks/explainability_demo.ipynb`
