# Metrics Reference

This document defines the coordination metrics implemented in
`envs/metrics/coordination.py`.  All three metrics are computed from a
snapshot of live :class:`~envs.sim.battalion.Battalion` objects and are
logged to W&B per episode during MAPPO training.

---

## 1 · flanking_ratio

**Module path:** `envs.metrics.coordination.flanking_ratio`

### Definition

$$
\text{flanking\_ratio} =
\frac{|\{(b,r) : \lVert b - r \rVert \le r.\text{fire\_range}
      \;\land\;
      \Delta\theta(b, r) \ge r.\text{fire\_arc}\}|}
     {|\{(b,r) : \lVert b - r \rVert \le r.\text{fire\_range}\}|}
$$

where the numerator counts (blue attacker, red target) pairs in which the
blue unit is within the red unit's fire range **and** lies outside the red
unit's frontal arc, and the denominator counts all in-range pairs.

Returns `0.0` when no in-range pairs exist.

### Interpretation

| Value | Meaning |
|-------|---------|
| `0.0` | All blue attacks are frontal — no flanking |
| `> 0.3` | **Meaningful flanking behaviour** |
| `1.0` | All blue attacks come from the flank or rear |

A frontal arc of ±45° (π/4 rad) is assumed by default
(`Battalion.fire_arc`).  A blue attacker whose bearing from the red unit's
perspective exceeds ±45° from the red unit's facing direction is classified
as flanking.

### Example

```python
from envs.metrics.coordination import flanking_ratio
from envs.sim.battalion import Battalion
import math

# Red faces east (theta=0); blue is directly behind (west) → pure flanking
blue = [Battalion(x=-100.0, y=0.0, theta=0.0, strength=1.0, team=0)]
red  = [Battalion(x=0.0,    y=0.0, theta=0.0, strength=1.0, team=1)]
print(flanking_ratio(blue, red))  # 1.0
```

---

## 2 · fire_concentration

**Module path:** `envs.metrics.coordination.fire_concentration`

### Definition

For each Blue battalion, the **nearest Red target it can fire at**
(within range **and** inside its own frontal arc, per
`Battalion.can_fire_at`) is identified.  The metric is the fraction of
*firing* Blue units that all aim at the same (most-targeted) Red unit:

$$
\text{fire\_concentration} =
\frac{\max_r \;|\{b : \text{target}(b) = r\}|}
     {|\{b : \exists r, \text{can\_fire\_at}(b, r)\}|}
$$

Returns `0.0` when no Blue unit can fire at any Red unit.

### Interpretation

| Value | Meaning |
|-------|---------|
| `0.0` | No blue unit is currently able to fire |
| `1/k` | Fire evenly spread across `k` red targets |
| `1.0` | All firing blue units concentrate on a single red target |

High fire concentration (focus fire) is generally desirable — it routes
the targeted red unit faster than distributed fire.

### Example

```python
from envs.metrics.coordination import fire_concentration
import math

# Both blue units face west and are close to the single red unit
blue = [
    Battalion(x=100.0, y=0.0, theta=math.pi, strength=1.0, team=0),
    Battalion(x=90.0,  y=5.0, theta=math.pi, strength=1.0, team=0),
]
red = [Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=1)]
print(fire_concentration(blue, red))  # 1.0
```

---

## 3 · mutual_support_score

**Module path:** `envs.metrics.coordination.mutual_support_score`

### Definition

$$
\text{mutual\_support\_score} =
\frac{1}{n_b} \sum_{i=1}^{n_b}
\frac{|\{j \ne i : \lVert b_i - b_j \rVert \le R_s\}|}{n_b - 1}
$$

where $n_b$ is the number of Blue battalions and $R_s$ is the
`support_radius` (default **300 m**, 1.5× the default fire range of 200 m).

Returns `0.0` when fewer than 2 Blue units are present.

### Interpretation

| Value | Meaning |
|-------|---------|
| `0.0` | Every blue unit is isolated (or only 1 unit) |
| `1.0` | Every blue unit is within support range of every other |

Units within mutual-support range can rapidly reorient to cover each
other's flanks, reducing vulnerability to enemy flanking manoeuvres.

### Example

```python
from envs.metrics.coordination import mutual_support_score

blue_close = [
    Battalion(x=0.0,   y=0.0, theta=0.0, strength=1.0, team=0),
    Battalion(x=100.0, y=0.0, theta=0.0, strength=1.0, team=0),
]
print(mutual_support_score(blue_close, support_radius=300.0))  # 1.0
```

---

## 4 · compute_all (convenience wrapper)

**Module path:** `envs.metrics.coordination.compute_all`

Returns all three metrics in a single `dict[str, float]` ready for direct
logging to W&B:

```python
from envs.metrics.coordination import compute_all

metrics = compute_all(blue_battalions, red_battalions, support_radius=300.0)
# {
#   "coordination/flanking_ratio":       0.67,
#   "coordination/fire_concentration":   0.50,
#   "coordination/mutual_support_score": 0.83,
# }
```

---

## 5 · W&B Logging

During MAPPO training (`training/train_mappo.py`) the metrics are computed
at **every environment step** and averaged over the episode, then logged to
W&B at each logging interval under the keys:

| W&B Key | Source metric |
|---------|---------------|
| `coordination/flanking_ratio` | `flanking_ratio()` |
| `coordination/fire_concentration` | `fire_concentration()` |
| `coordination/mutual_support_score` | `mutual_support_score()` |

The `MultiBattalionEnv.get_coordination_metrics()` method exposes the
computation at any point during an episode.

---

## 6 · Notes & Caveats

* **Routing** — Dead and routed battalions are excluded from metric
  computation (`strength > 0` and `routed == False`).  This means metrics
  can drop to `0.0` late in an episode as units are eliminated.
* **Fog of war** — Metrics operate on ground-truth battalion positions, not
  the fog-of-war observations visible to agents.
* **Fire arc convention** — `Battalion.fire_arc = π/4` means ±45°.
  `flanking_ratio` uses the *target's* fire arc to classify flanking, not
  the attacker's.
* **Units** — All distance arguments (`support_radius`, `fire_range`) are
  in metres, consistent with the default map scale (1 km × 1 km).
