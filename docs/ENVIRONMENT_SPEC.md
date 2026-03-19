# Wargames Training — Environment Specification

Reference document for `BattalionEnv` — the Gymnasium 1v1 battalion
reinforcement learning environment.

---

## Overview

`BattalionEnv` is a continuous 2D battle simulation where the agent controls
a **Blue** battalion against a **Red** opponent.  The environment follows the
standard `gymnasium.Env` API.

```python
from envs.battalion_env import BattalionEnv

env = BattalionEnv()
obs, info = env.reset(seed=42)
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

---

## Observation Space

`Box(shape=(12,), dtype=float32)`

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | `blue_x` | `[0, 1]` | Blue x-position normalised by map width |
| 1 | `blue_y` | `[0, 1]` | Blue y-position normalised by map height |
| 2 | `cos(blue_θ)` | `[-1, 1]` | Cosine of Blue's heading angle |
| 3 | `sin(blue_θ)` | `[-1, 1]` | Sine of Blue's heading angle |
| 4 | `blue_strength` | `[0, 1]` | Blue's remaining combat strength (1 = full) |
| 5 | `blue_morale` | `[0, 1]` | Blue's current morale (1 = full) |
| 6 | `dist_to_red` | `[0, 1]` | Euclidean distance to Red, normalised by map diagonal |
| 7 | `cos(bearing_to_red)` | `[-1, 1]` | Cosine of bearing from Blue to Red |
| 8 | `sin(bearing_to_red)` | `[-1, 1]` | Sine of bearing from Blue to Red |
| 9 | `red_strength` | `[0, 1]` | Red's remaining combat strength (1 = full) |
| 10 | `red_morale` | `[0, 1]` | Red's current morale (1 = full) |
| 11 | `step_norm` | `[0, 1]` | Current step / max_steps |

**Conventions:**
- All positions are normalised by map dimensions (`map_width`, `map_height`).
- All angles are represented as `(cos θ, sin θ)` pairs — never raw radians.
- Distances are normalised by the map diagonal
  (`sqrt(map_width² + map_height²)`).

---

## Action Space

`Box(shape=(3,), dtype=float32)`

| Index | Name | Range | Effect |
|---|---|---|---|
| 0 | `move` | `[-1, 1]` | Movement: positive = forward, negative = backward; scaled by `max_speed` |
| 1 | `rotate` | `[-1, 1]` | Rotation: positive = counter-clockwise; scaled by `max_turn_rate` |
| 2 | `fire` | `[0, 1]` | Fire intensity this step (0 = cease fire, 1 = full volley) |

All action values outside the declared range are clipped by the environment.

---

## Constructor Parameters

```python
BattalionEnv(
    map_width=1000.0,         # Map width in metres
    map_height=1000.0,        # Map height in metres
    max_steps=500,            # Episode length cap
    terrain=None,             # Optional fixed TerrainMap
    randomize_terrain=True,   # Generate new terrain each episode
    hill_speed_factor=0.5,    # Speed multiplier on max-elevation hills
    curriculum_level=5,       # Red opponent difficulty (1–5)
    reward_weights=None,      # RewardWeights instance (or None for defaults)
    red_policy=None,          # Optional policy to drive Red (SB3 model)
    render_mode=None,         # None or "human"
)
```

---

## Episode Lifecycle

1. **Reset** — battalions are randomly placed on opposite sides of the map.
   Terrain is regenerated if `randomize_terrain=True`.
2. **Step** — agent submits a 3-float action; the simulation advances by one
   time step (`DT = 0.1 s`).  Both sides' damage is computed simultaneously,
   then casualties are applied.
3. **Termination** — the episode ends when:
   - Blue routes (morale drops below the routing threshold).
   - Blue is destroyed (strength ≤ 0.01).
   - Red routes or is destroyed (Blue wins).
   - `max_steps` is reached (draw / timeout).

---

## Reward Function

The reward returned each step is the sum of the following components:

| Component | Formula | Notes |
|---|---|---|
| `delta_enemy_strength` | `w * dmg_b2r` | Reward for damage dealt to Red |
| `delta_own_strength` | `-w * dmg_r2b` | Penalty for damage received by Blue |
| `survival_bonus` | `w * blue_strength` | Per-step bonus (0 by default) |
| `win_bonus` | `w` | Terminal bonus when Blue wins |
| `loss_penalty` | `w` | Terminal penalty when Blue loses (negative `w`) |
| `time_penalty` | `w` | Constant per-step penalty (negative `w`) |

Default weights: `delta_enemy_strength=5.0`, `delta_own_strength=5.0`,
`survival_bonus=0.0`, `win_bonus=10.0`, `loss_penalty=-10.0`,
`time_penalty=-0.01`.

The per-component breakdown is available in the `info` dict under
`reward/<component>` keys (e.g. `info["reward/delta_enemy_strength"]`).

---

## `info` Dictionary

`env.step()` returns an `info` dict with the following keys:

| Key | Type | Description |
|---|---|---|
| `blue_damage_dealt` | `float` | Strength fraction dealt to Red this step (`dmg_b2r`) |
| `red_damage_dealt` | `float` | Strength fraction dealt to Blue this step (`dmg_r2b`) |
| `blue_routed` | `bool` | `True` if Blue is currently routing |
| `red_routed` | `bool` | `True` if Red is currently routing |
| `step_count` | `int` | Current episode step number |
| `reward/delta_enemy_strength` | `float` | Damage-dealt reward component |
| `reward/delta_own_strength` | `float` | Damage-received penalty component |
| `reward/survival_bonus` | `float` | Survival bonus component |
| `reward/win_bonus` | `float` | Win bonus component (non-zero only on terminal step) |
| `reward/loss_penalty` | `float` | Loss penalty component (non-zero only on terminal step) |
| `reward/time_penalty` | `float` | Time penalty component |
| `reward/total` | `float` | Sum of all components (equals the returned scalar reward) |

---

## Scripted Red Opponent (Curriculum Levels)

When no `red_policy` is supplied, Red is driven by a scripted policy whose
difficulty is controlled by `curriculum_level`:

| Level | Red behaviour |
|---|---|
| 1 | **Stationary** — Red does not move or fire. |
| 2 | **Turning only** — Red faces Blue but stays put. |
| 3 | **Advance only** — Red turns and advances; no fire. |
| 4 | **Soft fire** — Red turns, advances, fires at 50 % intensity. |
| 5 | **Full combat** — Red turns, advances, fires at 100 % intensity (default). |

---

## Custom Red Policy (Self-Play)

Pass any object with a `predict(obs, deterministic=False) -> (action, state)`
method to drive Red with a trained model:

```python
from stable_baselines3 import PPO
from envs.battalion_env import BattalionEnv

red_model = PPO.load("checkpoints/best/best_model.zip")
env = BattalionEnv(red_policy=red_model, curriculum_level=5)
obs, info = env.reset()
```

Use `env.set_red_policy(new_policy)` to swap the Red policy at runtime
(e.g. from inside a training callback).

---

## Terrain System

The map is a 2D grid with elevation and cover values in `[0, 1]`.

- **Elevation** modifies movement speed: a unit on maximum-elevation terrain
  moves at `hill_speed_factor` × its normal speed.
- **Cover** reduces damage taken: higher cover reduces incoming fire
  effectiveness.
- When `randomize_terrain=True` a new procedural terrain is generated from
  the episode seed at each `reset()`.
- Pass a fixed `TerrainMap` to the constructor to use the same map every
  episode (useful for reproducible evaluation).

---

## Opponent Identifiers (Evaluation Script)

The `training/evaluate.py` script accepts the following opponent strings:

| Identifier | Description |
|---|---|
| `scripted_l1` … `scripted_l5` | Built-in scripted Red at curriculum level 1–5 |
| `random` | Red samples uniformly random actions every step |
| `<path/to/model.zip>` | Any SB3 `.zip` checkpoint drives Red |

---

## Gymnasium Compatibility

`BattalionEnv` passes the Gymnasium environment checker:

```python
from gymnasium.utils.env_checker import check_env
from envs.battalion_env import BattalionEnv

check_env(BattalionEnv())
```

---

## Simulation Constants

| Constant | Value | Description |
|---|---|---|
| `DT` | `0.1 s` | Simulation time step |
| `MAX_STEPS` | `500` | Default episode length cap |
| `MAP_WIDTH` | `1000 m` | Default map width |
| `MAP_HEIGHT` | `1000 m` | Default map height |
| `DESTROYED_THRESHOLD` | `0.01` | Strength below which a unit is considered destroyed |
