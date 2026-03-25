# SPDX-License-Identifier: MIT
"""Seed v6–v12 roadmap issues: epics, key feature tasks, and sprint planning issues.

Idempotent — skips any issue whose title already exists.

Run via the seed-v6-v12-issues GitHub Actions workflow or locally with:
    GITHUB_TOKEN=... REPO_NAME=owner/repo python seed_v6_v12_issues.py
    GITHUB_TOKEN=... REPO_NAME=owner/repo DRY_RUN=true python seed_v6_v12_issues.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Attribution footer appended to every created issue body
# ---------------------------------------------------------------------------

ATTRIBUTION = (
    "\n\n---\n"
    "> 🤖 *Strategist Forge* created this issue automatically as part of v6–v12 roadmap seeding.\n"
    f"> Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n"
    "> Label `status: agent-created` was requested; it may be absent if the label does not exist on this repository.\n"
)

# Fallback color map for auto-created labels.  Matches the color scheme in
# setup_labels_and_milestones.py so auto-created labels stay visually consistent
# with those created by the bootstrap script.
_LABEL_COLOR_MAP: dict[str, str] = {
    "v6:":    "fff3cd",
    "v7:":    "d1ecf1",
    "v8:":    "d6d8db",
    "v9:":    "c3e6cb",
    "v10:":   "f5c6cb",
    "v11:":   "bee5eb",
    "v12:":   "e2e3e5",
    "type:":  "7057ff",
    "priority:": "fbca04",
    "status:":   "0075ca",
    "domain:":   "bfd4f2",
}


def _label_color(label_name: str) -> str:
    """Return an appropriate hex color for a label, based on its prefix."""
    for prefix, color in _LABEL_COLOR_MAP.items():
        if label_name.startswith(prefix):
            return color
    return "ededed"  # fallback grey

# ---------------------------------------------------------------------------
# ── v6 EPICS — Physics-Accurate Simulation Foundation ────────────────────
# ---------------------------------------------------------------------------
# v6 goal: Replace the abstract 2D point-mass simulation with a physics-
# accurate model of Napoleonic-era combat: terrain elevation, line-of-sight
# blockage, realistic weapon ranges & reload times, morale cascades, supply
# consumption, and weather effects. This is the long-term foundation for all
# strategic and operational layers that follow.
# ---------------------------------------------------------------------------

V6_EPICS: list[dict] = [
    {
        "title": "[EPIC] E6.1 — Terrain Elevation & Line-of-Sight Engine",
        "labels": [
            "type: epic", "priority: high",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Version

v6

### Goal

Replace the flat 2D terrain with a heightmap-driven terrain engine.
Line-of-sight calculations account for ridgelines, forests, and
built-up areas.  Movement costs are elevation-gradient-aware.
JAX-accelerated for step-time parity with the current engine.

### Motivation & Context

All historical Napoleonic engagements were shaped by terrain.  Flat terrain
produces unrealistic agent behaviours (no high-ground advantage, no screening)
and limits the tactical depth the agent can learn.  This epic is the
prerequisite for all v6+ realism improvements.

### Child Issues (Tasks)

- [ ] Design heightmap data format (GeoTIFF / procedural noise) and loader
- [ ] Implement `envs/sim/terrain_engine.py` — elevation queries, slope, cover
- [ ] Implement Bresenham-style line-of-sight on heightmap (JAX-compiled)
- [ ] Integrate terrain cost into movement model
- [ ] Update `BattalionEnv` observation space with elevation and cover features
- [ ] Add terrain generator for procedural training maps
- [ ] Add `tests/test_terrain_engine.py`

### Acceptance Criteria

- [ ] LOS correctly blocked by ridgelines in at least 3 unit test cases
- [ ] Terrain-aware movement cost increases episode length on hilly maps (verified empirically)
- [ ] Step throughput ≥ 80 % of flat-terrain baseline (JAX compilation)
- [ ] Heightmap procedural generator produces maps with controllable ruggedness parameter

### Priority

high

### Target Milestone

M13: Physics Simulation""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.2 — Realistic Weapon Ranges, Accuracy & Reload Cycles",
        "labels": [
            "type: epic", "priority: high",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Version

v6

### Goal

Replace the abstract DPS model with historically-grounded musket/cannon
combat parameters: effective range bands (close/effective/extreme), accuracy
distributions based on range and formation, reload time cycles (2–4 step
delay for muskets, 6–10 for artillery), and volley fire mechanics.

### Motivation & Context

Current combat resolution is an instantaneous damage function.  Real
Napoleonic tactics were dominated by range management and timing volleys.
Without reload cycles and range bands the agents cannot discover these
core tactics (hold fire until close range, controlled volley, counter-battery).

### Child Issues (Tasks)

- [ ] Implement `envs/sim/weapons.py` — weapon profiles (musket, rifle, cannon, howitzer)
- [ ] Implement range-band accuracy distributions (hit probability per target at range)
- [ ] Implement per-unit reload state machine (loaded → firing → reloading → loaded)
- [ ] Implement volley synchronization (coordinated fire window for a formed line)
- [ ] Update action space: separate `fire` / `hold_fire` / `advance_to_range` actions
- [ ] Add `tests/test_weapons.py` with historical accuracy cross-validation

### Acceptance Criteria

- [ ] Musket hit probability at 50m / 150m / 300m within ±5 % of Nafziger (1988) data
- [ ] Reload state machine correctly blocks firing during reload cycle
- [ ] Agents spontaneously learn to close to effective range before volley
- [ ] Artillery suppression effect (morale penalty without casualties) implemented

### Priority

high

### Target Milestone

M13: Physics Simulation""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.3 — Morale, Cohesion & Rout Mechanics",
        "labels": [
            "type: epic", "priority: high",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Version

v6

### Goal

Model battalion morale as a continuous state variable that degrades under
fire, friendly casualties, flank exposure, and commander distance.  A morale
threshold triggers cohesion loss; continued degradation triggers rout (forced
withdrawal) and eventual dispersal.

### Motivation & Context

Morale is the decisive factor in Napoleonic combat — most battles were decided
by rout rather than annihilation.  Without morale the agents cannot learn to
use psychological pressure (demonstration, feint, pursuit of routing units).

### Child Issues (Tasks)

- [ ] Implement `envs/sim/morale.py` — morale state machine with stressors
- [ ] Integrate morale into observation vector (normalized morale, cohesion)
- [ ] Implement rout: forced movement away from enemy; dispersal at 0 morale
- [ ] Implement morale recovery: distance from enemy, nearby friendly support
- [ ] Implement commander proximity bonus (battalion within brigade CO range)
- [ ] Add morale-driven reward shaping (bonus for breaking enemy, penalty for routing)
- [ ] Add `tests/test_morale.py`

### Acceptance Criteria

- [ ] Battalion routs reliably after sustained flanking fire in unit tests
- [ ] Routed units automatically move away from attackers (no agent control)
- [ ] Morale recovery rate configurable per scenario difficulty
- [ ] Agents learn to exploit routing enemies (pursuit) without explicit reward

### Priority

high

### Target Milestone

M13: Physics Simulation""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.4 — Formation System (Line, Column, Square, Skirmish)",
        "labels": [
            "type: epic", "priority: high",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Version

v6

### Goal

Implement the four canonical Napoleonic formations as discrete states with
associated firepower, movement speed, and vulnerability modifiers.  Formation
changes require transition time, creating tactical trade-offs.

### Motivation & Context

Formation choice is the central tactical decision in Napoleonic warfare:
line maximizes firepower, column maximizes speed, square counters cavalry,
skirmish enables loose screening.  Without formation the action space lacks
its most important dimension.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/formations.py` — Formation enum + attribute tables
- [ ] Implement formation transition timing (1–3 steps depending on formation pair)
- [ ] Apply formation modifiers to firepower, movement, morale resilience
- [ ] Add formation as discrete action and current formation to observation
- [ ] Implement cavalry charge vs. square resolution (cavalry bounces off square)
- [ ] Implement skirmisher screen mechanics (extended order, independent fire)
- [ ] Add `tests/test_formations.py`

### Acceptance Criteria

- [ ] All four formations accessible via action space
- [ ] Square reliably defeats unformed cavalry in unit tests
- [ ] Agents learn to form square proactively when cavalry threat detected
- [ ] Formation transition delay matches Nosworthy (1990) estimates (±20 %)

### Priority

high

### Target Milestone

M13: Physics Simulation""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.5 — Supply, Ammunition & Fatigue Model",
        "labels": [
            "type: epic", "priority: medium",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M14: v6 Complete",
        "body": """\
### Version

v6

### Goal

Track per-battalion ammunition, food/water supply, and cumulative fatigue.
Resupply requires proximity to a supply wagon unit or depot.  Sustained
operations without resupply degrade effectiveness.

### Motivation & Context

Supply and attrition shaped every Napoleonic campaign.  Without logistics
agents cannot discover the critical skill of conserving ammo, protecting
supply lines, and timing attacks to exploit an exhausted enemy.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/logistics.py` — ammo / supply / fatigue tracking
- [ ] Implement supply wagon unit type: slow, fragile, high-value target
- [ ] Integrate ammunition depletion into weapon firing (jams at 0 ammo)
- [ ] Implement fatigue accumulation (movement + combat) and recovery (halt)
- [ ] Add resupply zone mechanic (battalion halted near wagon recovers ammo/food)
- [ ] Expose supply state in observation vector
- [ ] Add `tests/test_logistics.py`

### Acceptance Criteria

- [ ] Agents that ignore resupply run out of ammo in ≥ 80 % of long episodes
- [ ] Supply wagons become emergent high-priority targets for opposing agents
- [ ] Fatigue correctly reduces movement speed and accuracy
- [ ] Resupply mechanic is configurable (off for short training episodes)

### Priority

medium

### Target Milestone

M14: v6 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.6 — Weather & Time-of-Day Effects",
        "labels": [
            "type: epic", "priority: medium",
            "v6: simulation", "domain: env", "status: agent-created",
        ],
        "milestone": "M14: v6 Complete",
        "body": """\
### Version

v6

### Goal

Parameterize episodes with weather conditions (clear, overcast, rain, fog,
snow) and time of day (dawn, day, dusk, night).  These affect visibility,
accuracy, movement speed, and morale.

### Motivation & Context

Weather and daylight were major operational constraints in the Napoleonic era.
Rain misfires, fog obscures, snow slows — all create randomization that
forces agents to generalize rather than memorize fixed strategies.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/weather.py` — weather state + effect multipliers
- [ ] Implement time-of-day progression (configurable day length)
- [ ] Apply weather to LOS range, accuracy, movement speed, morale stressor
- [ ] Add weather as observation features (weather_id, visibility_fraction)
- [ ] Create scenario YAML with weather override and progression schedule
- [ ] Add `tests/test_weather.py`

### Acceptance Criteria

- [ ] Heavy rain reduces musket accuracy by ≥ 30 % (historical misfire rate)
- [ ] Fog reduces LOS range to < 100 m (based on density setting)
- [ ] Agents generalize across weather conditions (no > 15 % drop in win rate across conditions)
- [ ] Weather randomization enabled by default in training configs

### Priority

medium

### Target Milestone

M14: v6 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E6.7 — v6 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v6: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M14: v6 Complete",
        "body": """\
### Version

v6

### Goal

Complete documentation for all v6 simulation components, publish the v6
release, and validate that existing v1–v5 agents can train on the new engine
(with degraded but non-zero performance).

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v6 epics complete
- [ ] Write `docs/simulation_engine.md` — full physics model reference
- [ ] Write `docs/historical_validation.md` — accuracy cross-references
- [ ] Draft `CHANGELOG.md` v6 section
- [ ] Benchmark: re-train v1 agent on v6 engine; verify convergence within 2× budget
- [ ] Create GitHub release tag `v6.0.0`
- [ ] Seed v7 planning issues

### Acceptance Criteria

- [ ] All v6 epics marked complete in ROADMAP.md
- [ ] v1 agent achieves > 55 % win rate on v6 engine (may require re-training)
- [ ] `v6.0.0` release tag exists with release notes
- [ ] v7 epics seeded and triaged

### Priority

low

### Target Milestone

M14: v6 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v6 TASKS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

V6_TASKS: list[dict] = [
    {
        "title": "Implement `envs/sim/terrain_engine.py` with JAX-compiled LOS",
        "labels": ["type: feature", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": "JAX-compiled Bresenham LOS on a 2-D heightmap. "
                "Expose `los(src, dst) -> bool` and `elevation(pos) -> float` APIs. "
                "Must achieve ≥ 80 % throughput vs. flat-terrain baseline.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/sim/weapons.py` — Napoleonic weapon profiles",
        "labels": ["type: feature", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": "Weapon profiles: smoothbore musket, rifle, 6-pdr cannon, howitzer. "
                "Each has `range_bands`, `accuracy_table`, `reload_steps`, and "
                "`suppression_radius`. Accuracy validated against Nafziger (1988).\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/sim/formations.py` — Formation enum and modifiers",
        "labels": ["type: feature", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": "Formation enum: LINE, COLUMN, SQUARE, SKIRMISH. "
                "Attribute multipliers for firepower, speed, and morale resilience. "
                "Transition time table based on Nosworthy (1990).\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/sim/morale.py` — morale state machine with rout",
        "labels": ["type: feature", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": "Morale [0, 1] degrades under stressors (fire, casualties, flank exposure). "
                "Threshold at 0.3 → cohesion loss (reduced effectiveness). "
                "Threshold at 0.0 → rout (forced withdrawal, agent loses control).\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/sim/logistics.py` — ammo, supply, fatigue tracking",
        "labels": ["type: feature", "priority: medium", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M14: v6 Complete",
        "body": "Per-battalion ammo counter (decrements on fire, replenished near supply wagon). "
                "Fatigue accumulator (increases with movement/combat, decreases on halt). "
                "Both exposed in observation vector as normalized [0, 1] values.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/sim/weather.py` — weather effects on simulation",
        "labels": ["type: feature", "priority: medium", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M14: v6 Complete",
        "body": "Weather enum: CLEAR, OVERCAST, RAIN, FOG, SNOW. "
                "Effect multipliers: visibility fraction, accuracy penalty, speed penalty. "
                "Weather randomized per episode from configurable distribution.\n" + ATTRIBUTION,
    },
    {
        "title": "Add procedural map generator with configurable ruggedness",
        "labels": ["type: feature", "priority: medium", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": "Perlin noise heightmap generator with configurable `ruggedness` (0 = flat, 1 = alpine). "
                "Supports forest/town feature placement. Used for domain randomization in training.\n" + ATTRIBUTION,
    },
    {
        "title": "Add cavalry unit type with charge mechanics",
        "labels": ["type: feature", "priority: medium", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M14: v6 Complete",
        "body": "Cavalry battalion: fast (3× infantry speed), weak vs. square, devastating vs. disordered infantry. "
                "Charge action: high-speed approach for one step, then melee resolution. "
                "Agents must learn to form square proactively when cavalry detected.\n" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v7 EPICS — Operational Scale (Corps / Army) ───────────────────────────
# ---------------------------------------------------------------------------

V7_EPICS: list[dict] = [
    {
        "title": "[EPIC] E7.1 — Corps-Level Operational Environment",
        "labels": [
            "type: epic", "priority: high",
            "v7: operational", "domain: env", "status: agent-created",
        ],
        "milestone": "M15: Corps Command",
        "body": """\
### Version

v7

### Goal

Extend the HRL stack to a corps-level operational environment: 3–5 divisions
per side, each division containing 3–4 brigades of 4–6 battalions.
The map expands to 20–50 km². Operational objectives replace tactical
kill-counting as primary rewards.

### Motivation & Context

v3 reached division-level HRL.  Corps command introduces road networks,
multi-axis maneuver, inter-division coordination, and operational tempo —
the decisive level of Napoleonic warfare (Napoleon's corps system).

### Child Issues (Tasks)

- [ ] Implement `envs/corps_env.py` — multi-division ParallelEnv wrapper
- [ ] Design corps-level observation space (sector control, road usage, supply lines)
- [ ] Implement operational objectives: capture objective hex, cut supply line, fix-and-flank
- [ ] Implement road network for movement speed bonuses on roads
- [ ] Implement inter-division communication radius
- [ ] Add `tests/test_corps_env.py`

### Acceptance Criteria

- [ ] Corps env supports 2 corps per side (6+ divisions total)
- [ ] Road network correctly applies 1.5× movement speed bonus
- [ ] Operational objectives produce emergent multi-axis maneuver in evaluation
- [ ] Episode length configurable (1 hour to full day of combat)

### Priority

high

### Target Milestone

M15: Corps Command""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E7.2 — Strategic Supply & Logistics Network",
        "labels": [
            "type: epic", "priority: high",
            "v7: operational", "domain: env", "status: agent-created",
        ],
        "milestone": "M15: Corps Command",
        "body": """\
### Version

v7

### Goal

Model strategic supply chains: depots, supply convoys, and the supply line
radius within which units can draw provisions.  Cutting supply lines is an
operational objective.  Units that are out of supply degrade in effectiveness
over hours.

### Motivation & Context

Napoleon's dictum "an army marches on its stomach" is the core operational
constraint at corps level.  Supply interdiction was decisive at Salamanca,
Vitoria, and the 1812 Russian campaign.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/supply_network.py` — depot nodes, convoy routes, supply radius
- [ ] Implement supply consumption: each step subtracts from nearby depot stock
- [ ] Implement supply line interdiction: capturing/destroying a depot degrades supply radius
- [ ] Add supply state to corps-level observation (supply_level per division)
- [ ] Add `tests/test_supply_network.py`

### Acceptance Criteria

- [ ] Cutting the main supply line degrades opponent effectiveness within 5 game hours
- [ ] Supply convoy becomes a high-value target discovered by agents without explicit reward
- [ ] Supply line metric logged to W&B per episode

### Priority

high

### Target Milestone

M15: Corps Command""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E7.3 — Multi-Corps Self-Play & League Extension",
        "labels": [
            "type: epic", "priority: medium",
            "v7: operational", "domain: ml", "status: agent-created",
        ],
        "milestone": "M16: v7 Complete",
        "body": """\
### Version

v7

### Goal

Extend the v4 league training infrastructure to operate at corps scale.
Main agents, exploiters, and league exploiters all operate in the
`CorpsEnv`.  Nash equilibrium sampling at the operational level.

### Child Issues (Tasks)

- [ ] Port `training/league/train_main_agent.py` to corps-level env
- [ ] Add corps-level snapshot versioning (larger model, longer training)
- [ ] Extend `MatchDatabase` with operational result fields (territory, casualties, supply)
- [ ] Benchmark distributed league on 8-GPU cluster
- [ ] Add `tests/test_corps_league.py`

### Acceptance Criteria

- [ ] Corps-level main agent beats scripted corps opponent ≥ 55 % after 5M steps
- [ ] Nash entropy remains > 0 (no strategy collapse) after 10M steps
- [ ] 8-GPU Ray cluster achieves ≥ 5× single-GPU throughput

### Priority

medium

### Target Milestone

M16: v7 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E7.4 — v7 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v7: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M16: v7 Complete",
        "body": """\
### Version

v7

### Goal

Complete documentation for corps-level components; publish `v7.0.0`.
Release includes tutorial: "From battalion to corps in 3 training runs."

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v7 epics complete
- [ ] Write `docs/corps_command_guide.md`
- [ ] Write `docs/operational_design.md` — objectives, supply, road networks
- [ ] Draft `CHANGELOG.md` v7 section
- [ ] Create GitHub release tag `v7.0.0`
- [ ] Seed v8 planning issues

### Acceptance Criteria

- [ ] All v7 epics marked complete in ROADMAP.md
- [ ] `v7.0.0` release tag exists with release notes

### Priority

low

### Target Milestone

M16: v7 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v8 EPICS — Transformer Policy & Attention Architecture ────────────────
# ---------------------------------------------------------------------------

V8_EPICS: list[dict] = [
    {
        "title": "[EPIC] E8.1 — Entity-Based Observation & Transformer Policy",
        "labels": [
            "type: epic", "priority: high",
            "v8: architecture", "domain: ml", "status: agent-created",
        ],
        "milestone": "M17: Transformer Policy",
        "body": """\
### Version

v8

### Goal

Replace the fixed-size concatenation observation with a variable-length
sequence of entity tokens (one per visible unit).  A transformer encoder
(multi-head self-attention) processes the sequence.  This enables the policy
to generalise to arbitrary army sizes without architectural changes.

### Motivation & Context

As army size grows to corps scale the fixed-size observation vector becomes
impractical.  Entity-based observations + transformers are the state-of-the-art
for multi-agent games (AlphaStar, OpenAI Five).  Variable-length encoding also
improves sample efficiency on smaller scenarios.

### Child Issues (Tasks)

- [ ] Design entity token schema: unit type, position, formation, ammo, morale, team
- [ ] Implement `models/entity_encoder.py` — multi-head self-attention over entity tokens
- [ ] Implement positional encoding for 2-D spatial positions
- [ ] Implement masking for variable-length entity sets (padding mask)
- [ ] Integrate entity encoder into actor and centralized critic
- [ ] Ablation: entity-transformer vs. flat-MLP on 4v4 and 8v8 scenarios
- [ ] Add `tests/test_entity_encoder.py`

### Acceptance Criteria

- [ ] Entity encoder handles variable N (1–64 units) without shape errors
- [ ] Attention ablation: entity-transformer ≥ flat-MLP win rate on 8v8 (documented in `[EXP]`)
- [ ] Inference latency < 8 ms on CPU for 32-entity input
- [ ] Attention weight visualizations added to explainability notebook

### Priority

high

### Target Milestone

M17: Transformer Policy""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E8.2 — Memory Module (LSTM / Temporal Context)",
        "labels": [
            "type: epic", "priority: medium",
            "v8: architecture", "domain: ml", "status: agent-created",
        ],
        "milestone": "M17: Transformer Policy",
        "body": """\
### Version

v8

### Goal

Add a recurrent memory module (LSTM or gated transformer) so agents can
track unit positions that have moved out of LOS, remember recent enemy
behaviour, and maintain operational intent across timesteps.

### Motivation & Context

Fog-of-war created in v6 means agents cannot rely on a fully-observable
state.  Recurrent memory allows agents to form internal models of unobserved
enemy positions — critical for realistic operational planning.

### Child Issues (Tasks)

- [ ] Implement `models/recurrent_policy.py` — LSTM wrapped around entity encoder
- [ ] Update rollout buffer to store LSTM hidden states across episodes
- [ ] Evaluate: LSTM vs. frame stacking vs. no memory on fog-of-war scenarios
- [ ] Add recurrent state checkpointing for multi-step inference
- [ ] Add `tests/test_recurrent_policy.py`

### Acceptance Criteria

- [ ] LSTM policy outperforms memoryless policy on fog-of-war scenario (documented in `[EXP]`)
- [ ] Hidden state correctly reset at episode boundaries
- [ ] Rollout buffer memory overhead < 20 % vs. non-recurrent baseline

### Priority

medium

### Target Milestone

M17: Transformer Policy""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E8.3 — Model Scaling & Hyperparameter Study",
        "labels": [
            "type: epic", "priority: medium",
            "v8: architecture", "domain: ml", "status: agent-created",
        ],
        "milestone": "M18: v8 Complete",
        "body": """\
### Version

v8

### Goal

Systematic scaling study: vary transformer depth (2–8 layers), width
(64–512 hidden), and attention heads (2–16).  Identify the Pareto frontier
of performance vs. inference latency for deployment.

### Child Issues (Tasks)

- [ ] Run W&B sweep: transformer depth × width × heads on 4v4 scenario
- [ ] Measure: win rate, convergence steps, inference latency
- [ ] Select and document the recommended "small", "medium", "large" configs
- [ ] Add `configs/models/transformer_small.yaml`, `transformer_medium.yaml`, `transformer_large.yaml`

### Acceptance Criteria

- [ ] Sweep covers ≥ 18 configurations
- [ ] Model configs documented in `docs/model_configs.md`
- [ ] "Small" model inference < 5 ms CPU; "large" model < 20 ms

### Priority

medium

### Target Milestone

M18: v8 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E8.4 — v8 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v8: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M18: v8 Complete",
        "body": """\
### Version

v8

### Goal

Document the transformer architecture, publish `v8.0.0`, and provide
a migration guide for users upgrading from MLP policies.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v8 epics complete
- [ ] Write `docs/transformer_policy.md`
- [ ] Write `docs/model_migration_guide.md` — MLP → transformer upgrade
- [ ] Draft `CHANGELOG.md` v8 section
- [ ] Create GitHub release tag `v8.0.0`
- [ ] Seed v9 planning issues

### Acceptance Criteria

- [ ] All v8 epics marked complete
- [ ] `v8.0.0` release tag exists

### Priority

low

### Target Milestone

M18: v8 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v9 EPICS — Human-in-the-Loop & Decision Support ──────────────────────
# ---------------------------------------------------------------------------

V9_EPICS: list[dict] = [
    {
        "title": "[EPIC] E9.1 — Interactive Web-Based Wargame Interface",
        "labels": [
            "type: epic", "priority: high",
            "v9: interface", "domain: viz", "status: agent-created",
        ],
        "milestone": "M19: Decision Support",
        "body": """\
### Version

v9

### Goal

A web-based real-time wargaming interface where a human commander
controls one side at any echelon (battalion, brigade, corps) and an
AI agent controls the other.  Built with React + WebGL renderer; AI
served via the v5 ONNX policy server.

### Child Issues (Tasks)

- [ ] Design UI: hex-grid map, unit panel, formation selector, order queue
- [ ] Implement `frontend/` React app with WebGL map renderer
- [ ] Implement WebSocket game loop: browser ↔ Python sim ↔ AI policy
- [ ] Integrate ONNX policy server for sub-10 ms AI response
- [ ] Add scenario editor: drag-and-drop unit placement, weather/terrain config
- [ ] Add replay viewer with step-through and annotation

### Acceptance Criteria

- [ ] Human can issue orders at battalion level via click/drag
- [ ] AI responds within 100 ms per game step
- [ ] Interface runs in Chrome/Firefox without plugins
- [ ] Replay can be exported as JSON and re-loaded

### Priority

high

### Target Milestone

M19: Decision Support""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E9.2 — AI-Assisted Course of Action (COA) Planning Tool",
        "labels": [
            "type: epic", "priority: high",
            "v9: coa", "domain: ml", "status: agent-created",
        ],
        "milestone": "M19: Decision Support",
        "body": """\
### Version

v9

### Goal

Extend the v5 COA generator into a full decision-support planning tool.
Given a current battle state, generate 5–10 ranked COAs with predicted
probability of success, expected casualties, and time-to-objective.
Human planners can modify a COA and re-evaluate.

### Child Issues (Tasks)

- [ ] Upgrade `analysis/coa_generator.py` to corps-level with v7 env
- [ ] Implement COA modification: human edits a COA and requests re-simulation
- [ ] Implement COA explanation: highlight which unit actions drive success
- [ ] Integrate with web interface: COA panel with ranked list and map overlay
- [ ] Add `[EXP]` validation: do AI COAs outperform expert human planners?

### Acceptance Criteria

- [ ] 10 COAs generated in < 120 seconds on CPU (Monte-Carlo rollouts)
- [ ] Top-ranked COA has ≥ 60 % agreement with expert planner's choice in user study
- [ ] COA explanation highlights ≥ 3 key decisions per COA

### Priority

high

### Target Milestone

M19: Decision Support""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E9.3 — After-Action Review & Training Feedback Loop",
        "labels": [
            "type: epic", "priority: medium",
            "v9: training", "domain: ml", "status: agent-created",
        ],
        "milestone": "M20: v9 Complete",
        "body": """\
### Version

v9

### Goal

Implement an after-action review (AAR) system that captures human player
decisions, annotates them with AI-predicted quality scores, and feeds
disagreements back into training as demonstration data (DAgger / GAIL).

### Child Issues (Tasks)

- [ ] Implement `training/human_feedback.py` — record human actions + state
- [ ] Implement DAgger-style training loop using human demonstrations
- [ ] Implement GAIL discriminator to reward AI for matching human style
- [ ] Build AAR interface: timeline, decision annotations, alternative paths
- [ ] Add `tests/test_human_feedback.py`

### Acceptance Criteria

- [ ] DAgger-trained policy achieves 10 % better win rate than base policy on human-designed scenarios
- [ ] GAIL discriminator accuracy > 65 % (human vs. random policy)
- [ ] AAR correctly annotates ≥ 90 % of decisive turning points

### Priority

medium

### Target Milestone

M20: v9 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E9.4 — v9 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v9: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M20: v9 Complete",
        "body": """\
### Version

v9

### Goal

Document the decision-support system; publish `v9.0.0` with a live demo
deployment on a public server.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v9 epics complete
- [ ] Write `docs/decision_support_guide.md`
- [ ] Deploy live demo (Docker Compose: sim + policy server + frontend)
- [ ] Draft `CHANGELOG.md` v9 section
- [ ] Create GitHub release tag `v9.0.0`
- [ ] Seed v10 planning issues

### Acceptance Criteria

- [ ] Live demo accessible at a stable URL
- [ ] `v9.0.0` release tag exists

### Priority

low

### Target Milestone

M20: v9 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v10 EPICS — Multi-Domain & Joint Operations ───────────────────────────
# ---------------------------------------------------------------------------

V10_EPICS: list[dict] = [
    {
        "title": "[EPIC] E10.1 — Naval Unit Type & Coastal Operations",
        "labels": [
            "type: epic", "priority: high",
            "v10: multi-domain", "domain: env", "status: agent-created",
        ],
        "milestone": "M21: Joint Operations",
        "body": """\
### Version

v10

### Goal

Add naval vessels (ships-of-the-line, frigates, gunboats) operating on
river and coastal map tiles.  Naval fire support provides bombardment of
coastal positions.  Amphibious landings create joint operation scenarios.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/naval.py` — ship unit type, sea/river tiles
- [ ] Implement naval gunfire support (ranged bombardment from water tiles)
- [ ] Implement amphibious landing: ship → beach → infantry deployment
- [ ] Create coastal map tiles and river crossing mechanics
- [ ] Add joint scenario configs: coastal assault, river crossing, naval blockade
- [ ] Add `tests/test_naval.py`

### Acceptance Criteria

- [ ] Naval gunfire correctly bombards coastal positions (accounting for LOS)
- [ ] Amphibious landing scenario produces emergent beach-head tactics
- [ ] River crossing scenario produces emergent bridgehead tactics

### Priority

high

### Target Milestone

M21: Joint Operations""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E10.2 — Cavalry Arm as Independent Maneuver Force",
        "labels": [
            "type: epic", "priority: medium",
            "v10: multi-domain", "domain: env", "status: agent-created",
        ],
        "milestone": "M21: Joint Operations",
        "body": """\
### Version

v10

### Goal

Elevate cavalry from a battalion-level unit type (v6) to an independent
operational arm with its own echelon: cavalry brigade → cavalry corps.
Strategic reconnaissance, deep raiding, and pursuit missions.

### Child Issues (Tasks)

- [ ] Implement cavalry corps environment wrapper
- [ ] Implement reconnaissance mission: cavalry reports enemy positions within radius
- [ ] Implement raiding mission: deep penetration to destroy supply depots
- [ ] Implement pursuit mission: exploit routed infantry
- [ ] Integrate cavalry intelligence with corps-level fog of war
- [ ] Add `tests/test_cavalry_corps.py`

### Acceptance Criteria

- [ ] Cavalry reconnaissance correctly reduces fog-of-war for allied infantry
- [ ] Raiding cavalry discovers supply depots as high-value targets without explicit reward
- [ ] Corps-level win rate improves ≥ 10 % when cavalry reconnaissance is available

### Priority

medium

### Target Milestone

M21: Joint Operations""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E10.3 — Artillery Arm: Grand Battery & Counter-Battery",
        "labels": [
            "type: epic", "priority: medium",
            "v10: multi-domain", "domain: env", "status: agent-created",
        ],
        "milestone": "M22: v10 Complete",
        "body": """\
### Version

v10

### Goal

Elevate artillery to an independent arm with grand battery formation,
counter-battery fire, and siege operations against fortifications.
Artillery is commanded at corps level with battalion-level execution.

### Child Issues (Tasks)

- [ ] Implement grand battery: concentrated artillery fire reduces enemy morale rapidly
- [ ] Implement counter-battery: prioritize enemy artillery over infantry
- [ ] Implement fortification: earthwork construction (takes 10+ steps, provides cover)
- [ ] Implement siege mechanics: artillery reduces fortification HP over time
- [ ] Add artillery arm config to corps-level env

### Acceptance Criteria

- [ ] Grand battery of ≥ 6 guns breaks enemy line in < 5 game minutes (unit test)
- [ ] Counter-battery fire silences enemy guns faster than targeting infantry
- [ ] Agents learn to build fortifications proactively on defensive maps

### Priority

medium

### Target Milestone

M22: v10 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E10.4 — v10 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v10: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M22: v10 Complete",
        "body": """\
### Version

v10

### Goal

Document joint operations; publish `v10.0.0`.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v10 epics complete
- [ ] Write `docs/joint_operations_guide.md`
- [ ] Draft `CHANGELOG.md` v10 section
- [ ] Create GitHub release tag `v10.0.0`
- [ ] Seed v11 planning issues

### Acceptance Criteria

- [ ] `v10.0.0` release tag exists with release notes

### Priority

low

### Target Milestone

M22: v10 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v11 EPICS — Real-World Data Integration & Transfer ────────────────────
# ---------------------------------------------------------------------------

V11_EPICS: list[dict] = [
    {
        "title": "[EPIC] E11.1 — Historical Battle Database & Scenario Importer",
        "labels": [
            "type: epic", "priority: high",
            "v11: real-world", "domain: eval", "status: agent-created",
        ],
        "milestone": "M23: Real-World Transfer",
        "body": """\
### Version

v11

### Goal

Build a database of 50+ historical Napoleonic engagements (from Corsican
Ogre, Napoleon's Battles, and Nafziger OOBs) importable as simulation
scenarios.  Evaluate trained agents against historical outcomes.

### Child Issues (Tasks)

- [ ] Implement `data/historical/` schema: OOB, initial positions, terrain, weather
- [ ] Implement batch scenario importer from CSV/JSON battle records
- [ ] Extend historical validator to cover 50 engagements
- [ ] Create automated benchmark: agent vs. historical AI across all 50 battles
- [ ] Publish results table in `docs/historical_benchmark.md`

### Acceptance Criteria

- [ ] Importer handles ≥ 50 engagements without errors
- [ ] Agent achieves historically plausible outcome (within 1 σ of historical casualty ratio) on ≥ 60 % of battles
- [ ] Full benchmark runs in < 2 hours on a single GPU

### Priority

high

### Target Milestone

M23: Real-World Transfer""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E11.2 — GIS Terrain Import (real-world maps)",
        "labels": [
            "type: epic", "priority: high",
            "v11: real-world", "domain: env", "status: agent-created",
        ],
        "milestone": "M23: Real-World Transfer",
        "body": """\
### Version

v11

### Goal

Import real-world terrain from SRTM elevation data and OpenStreetMap
road/forest/town layers for historical battle sites (Waterloo, Austerlitz,
Borodino, Salamanca).  Agents trained on procedural terrain should
transfer to real terrain with fine-tuning.

### Child Issues (Tasks)

- [ ] Implement `data/gis/terrain_importer.py` — SRTM GeoTIFF → simulation heightmap
- [ ] Implement OSM road/forest/town layer importer
- [ ] Create real-terrain scenarios: Waterloo, Austerlitz, Borodino, Salamanca
- [ ] Fine-tune corps-level agent on real Waterloo terrain
- [ ] Measure zero-shot vs. fine-tuned transfer performance

### Acceptance Criteria

- [ ] Waterloo terrain correctly replicates La Haye Sainte ridge line
- [ ] Zero-shot transfer loses < 20 % win rate vs. procedural baseline (on real terrain)
- [ ] Fine-tuned agent recovers to within 5 % of procedural performance after < 500k fine-tuning steps

### Priority

high

### Target Milestone

M23: Real-World Transfer""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E11.3 — Expert Demonstration Collection & Imitation Learning",
        "labels": [
            "type: epic", "priority: medium",
            "v11: real-world", "domain: ml", "status: agent-created",
        ],
        "milestone": "M24: v11 Complete",
        "body": """\
### Version

v11

### Goal

Collect demonstrations from domain-expert wargamers playing the v9
interface.  Use GAIL / behaviour cloning to pre-train agents on expert
style, then fine-tune with RL.

### Child Issues (Tasks)

- [ ] Design expert recruitment pipeline (wargaming community outreach)
- [ ] Implement demonstration recorder in v9 interface (saves state-action JSON)
- [ ] Implement behaviour cloning pre-training script
- [ ] Evaluate: BC pre-trained vs. RL-from-scratch on 10 held-out scenarios
- [ ] Publish demonstration dataset (anonymised) on HuggingFace

### Acceptance Criteria

- [ ] ≥ 20 expert demonstrations collected per scenario (3 scenarios minimum)
- [ ] BC pre-trained agent achieves ≥ 60 % win rate after only 500k RL fine-tuning steps
- [ ] Dataset published with CC-BY licence

### Priority

medium

### Target Milestone

M24: v11 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E11.4 — v11 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v11: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M24: v11 Complete",
        "body": """\
### Version

v11

### Goal

Document real-world data integration; publish `v11.0.0`; submit arXiv pre-print.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v11 epics complete
- [ ] Write `docs/real_world_data_guide.md`
- [ ] Draft arXiv pre-print: "Deep RL for Napoleonic Wargaming"
- [ ] Draft `CHANGELOG.md` v11 section
- [ ] Create GitHub release tag `v11.0.0`
- [ ] Seed v12 planning issues

### Acceptance Criteria

- [ ] arXiv pre-print submitted
- [ ] `v11.0.0` release tag exists

### Priority

low

### Target Milestone

M24: v11 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v12 EPICS — Foundation Model & Open Research Platform ─────────────────
# ---------------------------------------------------------------------------

V12_EPICS: list[dict] = [
    {
        "title": "[EPIC] E12.1 — Wargames Foundation Model (WFM-1)",
        "labels": [
            "type: epic", "priority: high",
            "v12: foundation-model", "domain: ml", "status: agent-created",
        ],
        "milestone": "M25: Foundation Model",
        "body": """\
### Version

v12

### Goal

Train a large foundation model (WFM-1) on the full v11 training distribution:
procedural + real terrain, all weather, corps to battalion scale, with all
unit types.  WFM-1 is a single transformer-based policy that zero-shot
generalises to unseen scenarios and can be fine-tuned in < 10k steps.

### Motivation & Context

The long-term goal of this project is a general wargaming intelligence
that understands Napoleonic warfare at every echelon, transfers to novel
maps and OOBs, and can serve as an AI opponent, planning assistant, or
research tool.  WFM-1 is that foundation.

### Child Issues (Tasks)

- [ ] Define training distribution: all v6–v11 scenario types, all scales
- [ ] Design WFM-1 architecture: hierarchical transformer, cross-echelon attention
- [ ] Implement multi-task training loop (simultaneously optimises across all echelons)
- [ ] Implement context-conditioned fine-tuning (scenario card → policy adaptation)
- [ ] Benchmark WFM-1: zero-shot vs. specialist on 20 held-out scenarios
- [ ] Publish WFM-1 checkpoint on HuggingFace Hub

### Acceptance Criteria

- [ ] WFM-1 zero-shot win rate ≥ 55 % on unseen procedural scenarios
- [ ] Fine-tuning for 10k steps achieves ≥ 80 % of fully-trained specialist performance
- [ ] Model checkpoint published on HuggingFace Hub with CC-BY-NC-SA licence
- [ ] arXiv paper describing WFM-1 submitted

### Priority

high

### Target Milestone

M25: Foundation Model""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E12.2 — Open Research Platform & Public Benchmark",
        "labels": [
            "type: epic", "priority: high",
            "v12: platform", "domain: infra", "status: agent-created",
        ],
        "milestone": "M25: Foundation Model",
        "body": """\
### Version

v12

### Goal

Open-source the full simulation, training, and evaluation stack as a
research platform.  Publish a standardized benchmark suite (WargamesBench)
enabling reproducible comparison of wargame AI methods.

### Child Issues (Tasks)

- [ ] Finalize public API: stable `envs`, `training`, `analysis` interfaces
- [ ] Implement `benchmarks/` — 20 standardized evaluation scenarios with baselines
- [ ] Write detailed `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`
- [ ] Create documentation website (MkDocs / ReadTheDocs)
- [ ] Submit WargamesBench to NeurIPS/ICLR workshop
- [ ] Set up community Discord / mailing list

### Acceptance Criteria

- [ ] `pip install wargames-training` installs the full stack
- [ ] WargamesBench results reproducible from seed within ± 2 % win rate
- [ ] Documentation website live with tutorial, API reference, and benchmark leaderboard
- [ ] Community channels set up with ≥ 50 initial members

### Priority

high

### Target Milestone

M25: Foundation Model""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E12.3 — Modern-Era Extension (v12 Stretch Goal)",
        "labels": [
            "type: epic", "priority: low",
            "v12: modern-era", "domain: env", "status: agent-created",
        ],
        "milestone": "M26: v12 Complete",
        "body": """\
### Version

v12

### Goal

Extend the physics engine to support a second historical era (WW1 or WW2
Western Front) using the same HRL stack but with era-appropriate unit types
(machine guns, tanks, aircraft, trench systems).  Validate that WFM-1
can transfer to a new era with fine-tuning.

### Child Issues (Tasks)

- [ ] Implement `configs/eras/ww1.yaml` — WW1 unit types and weapon profiles
- [ ] Implement trench and barbed-wire terrain features
- [ ] Implement gas/artillery barrage mechanics
- [ ] Evaluate WFM-1 fine-tuning: Napoleonic → WW1 in < 50k steps
- [ ] Create two WW1 historical scenarios (Verdun, Somme section)

### Acceptance Criteria

- [ ] WW1 env passes existing env test suite with WW1 config
- [ ] WFM-1 fine-tuned on WW1 achieves > 50 % win rate within 50k steps
- [ ] Community PR process for additional eras documented

### Priority

low

### Target Milestone

M26: v12 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E12.4 — v12 Documentation, Paper & Release",
        "labels": [
            "type: epic", "priority: low",
            "v12: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M26: v12 Complete",
        "body": """\
### Version

v12

### Goal

Publish the full system paper, release `v12.0.0` as the stable long-term
release of the core project, and hand over to community maintenance.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark all v12 epics complete
- [ ] Write and submit full system paper to JMLR or TMLR
- [ ] Record tutorial video series (10 episodes × 20 minutes)
- [ ] Draft `CHANGELOG.md` v12 section
- [ ] Create GitHub release tag `v12.0.0` (LTS)
- [ ] Transfer repository ownership to community governance board

### Acceptance Criteria

- [ ] System paper submitted to a tier-1 venue
- [ ] `v12.0.0` LTS release tag exists
- [ ] Community governance in place

### Priority

low

### Target Milestone

M26: v12 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── Sprint Planning Issues (v6–v12) ───────────────────────────────────────
# ---------------------------------------------------------------------------

SPRINT_ISSUES: list[dict] = [
    # v6 sprints
    {
        "title": "[EPIC] Sprint S26 — v6 Kickoff: Terrain & Weapons",
        "labels": ["type: epic", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Sprint S26 (Weeks 45–46)

**Goal:** Terrain engine and weapons system complete; basic simulation loop running on new physics model.

### Deliverables

- [ ] `envs/sim/terrain_engine.py` — JAX LOS + elevation
- [ ] `envs/sim/weapons.py` — weapon profiles (musket, cannon)
- [ ] Procedural map generator
- [ ] `tests/test_terrain_engine.py`, `tests/test_weapons.py`

### Exit Criteria

LOS unit tests pass; weapon accuracy within ±5 % of historical data.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S27 — Formations & Morale",
        "labels": ["type: epic", "priority: high", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Sprint S27 (Weeks 47–48)

**Goal:** Formation system and morale model integrated; rout mechanics functional.

### Deliverables

- [ ] `envs/sim/formations.py` — LINE, COLUMN, SQUARE, SKIRMISH
- [ ] `envs/sim/morale.py` — morale state machine + rout
- [ ] Formation action in env action space
- [ ] `tests/test_formations.py`, `tests/test_morale.py`

### Exit Criteria

Square reliably defeats cavalry in unit tests; rout triggers at morale threshold.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S28 — Logistics, Weather & v6 Release",
        "labels": ["type: epic", "priority: medium", "v6: simulation", "domain: env", "status: agent-created"],
        "milestone": "M14: v6 Complete",
        "body": """\
### Sprint S28 (Weeks 49–50)

**Goal:** Supply, fatigue, and weather systems complete; v1 agent retrained on v6 engine; `v6.0.0` released.

### Deliverables

- [ ] `envs/sim/logistics.py` — ammo, supply, fatigue
- [ ] `envs/sim/weather.py` — weather effects
- [ ] Re-trained v1 agent achieving > 55 % win rate on v6 engine
- [ ] `docs/simulation_engine.md`
- [ ] `v6.0.0` GitHub release

### Exit Criteria

v1 agent convergent on v6 engine; all v6 unit tests passing; release tag created.
""" + ATTRIBUTION,
    },
    # v7 sprints
    {
        "title": "[EPIC] Sprint S29 — v7 Kickoff: Corps Environment",
        "labels": ["type: epic", "priority: high", "v7: operational", "domain: env", "status: agent-created"],
        "milestone": "M15: Corps Command",
        "body": """\
### Sprint S29 (Weeks 51–52)

**Goal:** Corps-level environment with road network and operational objectives running.

### Deliverables

- [ ] `envs/corps_env.py`
- [ ] Road network implementation
- [ ] Three operational objective types
- [ ] `tests/test_corps_env.py`

### Exit Criteria

6-division corps env steps without errors; road bonus correctly applied.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S30 — Supply Network & Corps League",
        "labels": ["type: epic", "priority: medium", "v7: operational", "domain: ml", "status: agent-created"],
        "milestone": "M15: Corps Command",
        "body": """\
### Sprint S30 (Weeks 53–54)

**Goal:** Strategic supply network complete; corps-level league training running.

### Deliverables

- [ ] `envs/sim/supply_network.py`
- [ ] Corps-level league training config
- [ ] W&B run with corps win rate and supply metrics

### Exit Criteria

Supply interdiction changes agent behaviour; corps main agent Elo increasing.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S31 — v7 Release",
        "labels": ["type: epic", "priority: low", "v7: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M16: v7 Complete",
        "body": """\
### Sprint S31 (Weeks 55–56)

**Goal:** v7 documentation complete; `v7.0.0` released.

### Deliverables

- [ ] `docs/corps_command_guide.md`
- [ ] ROADMAP.md v7 section marked complete
- [ ] `v7.0.0` GitHub release

### Exit Criteria

`v7.0.0` tag exists; v8 epics seeded.
""" + ATTRIBUTION,
    },
    # v8 sprints
    {
        "title": "[EPIC] Sprint S32 — v8 Kickoff: Entity Encoder",
        "labels": ["type: epic", "priority: high", "v8: architecture", "domain: ml", "status: agent-created"],
        "milestone": "M17: Transformer Policy",
        "body": """\
### Sprint S32 (Weeks 57–58)

**Goal:** Entity encoder transformer running on 4v4 scenario; attention maps visualised.

### Deliverables

- [ ] `models/entity_encoder.py`
- [ ] Integration into actor + critic
- [ ] `tests/test_entity_encoder.py`
- [ ] Attention visualization in explainability notebook

### Exit Criteria

Entity encoder handles variable N without errors; ablation `[EXP]` issue filed.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S33 — Memory Module, Scaling Study & v8 Release",
        "labels": ["type: epic", "priority: medium", "v8: architecture", "domain: ml", "status: agent-created"],
        "milestone": "M18: v8 Complete",
        "body": """\
### Sprint S33 (Weeks 59–60)

**Goal:** Recurrent memory integrated; W&B scaling sweep complete; `v8.0.0` released.

### Deliverables

- [ ] `models/recurrent_policy.py`
- [ ] W&B sweep covering ≥ 18 transformer configs
- [ ] `configs/models/transformer_{small,medium,large}.yaml`
- [ ] `v8.0.0` GitHub release

### Exit Criteria

LSTM outperforms memoryless on fog-of-war; sweep complete; release tag exists.
""" + ATTRIBUTION,
    },
    # v9 sprints
    {
        "title": "[EPIC] Sprint S34 — v9 Kickoff: Web Interface",
        "labels": ["type: epic", "priority: high", "v9: interface", "domain: viz", "status: agent-created"],
        "milestone": "M19: Decision Support",
        "body": """\
### Sprint S34 (Weeks 61–62)

**Goal:** React + WebGL wargame interface serving real-time AI via ONNX policy server.

### Deliverables

- [ ] `frontend/` React app with WebGL map renderer
- [ ] WebSocket game loop (browser ↔ sim ↔ AI)
- [ ] AI responds in < 100 ms

### Exit Criteria

Human can play a full 2v2 battle in browser against AI opponent.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S35 — COA Tool, AAR & v9 Release",
        "labels": ["type: epic", "priority: medium", "v9: coa", "v9: training", "domain: ml", "status: agent-created"],
        "milestone": "M20: v9 Complete",
        "body": """\
### Sprint S35 (Weeks 63–64)

**Goal:** COA planning tool and AAR system complete; live demo deployed; `v9.0.0` released.

### Deliverables

- [ ] COA panel in web interface (ranked list + map overlay)
- [ ] `training/human_feedback.py` + DAgger loop
- [ ] Docker Compose live demo
- [ ] `v9.0.0` GitHub release

### Exit Criteria

10 COAs generated in < 120 s; live demo accessible; DAgger improves win rate.
""" + ATTRIBUTION,
    },
    # v10 sprints
    {
        "title": "[EPIC] Sprint S36 — v10 Kickoff: Naval & Joint Operations",
        "labels": ["type: epic", "priority: high", "v10: multi-domain", "domain: env", "status: agent-created"],
        "milestone": "M21: Joint Operations",
        "body": """\
### Sprint S36 (Weeks 65–66)

**Goal:** Naval units operational; amphibious landing scenario running.

### Deliverables

- [ ] `envs/sim/naval.py`
- [ ] Coastal map tiles
- [ ] Amphibious landing scenario config
- [ ] `tests/test_naval.py`

### Exit Criteria

Naval gunfire correctly accounts for LOS; amphibious landing produces beach-head tactics.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S37 — Cavalry Corps, Grand Battery & v10 Release",
        "labels": ["type: epic", "priority: medium", "v10: multi-domain", "domain: env", "status: agent-created"],
        "milestone": "M22: v10 Complete",
        "body": """\
### Sprint S37 (Weeks 67–68)

**Goal:** Cavalry corps reconnaissance and grand battery mechanics complete; `v10.0.0` released.

### Deliverables

- [ ] Cavalry corps env wrapper
- [ ] Grand battery + counter-battery implementation
- [ ] `docs/joint_operations_guide.md`
- [ ] `v10.0.0` GitHub release

### Exit Criteria

Cavalry reduces fog-of-war for allies; grand battery breaks enemy line; release tag exists.
""" + ATTRIBUTION,
    },
    # v11 sprints
    {
        "title": "[EPIC] Sprint S38 — v11 Kickoff: Historical Database & GIS Import",
        "labels": ["type: epic", "priority: high", "v11: real-world", "domain: eval", "status: agent-created"],
        "milestone": "M23: Real-World Transfer",
        "body": """\
### Sprint S38 (Weeks 69–70)

**Goal:** 50-battle historical database imported; GIS terrain for Waterloo and Austerlitz loaded.

### Deliverables

- [ ] `data/historical/` with 50 battle records
- [ ] `data/gis/terrain_importer.py`
- [ ] Waterloo and Austerlitz real-terrain scenarios
- [ ] Benchmark: agent vs. historical AI on 50 battles

### Exit Criteria

Importer handles 50 battles without errors; agent achieves historically plausible outcome on ≥ 60 %.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S39 — Expert Demonstrations & v11 Release",
        "labels": ["type: epic", "priority: medium", "v11: real-world", "domain: ml", "status: agent-created"],
        "milestone": "M24: v11 Complete",
        "body": """\
### Sprint S39 (Weeks 71–72)

**Goal:** Expert demonstrations collected; BC pre-training pipeline complete; arXiv pre-print submitted; `v11.0.0` released.

### Deliverables

- [ ] ≥ 20 expert demos per scenario (3 scenarios)
- [ ] BC pre-training script
- [ ] arXiv pre-print submitted
- [ ] `v11.0.0` GitHub release

### Exit Criteria

BC pre-trained agent reaches 60 % win rate after 500k fine-tuning steps; pre-print submitted.
""" + ATTRIBUTION,
    },
    # v12 sprints
    {
        "title": "[EPIC] Sprint S40 — v12 Kickoff: WFM-1 Foundation Model",
        "labels": ["type: epic", "priority: high", "v12: foundation-model", "domain: ml", "status: agent-created"],
        "milestone": "M25: Foundation Model",
        "body": """\
### Sprint S40 (Weeks 73–74)

**Goal:** WFM-1 architecture defined; multi-task training loop running on full distribution.

### Deliverables

- [ ] WFM-1 architecture spec
- [ ] Multi-task training loop (all echelons simultaneously)
- [ ] Initial W&B run with cross-scenario win rates

### Exit Criteria

Multi-task loop trains without errors; zero-shot evaluation started.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S41 — Open Platform, Paper & v12 Release",
        "labels": ["type: epic", "priority: medium", "v12: platform", "v12: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M26: v12 Complete",
        "body": """\
### Sprint S41 (Weeks 75–76)

**Goal:** WFM-1 checkpoint published; WargamesBench standardized; system paper submitted; `v12.0.0` LTS released.

### Deliverables

- [ ] WFM-1 checkpoint on HuggingFace Hub
- [ ] `benchmarks/` with 20 standardized scenarios
- [ ] `pip install wargames-training` working
- [ ] Documentation website live
- [ ] System paper submitted
- [ ] `v12.0.0` LTS GitHub release

### Exit Criteria

`v12.0.0` tag exists; WFM-1 on HuggingFace; paper submitted; community channels active.
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# Flat list of all issues to create
# ---------------------------------------------------------------------------

ALL_ISSUES: list[dict] = (
    V6_EPICS
    + V6_TASKS
    + V7_EPICS
    + V8_EPICS
    + V9_EPICS
    + V10_EPICS
    + V11_EPICS
    + V12_EPICS
    + SPRINT_ISSUES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_milestone_by_title(repo, title: str) -> object | None:
    """Return the milestone whose title matches (open or closed), or None."""
    for ms in repo.get_milestones(state="all"):
        if ms.title == title:
            return ms
    return None


def existing_titles(repo) -> set[str]:
    """Return a normalised (lower-stripped) set of all non-PR issue titles (open + closed).

    Fetching all titles once upfront avoids one API round-trip per issue
    during the creation loop.
    """
    titles: set[str] = set()
    for item in repo.get_issues(state="all"):
        # The issues API can return pull requests; filter them out.
        if getattr(item, "pull_request", None) is None:
            titles.add(item.title.strip().lower())
    return titles


def create_issue(repo, issue_def: dict, known: set[str], *, dry_run: bool) -> str:
    """Create a single GitHub issue. Returns 'created', 'exists', or 'skipped'."""
    title = issue_def["title"]

    if title.strip().lower() in known:
        print(f"  ↩ exists: {title[:80]}")
        return "exists"

    if dry_run:
        print(f"  [dry-run] Would create: {title[:80]}")
        return "skipped"

    # Resolve labels; auto-create any that are missing so seeded issues get
    # consistent metadata even if setup_labels_and_milestones.py is outdated.
    label_objects = []
    for label_name in issue_def.get("labels", []):
        try:
            label_objects.append(repo.get_label(label_name))
        except Exception:
            try:
                auto_label = repo.create_label(
                    name=label_name,
                    color=_label_color(label_name),
                    description=(
                        "Auto-created by v6–v12 roadmap seeding script; "
                        "consider updating setup_labels_and_milestones.py."
                    ),
                )
                label_objects.append(auto_label)
                print(f"    [+] Auto-created missing label: {label_name}")
            except Exception:
                print(f"    [warn] Label not found and could not be created: {label_name} — skipping")

    # Resolve milestone; auto-create if missing so that long-range roadmap
    # issues stay properly grouped.
    milestone_obj = None
    if issue_def.get("milestone"):
        milestone_title = issue_def["milestone"]
        milestone_obj = get_milestone_by_title(repo, milestone_title)
        if milestone_obj is None:
            try:
                milestone_obj = repo.create_milestone(title=milestone_title)
                print(f"    [+] Auto-created missing milestone: {milestone_title}")
            except Exception:
                print(
                    f"    [warn] Milestone not found and could not be created: {milestone_title} — issue will have no milestone"
                )

    try:
        kwargs: dict = {
            "title": title,
            "body": issue_def.get("body", ""),
            "labels": label_objects,
        }
        if milestone_obj is not None:
            kwargs["milestone"] = milestone_obj

        repo.create_issue(**kwargs)
        print(f"  + created: {title[:80]}")
        return "created"
    except Exception as exc:
        print(f"  ! FAILED: {title[:80]}\n    {exc}", file=sys.stderr)
        return "skipped"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPO_NAME")
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"

    if not token or not repo_name:
        print(
            "ERROR: GITHUB_TOKEN and REPO_NAME environment variables must be set.",
            file=sys.stderr,
        )
        return 1

    try:
        from github import Auth, Github
    except ImportError:
        print("ERROR: PyGithub is not installed.  Run: pip install PyGithub", file=sys.stderr)
        return 1

    auth = Auth.Token(token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    print(f"Connected to {repo.full_name}{'  [DRY RUN]' if dry_run else ''}")
    print(f"Total issues to seed: {len(ALL_ISSUES)}")

    # Fetch all existing issue titles once to avoid N×2 API calls in the loop.
    print("\nFetching existing issue titles...")
    known = existing_titles(repo)
    print(f"  {len(known)} existing issues found.")

    # Group by section for readable output
    sections = [
        ("── v6 Epics", V6_EPICS),
        ("── v6 Tasks", V6_TASKS),
        ("── v7 Epics", V7_EPICS),
        ("── v8 Epics", V8_EPICS),
        ("── v9 Epics", V9_EPICS),
        ("── v10 Epics", V10_EPICS),
        ("── v11 Epics", V11_EPICS),
        ("── v12 Epics", V12_EPICS),
        ("── Sprint Planning Issues", SPRINT_ISSUES),
    ]

    total_stats: dict[str, int] = {"created": 0, "exists": 0, "skipped": 0}

    for section_name, issues in sections:
        print(f"\n{section_name} ({len(issues)} issues)")
        for issue_def in issues:
            result = create_issue(repo, issue_def, known, dry_run=dry_run)
            total_stats[result] = total_stats.get(result, 0) + 1

    print(
        f"\n{'[dry-run] ' if dry_run else ''}✅ Seeding complete: "
        f"{total_stats.get('created', 0)} created, "
        f"{total_stats.get('exists', 0)} already exist, "
        f"{total_stats.get('skipped', 0)} skipped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
