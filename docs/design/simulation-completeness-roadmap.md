# Simulation Completeness & Cross-Version Integration Roadmap

## Context

Work from v9–v11 (Human-in-the-Loop, Multi-Domain, Real-World Data) has been
partially implemented out of roadmap order as isolated epics. The following
components are already built and tested:

| Epic | Module | Status |
|---|---|---|
| E9.2 | `analysis/coa_generator.py` — CorpsCOAGenerator, COAScore, COAExplanation | ✅ Done |
| E9.3 | `training/human_feedback.py` — DAggerTrainer, GAILDiscriminator, AARAnnotator | ✅ Done |
| E10.1 | `envs/sim/naval.py` — ShipType, CoastalMap, AmphibiousLanding, RiverCrossing | ✅ Done |
| E10.2 | `envs/sim/cavalry_corps.py` + `envs/cavalry_corps_env.py` | ✅ Done |
| E10.3 | `envs/sim/artillery_corps.py` + `envs/artillery_corps_env.py` | ✅ Done |
| E11.2 | `data/gis/terrain_importer.py` — SRTMImporter, OSMLayerImporter, GISTerrainBuilder | ✅ Done |

However, the foundation versions that make these components meaningful
(v6: Physics Simulation, v7: Operational Scale, v8: Transformer Policy) are
still listed as "Planned" in the roadmap. This epic tracks completing those
foundations and properly integrating the out-of-order work.

---

## Epic 1: Complete v6 — Physics-Accurate Simulation Foundation

**Goal:** The abstract 2D simulation has significant gaps that weaken the
historical validity of all trained agents. This epic closes those gaps.

### E6 Remaining Work

**E6.1 — Terrain Engine Integration** *(partial — GIS terrain exists, sim integration incomplete)*
- Connect `data/gis/terrain_importer.py` output directly to `envs/sim/terrain_engine.py`
- Ensure `TerrainEngine` tile types drive `BattalionEnv` movement cost and LOS
- Add slope-based movement penalty (currently flat speed modifier only)
- Validate: agents trained on GIS terrain generalise to procedural terrain and vice versa

**E6.2 — Weapon System Realism**
- `envs/sim/weapons.py` currently has simplified range/accuracy; calibrate against
  historical effective ranges (musket ~50m, rifle ~200m, artillery ~800m)
- Add reload cycle mechanics (musket ~20s; already partially in sim — needs exposure
  in observation space as `reload_fraction` dim)
- Add canister/grape shot range band for artillery at <200m

**E6.3 — Formation Mechanics Coupling** *(depends on Battalion Doctrine Realignment E-link)*
- The `FORMATION_ATTRIBUTES` in `formations.py` are not yet fully driving combat
  outcomes in `combat.py`; `cavalry_charge_modifier` is unused in most combat paths
- Fix: `CombatResult` must read formation modifiers from both attacker and defender

**E6.4 — Morale Cascade & Rout Propagation**
- `envs/sim/morale.py` handles individual unit morale; brigade-level morale cascade
  (a routing battalion demoralizing adjacent units) is not implemented
- Add `MoraleNetwork` in `morale.py`: adjacent units within 150m receive morale
  penalty when a friendly unit routes
- Add cascade to `BrigadeEnv.step()` morale update loop

**E6.5 — Logistics / Supply Consumption Realism**
- `envs/sim/logistics.py` models ammunition and ration supply; consumption rates
  need historical calibration (60 rounds per man per engagement, 3 days rations)
- Add `SupplyDepot` abstraction: a fixed-position tile that replenishes nearby units
  during IDLE steps (simulates wagon train resupply)
- Expose supply level in BattalionEnv observation (currently optional dim — make default)

**E6.6 — Weather System Integration**
- `envs/sim/weather.py` exists; `BattalionEnv` has optional +2 weather dims
- Weather currently does not affect artillery effectiveness (rain should reduce
  fire rate ~50%, heavy rain ~80%)
- Connect `WeatherState` to artillery fire calculation in `artillery_corps.py`

---

## Epic 2: Complete v7 — Operational-Scale Corps Command

**Goal:** The existing `CorpsEnv` / `ArtilleryCorpsEnv` / `CavalryCorpsEnv`
provide the mechanistic foundation. This epic integrates them into a coherent
operational command environment matching the Black (NPS 2024) corps architecture.

### E7 Remaining Work

**E7.1 — Road Network + March Doctrine**
- `envs/sim/road_network.py` exists; `RoadNetwork` is not yet integrated into
  `CorpsEnv` movement — units move at flat speed regardless of road vs off-road
- Add: units on road tiles use `COLUMN` formation speed bonus automatically
- Add: road march observation feature (is_on_road boolean in corps obs)

**E7.2 — Operational Objectives**
- `CorpsEnv` currently has no strategic objective layer; win/loss is based on
  force ratio only
- Add `ObjectivePoint` tiles (capture by occupying for N steps)
- Add `supply_line_control` feature: does the corps control a road connecting
  to the map edge (supply line intact)?

**E7.3 — Combined Arms Integration**
- `ArtilleryCorpsEnv` and `CavalryCorpsEnv` are standalone; `CorpsEnv` has no
  mechanism to command attached artillery / cavalry sub-units
- Add `AttachedArmsConfig` to `CorpsEnv`: optional artillery battery + cavalry
  squadron with restricted action slots
- This is the integration point for E10.1–E10.3 components

**E7.4 — Naval-Land Coordination**
- `envs/sim/naval.py` (E10.1) is standalone; it is not integrated with `CorpsEnv`
- Add `NavalFireSupportEnv` extending `CorpsEnv` with a `CoastalMap` and a
  gunboat that can be assigned BOMBARD mission each step

---

## Epic 3: Complete v8 — Transformer Policy & Architecture

**Goal:** Replace fixed-size concatenation observations with variable-length
entity-token sequences processed by multi-head self-attention.

### E8 Work

**E8.1 — Entity Encoder**
- New module: `models/entity_encoder.py`
- Each unit (blue or red) becomes a token: `[x, y, cos_θ, sin_θ, strength, morale, formation]`
- Self-attention over all visible tokens; output is pooled per-agent
- Compatible with existing `BattalionEnv` and `CorpsEnv` observation structures

**E8.2 — Recurrent Memory for Fog-of-War**
- Add GRU memory module to `models/` wrapping entity encoder output
- Required for `CorpsEnv` with `fog_of_war=True` (units only visible within `comm_radius`)
- SB3-compatible via `RecurrentPPO` from `sb3-contrib`

**E8.3 — Scaling Study**
- Systematic benchmark: MLP (current) vs Entity Encoder vs Entity+GRU
- Evaluation across 1v1, 4v4, corps scenarios using existing `TransferBenchmark`
- W&B sweep logged as `[EXP]` issues

---

## Epic 4: Complete v9 — Human-in-the-Loop Integration

**Goal:** The AAR and COA components (E9.2, E9.3) are built but not yet wired
into the web interface.

### E9 Remaining Work

**E9.1 — Web Interface + AI Policy Server** *(not yet started)*
- `server/` directory exists but has no battle replay endpoint
- Add: `/api/replay/<run_id>` streaming the step-by-step battle state
- Connect `envs/rendering/web_renderer.py` output to a WebSocket endpoint
- The `frontend/` React app should consume this stream for live replay

**E9.4 — v9 Documentation & Release** *(not yet started)*
- Human feedback workflow guide
- COA tool user manual
- API reference for `/corps/coas`, `/corps/coas/modify`, `/corps/coas/explain`

---

## Epic 5: Complete v11 — Real-World Data & Transfer

**Goal:** The GIS terrain importer (E11.2) is built; the historical battle
database and transfer benchmark need completion.

### E11 Remaining Work

**E11.1 — Historical Battle Database & Scenario Importer** *(partial)*
- `envs/scenarios/historical.py` has 4 GIS scenarios (Waterloo/Borodino/Austerlitz/Leipzig)
- Remaining: add 4 more battle sites; validate agent performance on each as
  transfer benchmark targets
- `training/transfer_benchmark.py` (`TransferBenchmark`) is built — needs
  automated CI run comparing GIS-trained vs random-terrain-trained agents

---

## Integration Priority Order

The recommended completion sequence respects dependencies:

1. **E6.3 + E6.4** (formation coupling + morale cascade) — unblocks battalion doctrine realignment
2. **E6.1 + E6.6** (terrain + weather integration) — unblocks historical fidelity
3. **E7.1 + E7.2** (road network + objectives) — unblocks meaningful corps training
4. **E7.3 + E7.4** (combined arms + naval) — integrates E10.1–E10.3
5. **E8.1** (entity encoder) — can start in parallel with E7
6. **E9.1** (web interface) — can start in parallel; gates E9.4
7. **E11.1** (historical database completion) — final validation layer

## Acceptance Criteria

- [ ] All v6 epics complete: formation coupling, morale cascade, logistics, weather→artillery
- [ ] `CorpsEnv` supports road march bonus, objective points, and supply line control
- [ ] `ArtilleryCorpsEnv`/`CavalryCorpsEnv` are composable inside `CorpsEnv` via `AttachedArmsConfig`
- [ ] `NavalFireSupportEnv` integrates naval gunfire into a corps episode
- [ ] Entity encoder in `models/` with SB3-compatible wrapper; scaling benchmark run and logged
- [ ] `/api/replay/<run_id>` endpoint live; frontend battle replay working
- [ ] Transfer benchmark runs in CI across 4 GIS battle sites

## Priority
high
