# Battalion Doctrine Realignment

## Problem

The current `BattalionEnv` continuous `[move, rotate, fire]` action space models
company-level free movement while being labeled battalion-level. This is
historically inverted: Napoleonic battalions were highly constrained drill-execution
machines, not free agents. The actual free-moving primitive in this period was the
light company (voltigeurs / Jäger). Meanwhile, `BrigadeEnv` already has the
doctrinally correct MultiDiscrete action space for issuing battalion-level orders.

## Target Architecture

```
CorpsEnv / DivisionEnv      ← strategic positioning, resource allocation
    └── BrigadeEnv          ← RL agent; issues orders to battalions
            └── BattalionEnv← FSM executing drill orders (formation transitions)
                    └── CompanyEnv  ← continuous free-mover (skirmishers/light inf.)
```

## Phases

### Phase 1 — FormationConstraintWrapper (v1-compatible, non-breaking)

Add a `gym.Wrapper` that gates the raw `[move, rotate, fire]` continuous action
through the unit's active formation state before passing it to `BattalionEnv.step()`.

Formation constraints (calibrated against `FORMATION_ATTRIBUTES` in `envs/sim/formations.py`):
- **LINE**: rotation capped at ±5°/step, no rearward movement
- **COLUMN**: full movement speed, fire intensity halved
- **SQUARE**: movement ≈ 0, rotation ≈ 0, no advance
- **SKIRMISH**: full movement freedom, lower fire intensity

Implementation notes:
- New file: `envs/formation_constraint_wrapper.py`
- Reads `obs[formation_idx]` (available when `enable_formations=True` in 19-D obs)
- No changes to `BattalionEnv` itself — fully backward-compatible
- ~20 tests in `tests/test_formation_constraints.py`

### Phase 2 — Doctrine Vocabulary Fix (v1-compatible, minor action-space change)

Update `envs/options.py` to reflect historical Napoleonic drill vocabulary:

| Old | New | Notes |
|---|---|---|
| `ADVANCE_SECTOR` (0) | `ADVANCE_IN_COLUMN` (0) | Battalions advance in column |
| `DEFEND_POSITION` (1) | `HOLD_IN_LINE` (1) | Defensive fire from line |
| *(missing)* | `FORM_SQUARE` (6) | **Critical omission** — cavalry defence |
| *(missing)* | `DEPLOY_SKIRMISHERS` (7) | Detach voltigeur screen |

Total macro-actions: 6 → 8. Update `SMDPWrapper` option count and `BrigadeEnv`
MultiDiscrete dimension to match.

Add docstring to `BattalionEnv` clarifying: *"This environment models company-level
execution. The doctrinally correct battalion decision layer is `BrigadeEnv`."*

### Phase 3 — Learned Drill Sub-Policies (post-v1 milestone 1)

Replace the scripted lambda policies in `options.py` with learned low-level PPO
sub-policies, each:
- Constrained by Phase 1's `FormationConstraintWrapper`
- BC-initialized from current scripted lambda
- With a dedicated termination condition (reach sector / cavalry within 150m / etc.)

New class: `DrillPolicyLibrary` in `envs/options.py` — loads pre-trained sub-policies
by `MacroAction` index.

### Phase 4 — CompanyEnv (post-v1 milestone 2)

Create `envs/company_env.py` — the genuine continuous free-mover:
- ~80 men (vs ~600 for battalion)
- Formation locked to SKIRMISH (or COLUMN for movement only)
- Shorter engagement range, higher per-man lethality
- Observation adds: parent battalion bearing + screen objective vector
- Backed by new `envs/sim/company.py` (analogous to `envs/sim/battalion.py`)

`BattalionEnv` wraps 1–2 `CompanyEnv` instances for its skirmisher screen.

### Phase 5 — True BattalionEnv State Machine (post-v1 milestone 3)

Replace `BattalionEnv` continuous `Box(3)` with `Discrete(8)` (the Phase 2 drill
vocabulary). Internal FSM: `IDLE → EXECUTING_ORDER → COMPLETED/ROUTING`. Each
discrete order triggers a Phase 3 learned sub-policy, reporting completion back to
`BrigadeEnv` — this is exactly `SMDPWrapper` semantics internalized into the battalion.

`BrigadeEnv` becomes the primary RL training target. Current `BattalionEnv` deprecated
or renamed `CompanyGroupEnv`.

## Acceptance Criteria

- [ ] Phase 1: `FormationConstraintWrapper` passes all formation constraint tests; existing `BattalionEnv` tests unchanged
- [ ] Phase 2: `options.py` has 8 macro-actions; `BrigadeEnv` MultiDiscrete updated; BattalionEnv docstring added
- [ ] Phase 3: Each `MacroAction` has a trained sub-policy; `DrillPolicyLibrary` loads all 8
- [ ] Phase 4: `CompanyEnv` registered in `envs/__init__.py`; BattalionEnv wraps skirmisher screen
- [ ] Phase 5: `BattalionEnv` action space is `Discrete(8)`; `BrigadeEnv` is the primary RL agent

## Files Affected

| File | Change |
|---|---|
| `envs/formation_constraint_wrapper.py` | NEW (Phase 1) |
| `envs/company_env.py` | NEW (Phase 4) |
| `envs/sim/company.py` | NEW (Phase 4) |
| `envs/options.py` | Rename + add macro-actions (Phase 2) |
| `envs/smdp_wrapper.py` | Update option count (Phase 2) |
| `envs/brigade_env.py` | Update MultiDiscrete dim (Phase 2) |
| `envs/battalion_env.py` | Add docstring (Phase 2); action space refactor (Phase 5) |
| `envs/__init__.py` | Export new types |
| `tests/test_formation_constraints.py` | NEW (Phase 1) |
| `tests/test_company_env.py` | NEW (Phase 4) |
