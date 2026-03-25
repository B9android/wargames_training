# SPDX-License-Identifier: MIT
"""Seed v1 roadmap issues: creates all E1.2–E1.10 epics and key feature tasks.

Idempotent — skips any issue whose title already exists.
Run via the seed-v1-issues GitHub Actions workflow or locally with:
    GITHUB_TOKEN=... REPO_NAME=owner/repo python seed_v1_issues.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Issue definitions
# ---------------------------------------------------------------------------

ATTRIBUTION = (
    "\n\n---\n"
    "> 🤖 *Strategist Forge* created this issue automatically as part of v1 roadmap seeding.\n"
    f"> Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n"
    "> Label `status: agent-created` was applied automatically.\n"
)

EPICS: list[dict] = [
    {
        "title": "[EPIC] E1.2 — Core Simulation Engine",
        "labels": ["type: epic", "priority: high", "v1: simulation", "domain: sim", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
### Version

v1

### Goal

A complete, self-contained simulation engine: robust combat resolution,
battalion physics with morale and formation effects, terrain interaction,
and a thin Python API that the Gymnasium environment layer can call.

### Motivation & Context

`battalion.py` has basic movement and a stub fire system but no damage
accumulation, no morale, and no terrain coupling.  `combat.py` and
`terrain.py` are one-line stubs.  Nothing can be trained until the sim
produces meaningful episode outcomes.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/combat.py` — fire damage accumulation, casualty mechanics, morale check
- [ ] Extend `envs/sim/battalion.py` — morale state, routing threshold, formation cohesion bonus
- [ ] Implement `envs/sim/terrain.py` — flat-map scaffold with cover/elevation hooks
- [ ] Add unit tests for combat resolution (damage falloff, range checks, morale)
- [ ] Add unit tests for battalion movement edge cases (boundary clamping, rotation limits)
- [ ] Create a headless `scenario_runner.py` helper for quick integration checks

### Acceptance Criteria

- [ ] A battle between two battalions resolves to a winner (one side routes or is destroyed) in ≤ 500 sim steps
- [ ] Combat damage scales correctly with range and firing arc
- [ ] Morale drops under fire and triggers routing below threshold
- [ ] All sim unit tests pass (`pytest tests/test_sim.py`)
- [ ] No Gymnasium or RL code required to run a sim episode

### Priority

high

### Target Milestone

M1: 1v1 Competence""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.3 — Gymnasium Environment (1v1)",
        "labels": ["type: epic", "priority: high", "v1: environment", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
### Version

v1

### Goal

A fully compliant `gymnasium.Env` subclass wrapping the sim engine for
1v1 battalion combat.  Observation and action spaces are clearly defined,
observations are normalised, and the environment passes `gym.utils.env_checker`.

### Motivation & Context

`envs/battalion_env.py` is a one-line stub.  Without a working Gymnasium
environment, no RL training can begin.

### Child Issues (Tasks)

- [ ] Implement `envs/battalion_env.py` — `reset()`, `step()`, `render()`, `close()`
- [ ] Define observation space: normalised position (x/y by map dims), angle as (cos θ, sin θ), strength ratio, relative enemy state
- [ ] Define action space: continuous (Δmove, Δrotate, fire_flag) or discrete action set
- [ ] Implement episode termination conditions (routing, time limit, out-of-bounds)
- [ ] Add `gymnasium.utils.env_checker` call in tests
- [ ] Add `tests/test_battalion_env.py` with reset/step/termination tests

### Acceptance Criteria

- [ ] `BattalionEnv()` passes `gymnasium.utils.env_checker`
- [ ] Observation vectors are in `[-1, 1]` or `[0, 1]` ranges
- [ ] Angles are encoded as `(cos θ, sin θ)` pairs — never raw radians
- [ ] Episodes terminate deterministically (no infinite loops)
- [ ] Environment can be seeded for reproducibility

### Priority

high

### Target Milestone

M1: 1v1 Competence""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.4 — Baseline Training Loop (PPO + scripted opponent)",
        "labels": ["type: epic", "priority: high", "v1: training", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
### Version

v1

### Goal

A complete, reproducible training pipeline: PPO agent trains against a
scripted opponent, metrics are logged to W&B, and the agent demonstrably
beats the scripted baseline after a reasonable training budget.

### Motivation & Context

`training/train.py`, `models/mlp_policy.py`, and `configs/default.yaml`
are all stubs.  This epic delivers the first real training run.

### Child Issues (Tasks)

- [ ] Implement `training/train.py` — SB3 PPO loop, env instantiation, W&B run init, checkpoint saving
- [ ] Implement `models/mlp_policy.py` — feedforward MLP compatible with SB3 custom policy API
- [ ] Add a scripted opponent (`envs/opponents/scripted.py`) — advance and fire, simple heuristic
- [ ] Fill in `configs/default.yaml` — learning rate, batch size, n_steps, gamma, ent_coef, total_timesteps
- [ ] Create `configs/experiment_1.yaml` — first real experiment overrides
- [ ] Add `training/evaluate.py` — load checkpoint, run N evaluation episodes, log win rate
- [ ] W&B run tagged with config dict (hyperparams + git SHA)

### Acceptance Criteria

- [ ] `python training/train.py` runs to completion without errors
- [ ] W&B run appears with reward curves and episode length metrics
- [ ] Trained agent beats scripted opponent with ≥ 55 % win rate after training budget
- [ ] Model checkpoint saved to `checkpoints/<run_id>/`
- [ ] `python training/evaluate.py --checkpoint <path>` prints win rate

### Priority

high

### Target Milestone

M1: 1v1 Competence""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.5 — Terrain & Environmental Randomization",
        "labels": ["type: epic", "priority: medium", "v1: terrain", "domain: sim", "status: agent-created"],
        "milestone": "M2: Terrain & Generalization",
        "body": """\
### Version

v1

### Goal

Add terrain features to the simulation and randomise the initial episode
configuration so the trained policy generalises beyond a fixed layout.

### Motivation & Context

A policy trained on a single flat map will overfit and fail to generalise.
This epic introduces environmental diversity via procedural terrain and
parameter randomisation.

### Child Issues (Tasks)

- [ ] Implement `envs/sim/terrain.py` — hills (movement penalty), forests (cover/visibility), open ground
- [ ] Add terrain effects to combat resolution (cover modifier on damage)
- [ ] Add terrain effects to movement (speed penalty on hills)
- [ ] Add map randomisation to `BattalionEnv.reset()` — random start positions, random terrain seeds
- [ ] Add parameter randomisation — strength variance, map size, time limit
- [ ] Add `tests/test_terrain.py`

### Acceptance Criteria

- [ ] `BattalionEnv.reset(seed=...)` produces different terrain each episode
- [ ] Hill tiles reduce movement speed by a configurable factor
- [ ] Forest tiles apply a cover modifier to incoming fire damage
- [ ] Policy trained with terrain randomisation generalises to held-out terrain seeds

### Priority

medium

### Target Milestone

M2: Terrain & Generalization""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.6 — Reward Shaping & Curriculum Design",
        "labels": ["type: epic", "priority: high", "v1: rewards", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
### Version

v1

### Goal

Design and implement a shaped reward function that accelerates learning;
implement a curriculum with 3–5 difficulty levels progressing from trivial
to full 1v1 combat.

### Motivation & Context

A sparse win/loss reward makes PPO training extremely slow.  Shaped
components (damage dealt, enemy strength reduction, survival time) and a
curriculum (scripted opponent strength ramp) are standard practice.

### Child Issues (Tasks)

- [ ] Define reward components: Δenemy_strength, own_survival_bonus, win_bonus, time_penalty
- [ ] Implement `envs/reward.py` with configurable component weights
- [ ] Implement curriculum levels L1–L5 (stationary target → maneuvering opponent → full combat)
- [ ] Add `curriculum_level` parameter to `BattalionEnv` config
- [ ] Log per-component reward breakdown to W&B
- [ ] Ablation experiment: shaped vs sparse reward ([EXP] issue)

### Acceptance Criteria

- [ ] Reward components are independently configurable via `configs/default.yaml`
- [ ] Each curriculum level passes a minimum competence threshold before advancing
- [ ] W&B reward breakdown shows learning signal from each component
- [ ] Agent completes L1 (stationary target) within 500 k timesteps

### Priority

high

### Target Milestone

M1: 1v1 Competence""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.7 — Self-Play Implementation",
        "labels": ["type: epic", "priority: medium", "v1: self-play", "domain: ml", "status: agent-created"],
        "milestone": "M3: Self-Play Baseline",
        "body": """\
### Version

v1

### Goal

Implement a self-play training loop where the agent trains against a pool
of frozen snapshots of itself, preventing policy collapse and promoting
robust strategies.

### Motivation & Context

Once the agent reliably beats the scripted baseline, self-play is the
primary mechanism for continued improvement.  Policy snapshots act as
"historical opponents" preventing catastrophic forgetting.

### Child Issues (Tasks)

- [ ] Implement `training/self_play.py` — snapshot pool management, opponent sampling
- [ ] Implement policy versioning and snapshot serialisation
- [ ] Add `OpponentPool` class — stores N latest snapshots, samples uniformly
- [ ] Integrate self-play opponent into `BattalionEnv` as the "red" side
- [ ] Add win-rate-vs-pool metric to W&B logging
- [ ] Add `configs/self_play.yaml` with pool size, snapshot interval

### Acceptance Criteria

- [ ] Self-play training runs for ≥ 1 M timesteps without policy collapse
- [ ] Win rate vs latest snapshot is tracked and logged
- [ ] Policy pool is serialised and loadable across training restarts
- [ ] Self-play agent beats scripted baseline with ≥ 65 % win rate

### Priority

medium

### Target Milestone

M3: Self-Play Baseline""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.8 — Evaluation Framework & Elo Tracking",
        "labels": ["type: epic", "priority: medium", "v1: evaluation", "domain: ml", "status: agent-created"],
        "milestone": "M3: Self-Play Baseline",
        "body": """\
### Version

v1

### Goal

Systematic, reproducible evaluation of trained policies with Elo rating
against a fixed baseline ladder, enabling objective comparison across runs.

### Motivation & Context

Without a structured evaluation protocol, it's impossible to compare
runs or track long-term improvement.  Elo provides a single scalar
progress metric that's easy to monitor.

### Child Issues (Tasks)

- [ ] Implement `training/evaluate.py` — load checkpoint, run N episodes, return win/loss/draw
- [ ] Implement `training/elo.py` — Elo update rule, K-factor schedule, rating persistence
- [ ] Create baseline ladder: scripted L1–L5, random policy, previous self-play snapshots
- [ ] Integrate Elo updates into W&B run logging
- [ ] Add evaluation callback to training loop (eval every N timesteps)
- [ ] Add `tests/test_evaluation.py`

### Acceptance Criteria

- [ ] `python training/evaluate.py --checkpoint <path> --opponent scripted_l3` prints win rate and Elo delta
- [ ] Elo ratings are persisted to `checkpoints/elo_registry.json`
- [ ] W&B run shows Elo curve over training time
- [ ] Evaluation is reproducible (same seed → same result)

### Priority

medium

### Target Milestone

M3: Self-Play Baseline""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.9 — Visualization & Replay System",
        "labels": ["type: epic", "priority: low", "v1: visualization", "domain: viz", "status: agent-created"],
        "milestone": "M4: v1 Complete",
        "body": """\
### Version

v1

### Goal

A pygame-based renderer that can display live battles and replay saved
episodes, enabling qualitative analysis of learned behaviours.

### Motivation & Context

`envs/rendering/renderer.py` is a stub.  Without visualisation, debugging
emergent tactical behaviour is nearly impossible.

### Child Issues (Tasks)

- [ ] Implement `envs/rendering/renderer.py` — pygame window, battalion sprites, terrain overlay
- [ ] Draw battalions as line segments oriented by angle θ
- [ ] Draw strength bar and morale indicator per battalion
- [ ] Implement episode recording to JSON (`envs/rendering/recorder.py`)
- [ ] Implement replay from JSON recording
- [ ] Add `--render` flag to `training/evaluate.py`

### Acceptance Criteria

- [ ] `python training/evaluate.py --checkpoint <path> --render` opens a pygame window and displays the battle
- [ ] Battalion orientation matches simulation angle
- [ ] Recording saves full episode trajectory to `replays/<id>.json`
- [ ] Replay from JSON produces identical visuals

### Priority

low

### Target Milestone

M4: v1 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E1.10 — v1 Documentation & Release",
        "labels": ["type: epic", "priority: low", "v1: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M4: v1 Complete",
        "body": """\
### Version

v1

### Goal

Complete end-user documentation, a reproducible quickstart guide, and a
formal v1.0.0 GitHub release with trained checkpoint attached.

### Motivation & Context

The project will be shared externally and needs clear setup instructions,
training guide, and experiment reproducibility notes before v1 is cut.

### Child Issues (Tasks)

- [ ] Complete `README.md` — setup, quickstart training run, evaluation guide
- [ ] Write `docs/TRAINING_GUIDE.md` — hyperparameters, tips, W&B integration
- [ ] Write `docs/ENVIRONMENT_SPEC.md` — observation/action space reference
- [ ] Add `CONTRIBUTING.md` — coding conventions, PR checklist
- [ ] Tag v1.0.0 release with CHANGELOG entry and trained baseline checkpoint
- [ ] Update `docs/ROADMAP.md` to mark v1 complete and outline v2

### Acceptance Criteria

- [ ] A new contributor can `pip install -r requirements.txt && python training/train.py` successfully using only the README
- [ ] v1.0.0 GitHub Release exists with trained checkpoint and CHANGELOG
- [ ] All docs spell-check and link-check pass

### Priority

low

### Target Milestone

M4: v1 Complete""" + ATTRIBUTION,
    },
]

FEATURES: list[dict] = [
    {
        "title": "[FEAT] Implement combat.py — damage accumulation and morale mechanics",
        "labels": ["type: feature", "priority: high", "v1: simulation", "domain: sim", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

`envs/sim/combat.py` is a one-line stub.  `Battalion.fire_at()` in
`battalion.py` computes a damage value but does not apply it.  There is
no morale system, so battles never resolve.

## Task

Implement `envs/sim/combat.py` with:

1. `apply_fire(attacker, defender)` — applies damage from `attacker.fire_at(defender)` to `defender.strength`
2. `check_morale(battalion, threshold)` — returns `True` if battalion routes (strength below threshold)
3. `resolve_step(blue, red)` — single sim tick: both sides fire simultaneously, damage applied, morale checked

All functions must be pure (no side-effects beyond modifying the passed
battalion objects).

## Acceptance Criteria

- [ ] `apply_fire` reduces defender strength by the value returned from `fire_at`
- [ ] `check_morale` returns `True` when `strength <= threshold`
- [ ] `resolve_step` applies fire from both sides simultaneously (simultaneous fire model)
- [ ] Routing battalion is marked with a `routed: bool` flag on the `Battalion` dataclass
- [ ] `tests/test_combat.py` covers: damage application, morale threshold, simultaneous fire

## Notes

- Part of epic: E1.2 — Core Simulation Engine
- Keep functions importable standalone — no Gymnasium dependency
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement battalion_env.py — Gymnasium 1v1 environment",
        "labels": ["type: feature", "priority: high", "v1: environment", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

`envs/battalion_env.py` is a one-line stub.  No RL training can occur
until a compliant Gymnasium environment exists.

## Task

Implement `BattalionEnv(gymnasium.Env)` with:

1. **Observation space** — `Box` containing (per battalion):
   - Normalised position: `x / map_width`, `y / map_height` ∈ `[0, 1]`
   - Angle encoded as `(cos θ, sin θ)` ∈ `[-1, 1]`
   - Strength ratio: `strength / initial_strength` ∈ `[0, 1]`
   - Enemy relative bearing and distance (normalised)
2. **Action space** — `Box` with three continuous dims: `(Δmove [-1,1], Δrotate [-1,1], fire [0,1])`
3. `reset(seed, options)` — random initial positions within map bounds, return obs + info
4. `step(action)` — advance sim one tick, compute reward, check termination
5. `render()` — stub (calls renderer if available)
6. Episode terminates on: routing, strength ≤ 0, or `max_steps`

## Acceptance Criteria

- [ ] `gymnasium.utils.env_checker(BattalionEnv())` passes with no errors
- [ ] Angles are `(cos θ, sin θ)` — never raw radians
- [ ] `reset(seed=42)` is deterministic
- [ ] Episode ends within `max_steps` with `truncated=True` if no earlier termination
- [ ] `tests/test_battalion_env.py` covers reset, step, termination, seeding

## Notes

- Depends on `combat.py` being implemented (see E1.2 epic)
- Part of epic: E1.3 — Gymnasium Environment (1v1)
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement train.py — PPO training loop with W&B logging",
        "labels": ["type: feature", "priority: high", "v1: training", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

`training/train.py` is a one-line stub.  This is the entry point for all
RL training.

## Task

Implement `training/train.py` as a Stable-Baselines3 PPO training script:

```python
# Pseudocode sketch
env = BattalionEnv(config)
model = PPO("MlpPolicy", env, **hyperparams)
wandb.init(project="wargames_training", config=hyperparams)
model.learn(total_timesteps=..., callback=[WandbCallback(), EvalCallback(...)])
model.save(f"checkpoints/{run_id}/final")
```

Requirements:
1. Loads hyperparameters from `configs/default.yaml` (Hydra or plain PyYAML)
2. Initialises W&B run with full config dict including git SHA
3. Uses `EvalCallback` to periodically evaluate vs scripted opponent
4. Saves checkpoints every N steps and at end of run
5. Accepts CLI overrides: `--total-timesteps`, `--seed`, `--config`

## Acceptance Criteria

- [ ] `python training/train.py` runs to completion on CPU in < 30 min for 100 k timesteps
- [ ] W&B run created with `ep_rew_mean` and `ep_len_mean` curves
- [ ] Checkpoint saved to `checkpoints/<run_id>/`
- [ ] W&B config includes all hyperparameters and git SHA
- [ ] Script exits cleanly on `KeyboardInterrupt` (saves checkpoint before exit)

## Notes

- Depends on `battalion_env.py` (E1.3) and `configs/default.yaml`
- Part of epic: E1.4 — Baseline Training Loop
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement mlp_policy.py — feedforward policy network",
        "labels": ["type: feature", "priority: medium", "v1: training", "domain: ml", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

`models/mlp_policy.py` is a one-line stub.  While SB3's built-in
`MlpPolicy` may work initially, a custom architecture lets us tune for
the battalion observation structure.

## Task

Implement `BattalionMlpPolicy` as a SB3-compatible custom policy:

1. Shared feature extractor: `[obs_dim → 128 → 64]` with ReLU, LayerNorm
2. Separate actor head: `[64 → 32 → action_dim]` with Tanh output
3. Separate critic head: `[64 → 32 → 1]`
4. Policy is registered so it can be passed as `policy="BattalionMlpPolicy"` to PPO

## Acceptance Criteria

- [ ] `PPO(BattalionMlpPolicy, env)` initialises without errors
- [ ] Policy parameter count is logged to W&B at run start
- [ ] Forward pass is < 1 ms on CPU for batch size 64
- [ ] `tests/test_mlp_policy.py` covers forward pass shapes

## Notes

- Part of epic: E1.4 — Baseline Training Loop
- Keep architecture simple for v1; transformers are E1.4+ stretch goal
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Add scripted opponent baseline for training",
        "labels": ["type: feature", "priority: high", "v1: training", "domain: sim", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

There is no scripted opponent to train against.  A deterministic,
heuristic-based opponent is needed as the initial curriculum target
and as a permanent evaluation baseline.

## Task

Implement `envs/opponents/scripted.py` with `ScriptedOpponent`:

```python
class ScriptedOpponent:
    def act(self, obs: np.ndarray) -> np.ndarray:
        \"\"\"Return action array given observation.\"\"\"
        ...
```

Behaviours (configurable difficulty level):
- **L1**: Stationary, fires at fixed intervals
- **L2**: Advance toward enemy, fire when in range
- **L3**: Manoeuvre to optimal fire range, retreat if low strength
- **L4**: Flanking attempt (moves perpendicular before engaging)

## Acceptance Criteria

- [ ] `ScriptedOpponent(level=1..4)` is importable and callable
- [ ] L1 opponent is beatable by a random policy within 1 k steps
- [ ] L3 opponent cannot be beaten by a random policy
- [ ] `tests/test_scripted_opponent.py` covers all 4 levels: valid actions, no crashes

## Notes

- Part of epic: E1.4 — Baseline Training Loop
- L4 is a stretch goal; L1–L3 are required for M1
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Fill in configs/default.yaml with training hyperparameters",
        "labels": ["type: feature", "priority: medium", "v1: training", "domain: infra", "status: agent-created"],
        "milestone": "M1: 1v1 Competence",
        "body": """\
## Context

`configs/default.yaml` is a near-empty stub.  Without a proper config
file, `train.py` has no defaults to load and experiments cannot be
reproduced.

## Task

Fill in `configs/default.yaml` with sensible PPO defaults for
battalion training:

```yaml
wandb:
  project: wargames_training
  entity: null  # set via env var

env:
  map_width: 100.0
  map_height: 100.0
  max_steps: 500
  curriculum_level: 1

training:
  algorithm: PPO
  total_timesteps: 1_000_000
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  seed: 42

eval:
  n_eval_episodes: 20
  eval_freq: 50_000
  checkpoint_freq: 100_000

reward:
  win_bonus: 10.0
  damage_scale: 1.0
  survival_bonus: 0.01
  time_penalty: -0.001
```

Also create `configs/experiment_1.yaml` that overrides for the first
real experiment (e.g., increased timesteps, specific seed).

## Acceptance Criteria

- [ ] `python training/train.py` loads `configs/default.yaml` without errors
- [ ] All hyperparameters are documented with inline comments
- [ ] `configs/experiment_1.yaml` overrides at least `seed` and `total_timesteps`
- [ ] W&B run config matches the loaded YAML (no silent defaults)

## Notes

- Part of epic: E1.4 — Baseline Training Loop
""" + ATTRIBUTION,
    },
]

ALL_ISSUES = EPICS + FEATURES


# ---------------------------------------------------------------------------
# Creation logic
# ---------------------------------------------------------------------------


def get_or_none(repo, milestone_title: str, all_milestones: dict):
    """Return milestone object if title exists, else None."""
    if milestone_title in all_milestones:
        return repo.get_milestone(all_milestones[milestone_title])
    return None


def existing_titles(repo) -> set[str]:
    """Return set of all open+closed issue titles (normalised to lower)."""
    titles: set[str] = set()
    for issue in repo.get_issues(state="all"):
        titles.add(issue.title.strip().lower())
    return titles


def ensure_label(repo, name: str, color: str = "ededed", description: str = "") -> None:
    """Create label if it doesn't exist (best-effort)."""
    try:
        repo.get_label(name)
    except Exception:
        try:
            repo.create_label(name=name, color=color, description=description)
            print(f"  Created label: {name}")
        except Exception as exc:
            print(f"  Could not create label '{name}': {exc}", file=sys.stderr)


def run() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPO_NAME")

    if not token or not repo_name:
        print("ERROR: GITHUB_TOKEN and REPO_NAME must be set.", file=sys.stderr)
        sys.exit(1)

    import github
    auth = github.Auth.Token(token)
    gh = github.Github(auth=auth)
    repo = gh.get_repo(repo_name)

    print(f"Connected to {repo.full_name}")

    # Collect milestone map
    all_milestones = {
        m.title: m.number
        for m in repo.get_milestones(state="open")
    }
    print(f"Open milestones: {list(all_milestones.keys()) or '(none)'}")

    # Ensure required labels exist
    required_labels = {
        "status: agent-created": ("0075ca", "Created by automation agent"),
        "type: epic": ("7057ff", "Large multi-task initiative"),
        "type: feature": ("a2eeef", "New feature or enhancement"),
        "v1: simulation": ("bfd4f2", "v1 sim engine work"),
        "v1: environment": ("bfd4f2", "v1 Gymnasium env work"),
        "v1: training": ("bfd4f2", "v1 training pipeline work"),
        "v1: terrain": ("bfd4f2", "v1 terrain work"),
        "v1: rewards": ("bfd4f2", "v1 reward shaping work"),
        "v1: self-play": ("bfd4f2", "v1 self-play work"),
        "v1: evaluation": ("bfd4f2", "v1 evaluation framework"),
        "v1: visualization": ("bfd4f2", "v1 visualisation work"),
        "v1: documentation": ("bfd4f2", "v1 documentation work"),
        "domain: sim": ("e4e669", "Simulation engine domain"),
        "domain: ml": ("e4e669", "Machine learning domain"),
        "domain: viz": ("e4e669", "Visualisation domain"),
        "domain: infra": ("e4e669", "Infrastructure domain"),
        "priority: high": ("d93f0b", "High priority"),
        "priority: medium": ("fbca04", "Medium priority"),
        "priority: low": ("0e8a16", "Low priority"),
    }
    print("\nEnsuring labels exist...")
    for label_name, (color, desc) in required_labels.items():
        ensure_label(repo, label_name, color, desc)

    # Get existing issue titles (idempotency)
    print("\nFetching existing issue titles...")
    known = existing_titles(repo)
    print(f"  {len(known)} existing issues found.")

    created = 0
    skipped = 0

    print("\nCreating issues...")
    for issue_def in ALL_ISSUES:
        title = issue_def["title"]
        if title.strip().lower() in known:
            print(f"  SKIP (exists): {title}")
            skipped += 1
            continue

        # Resolve labels (skip missing ones with a warning, always include agent-created)
        all_repo_labels = {lbl.name for lbl in repo.get_labels()}
        labels_to_apply = [
            lbl for lbl in issue_def.get("labels", []) if lbl in all_repo_labels
        ]
        if "status: agent-created" not in labels_to_apply and "status: agent-created" in all_repo_labels:
            labels_to_apply.append("status: agent-created")

        milestone_obj = get_or_none(repo, issue_def.get("milestone", ""), all_milestones)

        new_issue = repo.create_issue(
            title=title,
            body=issue_def["body"],
            labels=labels_to_apply,
            milestone=milestone_obj,
        )
        print(f"  CREATED #{new_issue.number}: {title}")
        created += 1

    print(f"\nDone. Created: {created}  Skipped: {skipped}")


if __name__ == "__main__":
    run()
