# Development Playbook — Wargames Training

> **Ideology in one sentence:** Build the thinnest runnable end-to-end slice first,
> then deepen layer-by-layer with tight feedback loops, timeboxed spikes, and
> continuous refactoring and testing.

---

## Table of Contents

1. [The Walking Skeleton](#1-the-walking-skeleton)
2. [Iteration Cycle](#2-iteration-cycle)
3. [Risk-First Ordering](#3-risk-first-ordering)
4. [Timeboxed Spikes](#4-timeboxed-spikes)
5. [Refactor Triggers](#5-refactor-triggers)
6. [Definition of Done](#6-definition-of-done)
7. [Feature Checklists](#7-feature-checklists)
   - [New Environment Feature](#71-new-environment-feature)
   - [New Model Architecture](#72-new-model-architecture)
   - [New Training Experiment](#73-new-training-experiment)
   - [Self-Play Iteration](#74-self-play-iteration)

---

## 1. The Walking Skeleton

The **walking skeleton** for `wargames_training` is a minimal pipeline that is:

- **runnable** end-to-end from a single command,
- **exercising real interfaces** (not mocks or placeholders) at every layer,
- **observable** — logged to W&B so you can tell if something is broken.

### What counts as the skeleton

```
envs/battalion_env.py   →   models/mlp_policy.py   →   training/train.py
        ↓                                                       ↓
   sim/ (combat,                                        evaluation loop
   terrain, battalion)                                  (win rate metric)
                                                               ↓
                                                        W&B run logged
```

The skeleton is complete when:

```bash
python training/train.py          # runs without error
pytest tests/ -q                  # green
# → a W&B run appears with at least one logged metric
```

### Golden rule

**`main` must always be runnable.**  
Never merge a branch that breaks `pytest tests/ -q` or makes `train.py` crash
on startup. If you need to commit work-in-progress, use a feature branch and
open a draft PR.

---

## 2. Iteration Cycle

Each feature or improvement follows the same four-step loop:

```
 ┌─────────────────────────────────────────────────────────┐
 │  1. SLICE      — add the thinnest working version        │
 │  2. STABILIZE  — tests, clean seams, document decision   │
 │  3. DEEPEN     — improve quality / capability            │
 │  4. MEASURE    — W&B metrics confirm the improvement     │
 └─────────────────────────────────────────────────────────┘
          ↑                                        │
          └────────────────────────────────────────┘
                       repeat
```

### Slice

Add just enough code to make the feature exercisable end-to-end.
Stub what you have to, but keep real interfaces.

*Example:* Adding terrain → implement `TerrainMap.flat()` first so the env
runs with a valid terrain object before filling in `from_arrays()` and
`line_of_sight()`.

### Stabilize

Before moving on, leave the code better than you found it:

- At least one test per new public function.
- Remove dead code and obvious duplication introduced by the slice.
- Write a short docstring or comment for any non-obvious design choice.

### Deepen

Improve the layer in measurable, incremental commits.  
Each commit should be independently understandable and ideally passing tests.

### Measure

Every meaningful training change must produce a W&B run.  
Compare the new metric baseline with the previous run before merging.

---

## 3. Risk-First Ordering

When multiple things could be worked on, prioritize:

| Priority | Category | Examples in this repo |
|---|---|---|
| 1 | Interfaces / data formats | Observation vector schema, action space, W&B config keys |
| 2 | Core domain logic | `sim/combat.py`, reward function |
| 3 | Frequently-touched code | `battalion_env.py`, `train.py` |
| 4 | Performance-critical paths | Sim step time (only after measuring) |
| 5 | Everything else | Rendering, notebooks, helper scripts |

**Avoid future-proofing** unless you can name the specific v2/v3 requirement.
v1 is 1v1 — do not add multi-agent complexity until v1 milestones are met.

---

## 4. Timeboxed Spikes

A **spike** is a short, disposable investigation to reduce uncertainty before
committing to an implementation.

### What qualifies as a spike

- Evaluating an alternative observation encoding.
- Prototyping a new reward shaping idea.
- Testing whether a third-party library is suitable.
- Profiling whether a hot loop needs JAX or NumPy is sufficient.

### Spike rules

| Rule | Detail |
|---|---|
| **Timebox** | Max 2–3 days. Set a calendar reminder before starting. |
| **Label** | Branch name must start with `spike/` (e.g., `spike/transformer-policy`). |
| **Outcome** | Commit or throw away — document the decision either way. |
| **No merge** | Spike branches are never merged into `main` directly. |
| **Graduate** | If the spike succeeds, open a clean feature branch from `main` and re-implement with tests. |

### Graduating a spike

1. Open a `[FEAT]` or `[EXP]` GitHub issue describing what you learned.
2. Create a feature branch (`feat/` or `exp/`) off `main`.
3. Re-implement with the Definition of Done in mind (Section 6).
4. Reference the spike branch in the PR description for context.

---

## 5. Refactor Triggers

Refactor when you hit one of these signals — not before:

| Signal | Typical symptom |
|---|---|
| **Duplication** | The same logic appears in ≥ 2 places and is about to diverge. |
| **Awkward seam** | Adding a feature requires modifying ≥ 3 unrelated files. |
| **Untestable** | A function can only be tested by running a full training loop. |
| **Naming confusion** | A PR comment or code review asks "what does X mean?" twice. |
| **Performance regression** | A W&B metric degrades after a change to shared code. |

### Refactor rules

- Refactors go in their own commit or PR — do not mix with feature additions.
- A refactor must not change observable behaviour (tests stay green, W&B metrics
  are equivalent).
- Label the PR with `type: refactor`.

---

## 6. Definition of Done

A feature or experiment is **done** when all of the following are true:

### Code

- [ ] `pytest tests/ -q` passes with no new failures.
- [ ] `gymnasium.utils.env_checker.check_env` passes for any environment changes.
- [ ] No hardcoded paths, magic numbers, or secrets in committed code.

### Observations & conventions

- [ ] All new observations are normalized to a reasonable range (typically [−1, 1]
  or [0, 1]).
- [ ] Angles are represented as `(cos θ, sin θ)` pairs — never raw radians.
- [ ] Positions are normalized by map dimensions (`configs/default.yaml` →
  `env.map_width`, `env.map_height`).

### Experiment tracking

- [ ] A W&B config dict is passed at run start (see `configs/default.yaml`).
- [ ] Key metrics are logged every eval interval.
- [ ] The W&B run URL is linked in the PR or the tracking issue.

### Documentation

- [ ] Docstrings added for new public classes and functions.
- [ ] `docs/CHANGELOG.md` updated if the change affects public interfaces.
- [ ] If a project convention is changed, this playbook is updated.

### Experiment issue

- [ ] Every significant training change has a corresponding GitHub issue using the
  `[EXP]` title marker and the experiment issue template.

---

## 7. Feature Checklists

Use these checklists when opening a PR.  Copy the relevant section into your PR description.

---

### 7.1 New Environment Feature

```markdown
## Environment Feature Checklist
- [ ] Observation vector updated and documented (shape, dtype, range)
- [ ] New observations normalized (see playbook §6)
- [ ] Angles as (cos θ, sin θ) — no raw radians
- [ ] Positions normalized by map dimensions
- [ ] `check_env` passes: `python -c "from gymnasium.utils.env_checker import check_env; from YOUR_ENV_MODULE import YOUR_ENV_CLASS; check_env(YOUR_ENV_CLASS())"  # TODO: replace with actual env module/class`
- [ ] Unit tests added in `tests/`
- [ ] `pytest tests/ -q` green
- [ ] W&B config key added if new hyperparameter introduced
- [ ] `docs/CHANGELOG.md` updated
```

---

### 7.2 New Model Architecture

```markdown
## Model Architecture Checklist
- [ ] New module added under `models/`
- [ ] Input shape matches current observation spec (check `battalion_env.observation_space`)
- [ ] Output shape matches current action spec (check `battalion_env.action_space`)
- [ ] Architecture registered / instantiable from a config key
- [ ] Forward pass tested with a dummy observation tensor
- [ ] `pytest tests/ -q` green
- [ ] Comparison W&B run vs. baseline (`mlp_policy`) linked in PR
- [ ] `[EXP]` GitHub issue opened for the architecture experiment
```

---

### 7.3 New Training Experiment

```markdown
## Training Experiment Checklist
- [ ] Config file created in `configs/` (copy from `configs/default.yaml`)
- [ ] Config loaded and passed to W&B at run start
- [ ] `[EXP]` GitHub issue opened with hypothesis and expected outcome
- [ ] Training run completes without crash
- [ ] W&B run URL posted in the tracking issue
- [ ] Key metrics compared against the current baseline run
- [ ] Experiment result (pass/fail/inconclusive) recorded in the issue
- [ ] If successful: `configs/` and `docs/CHANGELOG.md` updated
```

---

### 7.4 Self-Play Iteration

```markdown
## Self-Play Iteration Checklist
- [ ] Self-play pool infrastructure in place (opponent sampling, versioning)
- [ ] Elo or win-rate tracking metric logged to W&B
- [ ] Opponent checkpoint naming convention documented
- [ ] Curriculum stage clearly labelled in config and W&B run tags
- [ ] `[EXP]` GitHub issue opened linking the previous self-play baseline
- [ ] Population collapse detection in place (win rate vs. random opponent)
- [ ] `pytest tests/ -q` green
- [ ] W&B run URL posted in tracking issue
```

---

*This playbook is a living document.
Open a PR with `type: documentation` to propose changes.*
