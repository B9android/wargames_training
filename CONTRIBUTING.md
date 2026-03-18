# Contributing to Wargames Training

Thank you for contributing! This document covers everything you need to know to
work effectively on this project.

---

## Development Methodology

Before diving in, please read the **[Development Playbook](docs/development_playbook.md)**.
It describes the core workflow for this project:

- **Build a walking skeleton first** — thin, end-to-end, always runnable.
- **Iterate layer-by-layer** — slice → stabilize → deepen → measure.
- **`main` stays green** — never merge code that breaks `pytest tests/ -q`.

---

## Quick-Start

```bash
git clone https://github.com/B9android/wargames_training.git
cd wargames_training
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login                      # create a free account at wandb.ai
pytest tests/ -q                 # must be green before you start
```

---

## Branch Naming

| Prefix | Use |
|---|---|
| `feat/` | New feature or environment capability |
| `fix/` | Bug fix |
| `exp/` | Training experiment (requires `[EXP]` issue) |
| `refactor/` | Code cleanup with no behaviour change |
| `docs/` | Documentation only |
| `spike/` | Disposable investigation — never merged directly (see Playbook §4) |

---

## Commit Style

- Use the imperative mood: `"Add terrain cover modifier"` not `"Added …"`.
- One logical change per commit.
- Reference the tracking issue: `"Fix morale routing threshold (#42)"`.

---

## Opening Issues

Use the appropriate title marker so the triage agent labels it correctly:

| Marker | When to use |
|---|---|
| `[BUG]` | Something is broken |
| `[EXP]` | Training experiment with a hypothesis |
| `[FEAT]` | New capability or environment feature |
| `[EPIC]` | Large multi-issue body of work |
| `[RESEARCH]` | Literature review, architectural study |

Every significant training change **must** have a `[EXP]` issue.
Significant means: new reward function, new observation encoding, new model
architecture, new hyperparameter sweep, or any self-play iteration.  
Minor housekeeping (typos, log formatting) does not need an issue.

---

## Code Conventions

These conventions are enforced by code review and the PR checklist:

### Observations
- Normalize all observations to a reasonable range (typically `[−1, 1]` or `[0, 1]`).
- Represent angles as `(cos θ, sin θ)` pairs — **never raw radians**.
- Normalize positions by map dimensions (`env.map_width`, `env.map_height` from
  `configs/default.yaml`).

### Environments
- Inherit from `gymnasium.Env` (single-agent) or `pettingzoo.ParallelEnv` (multi-agent).
- Once a concrete Gymnasium environment (not just a stub) is implemented, run and fix
  `check_env` from `gymnasium.utils.env_checker` before merging.

### Experiment Tracking
- Every training run **must** be logged to W&B with a config dict.
- Load your config from `configs/` and pass it verbatim to `wandb.init(config=...)`.
- Post the W&B run URL in your PR or tracking issue.

### v1 Scope Boundary
- **Do not add multi-agent complexity until v1 milestones are met.**
- If you think a v2+ feature is needed now, open a `[RESEARCH]` issue and discuss
  before implementing.

---

## Testing

```bash
pytest tests/ -q          # full suite, must be green
```

- Add at least one test per new public function.
- Tests live in `tests/` and follow existing naming (`test_<module>.py`).
- Use `unittest.TestCase` or plain pytest functions — both are fine (see existing tests).
- For environment changes, also run:

```python
from envs.battalion_env import BattalionEnv
from gymnasium.utils.env_checker import check_env
check_env(BattalionEnv())
```

---

## Pull Request Process

1. Fill in the **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`).
2. Copy the relevant checklist from the [Development Playbook §7](docs/development_playbook.md#7-feature-checklists)
   into your PR description.
3. Link the tracking issue (`Closes #N`).
4. Request a review — at least one approval is required before merge.

---

## Definition of Done

See [Playbook §6](docs/development_playbook.md#6-definition-of-done) for the full list.
Short version:

- `pytest tests/ -q` green ✓
- W&B run logged and URL shared ✓
- `docs/CHANGELOG.md` updated (if public interface changed) ✓
- `[EXP]` issue opened (if training change) ✓

---

## Resources

| Resource | Location |
|---|---|
| Development Playbook | [`docs/development_playbook.md`](docs/development_playbook.md) |
| Project Roadmap | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| Orchestration Runbook | [`docs/ORCHESTRATION_RUNBOOK.md`](docs/ORCHESTRATION_RUNBOOK.md) |
| Default Config | [`configs/default.yaml`](configs/default.yaml) |
| W&B Project | [wandb.ai — wargames_training](https://wandb.ai) |
