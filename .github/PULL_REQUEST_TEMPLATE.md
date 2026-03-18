## Summary
<!-- What does this PR do? Link to the issue it closes. -->
Closes #

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Experiment / training change
- [ ] Refactor
- [ ] Documentation
- [ ] Infrastructure

## Changes Made
<!-- List the key changes -->
-
-

## Testing
<!-- How did you verify this works? -->
- [ ] `pytest tests/ -q` passes (all existing tests green)
- [ ] If Gymnasium envs are implemented/modified: `check_env` passes for the affected env(s)
- [ ] Training run completed without errors
- [ ] New tests added for new public functions / behaviours

## Experiment Results (if applicable)
<!-- Link W&B run, paste key metrics. Required for any training change. -->
- [ ] `[EXP]` GitHub issue opened with hypothesis (link: #)
- W&B Run:
- Win rate vs baseline:
- Notes:

## Conventions Checklist
<!-- Verify all applicable items before requesting review. -->
- [ ] New observations normalized to `[−1, 1]` or `[0, 1]`
- [ ] Angles represented as `(cos θ, sin θ)` / `(cos(theta), sin(theta))` — no raw radians
- [ ] Positions normalized by map dimensions
- [ ] W&B config dict passed to `wandb.init()` for any training change
- [ ] No multi-agent complexity added (v1 scope — see [Playbook §3](docs/development_playbook.md#3-risk-first-ordering))
- [ ] No hardcoded paths or secrets

## General Checklist
- [ ] Code follows project style
- [ ] Config changes documented in `configs/`
- [ ] `docs/CHANGELOG.md` updated if public interface changed
- [ ] `docs/development_playbook.md` updated if a project convention changed
- [ ] Feature checklist from [Playbook §7](docs/development_playbook.md#7-feature-checklists) copied and completed above
