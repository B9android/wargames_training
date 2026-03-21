"""Seed doctrinal-opponent issues: creates E1.11 epic and all child tasks.

Addresses the requirement that AI opponents use real Napoleonic line battle
doctrine so that trained intelligence generalises beyond scripted-bot reflexes.

Idempotent — skips any issue whose title already exists.
Run via the orchestration.yml 'seed-roadmap' workflow_dispatch, or locally:

    GITHUB_TOKEN=... REPO_NAME=owner/repo python seed_doctrinal_issues.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Attribution footer appended to every body
# ---------------------------------------------------------------------------

ATTRIBUTION = (
    "\n\n---\n"
    "> 🤖 *Strategist Forge* created this issue automatically to reflect the\n"
    "> doctrinal-AI direction change requested by the repository owner.\n"
    f"> Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n"
    "> Label `status: agent-created` was applied automatically.\n"
)

# ---------------------------------------------------------------------------
# Milestone that all new issues target
# ---------------------------------------------------------------------------

TARGET_MILESTONE = "M5: 2v2 MARL"

# ---------------------------------------------------------------------------
# Epic definition
# ---------------------------------------------------------------------------

EPICS: list[dict] = [
    {
        "title": "[EPIC] E1.11 — Doctrinal Scripted Opponent: Napoleonic Line Battle Tactics",
        "labels": [
            "type: epic",
            "priority: high",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
### Version

v1 (retroactive improvement — prerequisite for all v2+ training quality)

### Goal

Replace the current turn-advance-fire scripted curriculum with a
historically grounded Napoleonic line battle opponent.  Each curriculum
level (4–6) corresponds to a real doctrinal behaviour drawn from period
manuals (Dundas 1788, Ney *Military Studies* 1833, Muller *Elements of
the Science of War* 1811).  The opponent must be credible enough that
defeating it trains transferable tactical intelligence, not just reflexes
shaped against an unrealistic placeholder bot.

### Motivation & Context

The current L4–L5 scripted opponent has Red advance in a straight line
and fire continuously.  This bears no resemblance to how Napoleonic
infantry actually engaged:

- **Volley fire discipline** — muskets reloaded between volleys
  (~15–30 s in practice); no side could sustain fire every second.
- **Optimal range management** — battalions halted at *effective*
  musket range (~80–120 m), not inside it.  Advancing through the kill
  zone without pausing to deliver a volley was tactically reckless.
- **Oblique advance** — units approached at an angle to threaten the
  enemy's flank rather than attacking head-on, forcing the defender to
  either pivot and expose a flank or accept an enfilading volley.
- **Tactical withdrawal** — when strength or morale fell critically,
  well-drilled battalions fell back in good order to reform rather than
  standing to be destroyed in place.
- **Formation alignment** — units dressed their line (maintained lateral
  spacing) while advancing, slowing forward speed but preserving
  firepower cohesion.

Without a doctrinally grounded opponent, agents trained at L4–L5 will
not generalise to self-play (the agent exploits the bot's unrealistic
behaviour) and the historical scenario validation planned for E5.4 will
be meaningless — the agent will never have trained against anything
resembling the historical adversary.

### Child Issues (Tasks)

- [ ] Research: Document Napoleonic line battle doctrine — key sources and simulation rules
- [ ] Implement volley fire discipline: reload cooldown prevents continuous fire in `_step_red()`
- [ ] Implement optimal range management: Red halts and holds at ~60 % fire range, not 80 %
- [ ] Implement oblique advance: Red approaches at ±15–30 ° to threaten Blue's flank
- [ ] Implement tactical withdrawal: Red pulls back when morale < 0.35 or strength < 0.45
- [ ] Revamp curriculum levels 4–5; add optional L6 (full doctrine)
- [ ] Add `tests/test_doctrinal_opponent.py` — validate reload timing, range halt, withdrawal

### Acceptance Criteria

- [ ] L5 Red demonstrates volley fire (≥ 5-step gaps between fire events)
  in ≥ 90 % of test episodes
- [ ] L5 Red consistently halts within ± 20 m of 60 % fire range
- [ ] L5 Red triggers withdrawal in ≥ 80 % of low-morale test cases
- [ ] L5 Red approaches with non-zero lateral offset in ≥ 70 % of
  fresh-engagement test cases
- [ ] All existing `tests/test_battalion_env.py` tests remain green
- [ ] Doctrinal AI beats a random policy > 95 % of the time but is
  consistently beatable by a trained PPO agent

### Priority

high

### Target Milestone

M5: 2v2 MARL (this work is prerequisite before v2 self-play produces
meaningful opponents)
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# Child feature / research issues
# ---------------------------------------------------------------------------

FEATURES: list[dict] = [
    {
        "title": "[RESEARCH] Survey Napoleonic line battle doctrine for scripted AI fidelity",
        "labels": [
            "type: research",
            "priority: medium",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Research Question

What specific behavioural rules from Napoleonic line battle doctrine can
be faithfully and minimally encoded as a scripted opponent in the current
simulation engine?

## Motivation

The scripted curriculum (E1.6) was designed for training convenience, not
historical accuracy.  Before implementing doctrinal AI (E1.11) we need a
concise, citable specification of the rules to code against.

## Approach

1. Review primary sources:
   - Dundas, D. (1788). *Principles of Military Movements*
   - Ney, M. (1833). *Military Studies*
   - Muller, W. (1811). *Elements of the Science of War*
2. Identify the 4–6 behaviours most relevant to a 1v1 continuous 2D
   simulation (range management, volley timing, oblique approach,
   withdrawal, dressing).
3. Translate each behaviour into a quantitative rule that can be coded
   in `_step_red()` and `_red_fire_intensity()`:
   - e.g. "halt when dist ≤ 0.6 × fire_range" for range management
   - e.g. "fire only every N steps" for volley discipline
4. Document the mapping in `docs/doctrinal_opponent_design.md`.

## Findings

*To be filled in when the research is complete.*

## Decision / Recommendation

*To be filled in: which rules are implemented in each curriculum level.*

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement volley fire discipline (reload cooldown) in scripted Red opponent",
        "labels": [
            "type: feature",
            "priority: high",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Problem / Motivation

The current scripted opponent fires at full intensity every single step.
Napoleonic muskets required 15–30 seconds to reload; no battalion could
maintain sustained fire.  An opponent that fires non-stop trains the agent
to absorb continuous punishment rather than exploit the pause after a
volley — a key window in historical combat.

## Proposed Solution

Add a reload cooldown to the scripted Red opponent in `BattalionEnv`:

1. Add `_red_reload_counter: int` to `BattalionEnv.__init__()`,
   initialised to `0` in `reset()`.
2. In `_red_fire_intensity()` (curriculum levels 4–6):
   - If `_red_reload_counter > 0`: return `0.0` (Red is reloading),
     decrement counter.
   - Otherwise: return the level's normal fire intensity **and** reset
     `_red_reload_counter = RED_RELOAD_STEPS`.
3. Define `RED_RELOAD_STEPS: int = 6` as a module-level constant
   (≈ 0.6 s sim time; calibrated to produce a meaningful reload window
   without paralysing Red).
4. Curriculum levels 1–3 are unaffected (they already return 0.0).

## Acceptance Criteria

- [ ] `_red_reload_counter` is initialised to 0 in `reset()`
- [ ] Consecutive non-zero fire events from Red are separated by
  `≥ RED_RELOAD_STEPS` steps in ≥ 90 % of test episodes
- [ ] `RED_RELOAD_STEPS` is a named constant, not a magic number
- [ ] All existing `tests/test_battalion_env.py` tests pass unchanged
- [ ] New test in `tests/test_doctrinal_opponent.py` verifies reload
  gap for levels 4, 5, and 6

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
- Dependent on research issue (doctrinal survey)
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement optimal range management — halt at effective musket range in scripted opponent",
        "labels": [
            "type: feature",
            "priority: high",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Problem / Motivation

The current L3–L5 scripted opponent advances until it is within 80 % of
`fire_range` and then stays there.  Historical battalions halted at the
distance where their volley would be most effective — around 60–70 % of
maximum range — then stood to deliver fire rather than continuing to
close.  Advancing inside effective range was a mistake that produced
unnecessary casualties.

## Proposed Solution

Change the halt distance in `_step_red()` for curriculum levels 4–6:

- **Before:** advance while `dist > fire_range * 0.8`
- **After:** advance while `dist > fire_range * 0.6`

This single constant change (`DOCTRINAL_HALT_FACTOR = 0.6`) is the
correct historical calibration for effective musket range within the
simulation's abstracted scale.

Additionally, at L6 (full doctrine), Red should *back up* if Blue
closes to `dist < fire_range * 0.4` (maintaining the effective range
rather than being overrun).

## Acceptance Criteria

- [ ] `DOCTRINAL_HALT_FACTOR = 0.6` is a named module constant
- [ ] In ≥ 90 % of test episodes, L5 Red stops advancing when within
  `fire_range * (0.6 ± 0.1)` of Blue
- [ ] L6 Red retreats when Blue closes to `< fire_range * 0.4`
- [ ] L3 behaviour (halt at 80 %) is preserved for backward
  compatibility of existing curriculum tests

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement oblique advance maneuver (flank-threatening approach) in scripted opponent",
        "labels": [
            "type: feature",
            "priority: high",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Problem / Motivation

The current scripted opponent always advances directly toward Blue along
the straight line between them.  Historical line infantry almost never
attacked head-on; they approached on an oblique to threaten the enemy's
flank.  This forced the defender to either pivot (exposing their own
flank) or accept an enfilading volley — a key doctrinal pressure point
that the current opponent never exerts.

## Proposed Solution

For curriculum level 6 (full doctrine), apply a lateral offset to Red's
advance heading:

1. Compute `target_angle` toward Blue (as today).
2. Add an oblique offset: `oblique_angle = target_angle ± OBLIQUE_OFFSET`
   where `OBLIQUE_OFFSET = math.pi / 10` (18 °).
3. The sign of the offset is chosen at episode start (random left or
   right) and held constant for the episode, simulating a committed
   flanking approach.
4. Once Red is within `fire_range * 0.6`, the oblique is dropped and
   Red squares up to face Blue directly for the volley exchange.
5. Add `_red_oblique_sign: float` (±1.0) to `BattalionEnv`, initialised
   in `reset()`.

## Acceptance Criteria

- [ ] `OBLIQUE_OFFSET` is a named constant
- [ ] `_red_oblique_sign` is initialised to ±1.0 uniformly at random
  in `reset()`
- [ ] In ≥ 70 % of L6 test episodes, Red's initial approach path has a
  measurable lateral displacement from the straight Blue-Red line
- [ ] Red squares up (lateral offset → 0) once within effective range
- [ ] L4/L5 behaviour is unchanged (no oblique)

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Implement tactical withdrawal when morale/strength falls below threshold in scripted opponent",
        "labels": [
            "type: feature",
            "priority: medium",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Problem / Motivation

The current scripted opponent never retreats.  It continues to advance
and fire even when its morale or strength is critically low, resulting in
it being destroyed in place without any attempt to reform.  This is
ahistorical: well-drilled Napoleonic battalions fell back in good order
when they reached the break point, rallied behind a ridge or reserve
line, and re-engaged when reformed.

## Proposed Solution

Add a withdrawal state to the scripted Red opponent for L5 and L6:

1. Add `_red_withdrawing: bool` to `BattalionEnv`, initialised to
   `False` in `reset()`.
2. In `_step_red()`, at the **start** of the method (before all other
   logic), check:
   - Enter withdrawal if `red.morale < WITHDRAWAL_MORALE_THRESHOLD`
     **or** `red.strength < WITHDRAWAL_STRENGTH_THRESHOLD`.
   - Exit withdrawal (rally) when `red.morale ≥ RALLY_MORALE_THRESHOLD`
     **and** `red.strength ≥ WITHDRAWAL_STRENGTH_THRESHOLD + 0.05`.
3. While `_red_withdrawing is True`: Red turns away from Blue and
   retreats at half max speed toward the edge of the map it started on,
   and does not fire.
4. Constants:
   - `WITHDRAWAL_MORALE_THRESHOLD = 0.35`
   - `WITHDRAWAL_STRENGTH_THRESHOLD = 0.45`
   - `RALLY_MORALE_THRESHOLD = 0.50`

## Acceptance Criteria

- [ ] `_red_withdrawing` is initialised to `False` in `reset()`
- [ ] In ≥ 80 % of test episodes where Red's morale is forced below
  `WITHDRAWAL_MORALE_THRESHOLD`, Red reverses direction within
  2 steps
- [ ] Red does not fire while withdrawing
- [ ] Red resumes advancing once morale recovers above
  `RALLY_MORALE_THRESHOLD`
- [ ] L4 and below behaviour is unchanged

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
""" + ATTRIBUTION,
    },
    {
        "title": "[FEAT] Revamp curriculum levels 4–5 with doctrinal behaviour; add L6 (full doctrine)",
        "labels": [
            "type: feature",
            "priority: high",
            "status: agent-created",
            "domain: ml",
            "v1: training",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Problem / Motivation

Curriculum levels 4 and 5 deliver a sudden difficulty jump from
"advance and fire at 50 %" to "advance and fire at 100 %".  There is no
progression of doctrinal sophistication, so the agent cannot develop
transferable skills step by step.  A doctrinally grounded curriculum
should layer behaviours incrementally.

## Proposed Solution

Redefine L4–L5 and add L6 as follows:

| Level | Movement | Fire | New behaviour |
|-------|----------|------|---------------|
| 1 | Stationary | None | *(unchanged)* |
| 2 | Turn only | None | *(unchanged)* |
| 3 | Advance to 80 % range | None | *(unchanged — easy baseline)* |
| 4 | Advance to 60 % range | Regulated volleys (reload cooldown) | Optimal range + volley discipline |
| 5 | Advance to 60 % range | Regulated volleys | + Tactical withdrawal |
| 6 | Oblique advance to 60 % range | Regulated volleys | + Oblique + withdrawal (full doctrine) |

Update:
- `NUM_CURRICULUM_LEVELS` constant from `5` → `6`
- `_step_red()` docstring and logic for L4–L6
- `_red_fire_intensity()` docstring and logic for L4–L6
- `configs/default.yaml` default `curriculum_level` remains `5`
  (now the withdrawal level; L6 is opt-in)
- `BattalionEnv` class docstring curriculum table

## Acceptance Criteria

- [ ] `NUM_CURRICULUM_LEVELS == 6`
- [ ] `BattalionEnv(curriculum_level=7)` still raises `ValueError`
- [ ] L6 agent beats random policy > 95 % of the time
- [ ] L6 agent is consistently beatable by a PPO-trained Blue agent
- [ ] Existing tests for L1–L3 pass unchanged
- [ ] `configs/default.yaml` documents all 6 levels

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
- Depends on: volley-fire, range-management, oblique-advance,
  and withdrawal issues
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add tests/test_doctrinal_opponent.py — validate doctrinal AI behaviour patterns",
        "labels": [
            "type: chore",
            "priority: medium",
            "status: agent-created",
            "domain: sim",
            "v1: simulation",
        ],
        "milestone": TARGET_MILESTONE,
        "body": """\
## Context

E1.11 introduces several new behavioural invariants for the scripted Red
opponent.  These need dedicated, deterministic tests that verify each
doctrinal property in isolation — not just that the environment doesn't
crash, but that the specific behaviours (reload gap, halt distance,
lateral offset, withdrawal) actually occur.

## Task

Create `tests/test_doctrinal_opponent.py` with the following test
classes:

### `TestVolleyFireDiscipline`
- Verify that consecutive non-zero fire events from Red are separated
  by `≥ RED_RELOAD_STEPS` steps at L4, L5, L6.
- Use a deterministic seed; run for 200 steps; count fire-on frames.

### `TestOptimalRangeManagement`
- Set Blue stationary at a known position; let Red approach from far
  away at L5/L6.
- Assert that Red stops advancing once within
  `fire_range * (0.6 ± 0.15)`.
- Assert that at L6 Red backs up if Blue is moved inside
  `fire_range * 0.4`.

### `TestObliqueAdvance`
- At L6, over 100 reset calls (varying seeds), measure the mean
  absolute lateral displacement of Red's path.
- Assert mean lateral displacement > 20 m (non-trivial oblique).

### `TestTacticalWithdrawal`
- Force `red.morale` below `WITHDRAWAL_MORALE_THRESHOLD` by direct
  assignment after `reset()`; call `step()`.
- Assert Red moves *away* from Blue within 2 steps.
- Assert Red does not fire while withdrawing.

### `TestCurriculumLevelValidation`
- Assert `BattalionEnv(curriculum_level=6)` initialises without error.
- Assert `BattalionEnv(curriculum_level=7)` raises `ValueError`.

## Acceptance Criteria

- [ ] All new tests pass with `pytest tests/test_doctrinal_opponent.py`
- [ ] No existing tests are modified or removed
- [ ] Tests are deterministic (fixed seeds; no random flakiness)

## Notes

- Part of epic: E1.11 — Doctrinal Scripted Opponent
""" + ATTRIBUTION,
    },
]

ALL_ISSUES = EPICS + FEATURES

# ---------------------------------------------------------------------------
# Comments to add to existing related issues
# ---------------------------------------------------------------------------

# Each entry: (issue_number, comment_body)
COMMENTS_ON_EXISTING: list[tuple[int, str]] = [
    (
        17,
        """\
**Update — Doctrinal AI Direction Change**

This issue originally proposed a 4-level scripted opponent
(`envs/opponents/scripted.py`).  The implementation was superseded by
the inline scripted curriculum in `BattalionEnv._step_red()` (closed
under E1.4 and E1.6).

A new epic — **E1.11 — Doctrinal Scripted Opponent: Napoleonic Line
Battle Tactics** — has been opened to revamp the scripted curriculum
with historically grounded behaviour (volley fire discipline, optimal
range management, oblique advance, tactical withdrawal).  The work in
E1.11 subsumes and extends what was originally scoped here.

Please track progress under E1.11 rather than this issue.

> 🤖 *Strategist Forge* added this comment automatically.
""",
    ),
    (
        33,
        """\
**Update — Doctrinal AI Direction Change**

The original scope of this issue ("advance and fire, simple heuristic")
has been implemented inside `BattalionEnv._step_red()` as part of E1.6.

A new epic — **E1.11 — Doctrinal Scripted Opponent: Napoleonic Line
Battle Tactics** — supersedes the simple heuristic with a doctrinally
grounded curriculum aligned with period manuals (Dundas, Ney, Muller).
All further scripted-opponent work should be tracked under E1.11.

> 🤖 *Strategist Forge* added this comment automatically.
""",
    ),
    (
        73,
        """\
**Dependency Notice — E1.11 must be complete before this epic**

The v2 2v2 curriculum (this epic) bootstraps from v1 checkpoint
policies.  If the v1 scripted opponent does not exhibit realistic
Napoleonic line battle behaviour, agents trained against it will carry
forward exploits and reflexes that are useless (or actively harmful) in
2v2 self-play.

**E1.11 — Doctrinal Scripted Opponent** has been opened to address this.
It is a blocking prerequisite for this epic: the v1 policy used as the
frozen seed in stage `1v1 → 2v1 → 2v2` must have been trained against a
doctrinally credible opponent.

Recommended: add E1.11 as a dependency in the child-issues checklist
above, and do not begin the `1v1 (frozen v1)` stage until E1.11 is
closed.

> 🤖 *Strategist Forge* added this comment automatically.
""",
    ),
    (
        117,
        """\
**Foundation Notice — E1.11 is prerequisite for meaningful validation**

Historical Scenario Validation (this epic) asks whether trained policies
reproduce historically documented tactics.  That question is only
answerable if the agent was trained against an opponent that itself
exhibits those tactics.

**E1.11 — Doctrinal Scripted Opponent** has been opened to ground the
scripted curriculum in real Napoleonic line battle doctrine (volley fire
discipline, optimal range management, oblique advance, tactical
withdrawal).  Completing E1.11 before E5.4 is essential: without it,
"historical validation" would compare a trained agent against a
doctrinally empty baseline and the results would be uninterpretable.

Recommendation: raise priority from `low` to `medium` and note E1.11
as a soft prerequisite in the acceptance criteria above.

> 🤖 *Strategist Forge* added this comment automatically.
""",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_milestone_obj(repo, milestone_title: str):
    """Return an open milestone object by title, or None."""
    for m in repo.get_milestones(state="open"):
        if m.title == milestone_title:
            return m
    return None


def existing_titles(repo) -> set[str]:
    """Return normalised titles of all open + closed issues."""
    return {issue.title.strip().lower() for issue in repo.get_issues(state="all")}


def ensure_label(repo, name: str, color: str = "ededed", description: str = "") -> None:
    """Create label if it does not exist (best-effort; never raises)."""
    try:
        repo.get_label(name)
    except Exception:
        try:
            repo.create_label(name=name, color=color, description=description)
            print(f"  Created label: {name}")
        except Exception as exc:
            print(f"  Could not create label '{name}': {exc}", file=sys.stderr)


def issue_already_commented(issue, marker: str) -> bool:
    """Return True if *marker* appears in any existing comment body."""
    for comment in issue.get_comments():
        if marker in comment.body:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPO_NAME")

    if not token or not repo_name:
        print("ERROR: GITHUB_TOKEN and REPO_NAME must be set.", file=sys.stderr)
        sys.exit(1)

    import github

    auth = github.Auth.Token(token)
    gh = github.Github(auth=auth, per_page=100)
    repo = gh.get_repo(repo_name)

    print(f"Connected to {repo.full_name}")

    # ------------------------------------------------------------------
    # Ensure required labels exist
    # ------------------------------------------------------------------
    required_labels = {
        "status: agent-created": ("0075ca", "Created by automation agent"),
        "type: epic": ("7057ff", "Large multi-task initiative"),
        "type: feature": ("a2eeef", "New feature or enhancement"),
        "type: research": ("c5def5", "Investigation or literature review"),
        "type: chore": ("e4e669", "Maintenance or housekeeping task"),
        "priority: high": ("d93f0b", "High priority"),
        "priority: medium": ("fbca04", "Medium priority"),
        "domain: sim": ("e4e669", "Simulation engine domain"),
        "domain: ml": ("e4e669", "Machine learning domain"),
        "v1: simulation": ("bfd4f2", "v1 sim engine work"),
        "v1: training": ("bfd4f2", "v1 training pipeline work"),
    }
    print("\nEnsuring labels exist...")
    for label_name, (color, desc) in required_labels.items():
        ensure_label(repo, label_name, color, desc)

    # ------------------------------------------------------------------
    # Resolve milestone
    # ------------------------------------------------------------------
    milestone_obj = get_milestone_obj(repo, TARGET_MILESTONE)
    if milestone_obj is None:
        print(
            f"  WARNING: milestone '{TARGET_MILESTONE}' not found — issues will be "
            "created without a milestone.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Create new issues (idempotent)
    # ------------------------------------------------------------------
    print("\nFetching existing issue titles for idempotency check...")
    known = existing_titles(repo)
    print(f"  {len(known)} existing issues found.")

    all_repo_labels = {lbl.name for lbl in repo.get_labels()}

    created = 0
    skipped = 0

    print("\nCreating issues and epics...")
    for issue_def in ALL_ISSUES:
        title = issue_def["title"]
        if title.strip().lower() in known:
            print(f"  SKIP (exists): {title}")
            skipped += 1
            continue

        labels_to_apply = [
            lbl for lbl in issue_def.get("labels", []) if lbl in all_repo_labels
        ]
        if (
            "status: agent-created" not in labels_to_apply
            and "status: agent-created" in all_repo_labels
        ):
            labels_to_apply.append("status: agent-created")

        new_issue = repo.create_issue(
            title=title,
            body=issue_def["body"],
            labels=labels_to_apply,
            milestone=milestone_obj,
        )
        print(f"  CREATED #{new_issue.number}: {title}")
        created += 1

    # ------------------------------------------------------------------
    # Add direction-change comments to existing related issues
    # ------------------------------------------------------------------
    # Use a unique marker so we never post the same comment twice.
    COMMENT_MARKER = "🤖 *Strategist Forge* added this comment automatically."

    print("\nAdding comments to related issues...")
    for issue_number, comment_body in COMMENTS_ON_EXISTING:
        try:
            issue = repo.get_issue(issue_number)
            if issue_already_commented(issue, COMMENT_MARKER):
                print(f"  SKIP comment on #{issue_number} (already present)")
                continue
            issue.create_comment(comment_body)
            print(f"  COMMENTED on #{issue_number}: {issue.title}")
        except Exception as exc:
            print(
                f"  WARNING: could not comment on #{issue_number}: {exc}",
                file=sys.stderr,
            )

    print(f"\nDone. Created: {created}  Skipped: {skipped}")


if __name__ == "__main__":
    run()
