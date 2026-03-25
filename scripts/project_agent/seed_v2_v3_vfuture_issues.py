# SPDX-License-Identifier: MIT
"""Seed v2, v3, v4, and v5 roadmap issues: creates all epics and key feature tasks.

Idempotent — skips any issue whose title already exists.

Run via the seed-v2-v3-vfuture-issues GitHub Actions workflow or locally with:
    GITHUB_TOKEN=... REPO_NAME=owner/repo python seed_v2_v3_vfuture_issues.py
    GITHUB_TOKEN=... REPO_NAME=owner/repo DRY_RUN=true python seed_v2_v3_vfuture_issues.py
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
    "> 🤖 *Strategist Forge* created this issue automatically as part of v2/v3/v4/v5 roadmap seeding.\n"
    f"> Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n"
    "> Label `status: agent-created` was requested; it may be absent if the label does not exist on this repository.\n"
)

# ---------------------------------------------------------------------------
# ── v2 EPICS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

V2_EPICS: list[dict] = [
    {
        "title": "[EPIC] E2.1 — PettingZoo Multi-Agent Environment",
        "labels": [
            "type: epic", "priority: high",
            "v2: environment", "domain: env", "status: agent-created",
        ],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Version

v2

### Goal

A fully compliant `pettingzoo.ParallelEnv` subclass supporting NvN battalion
combat.  Each agent receives its own local observation; a global state tensor
is provided for the centralized critic.  The API is backward-compatible with
v1's `BattalionEnv` for policy bootstrapping.

### Motivation & Context

v1 trains a single battalion against a scripted opponent.  v2 requires
multiple cooperating agents per side.  PettingZoo's `ParallelEnv` is the
standard interface for simultaneous multi-agent environments and integrates
cleanly with EPyMARL / custom MAPPO implementations.

### Child Issues (Tasks)

- [ ] Implement `envs/multi_battalion_env.py` — PettingZoo `ParallelEnv` for NvN
- [ ] Define per-agent observation space (local obs + communication radius)
- [ ] Expose global state tensor for centralized critic
- [ ] Implement partial observability / fog of war
- [ ] Add PettingZoo API test (`pettingzoo.test.api_test`) to CI
- [ ] Add `tests/test_multi_battalion_env.py`

### Acceptance Criteria

- [ ] `pettingzoo.test.api_test(MultiBattalionEnv(), verbose=True)` passes
- [ ] Observation vectors remain normalized to `[-1, 1]` / `[0, 1]`
- [ ] Angles encoded as `(cos θ, sin θ)` — never raw radians
- [ ] Global state tensor exposed via `state()` method
- [ ] Environment seeds for reproducibility

### Priority

high

### Target Milestone

M5: 2v2 MARL""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.2 — MAPPO Implementation (Centralized Critic)",
        "labels": [
            "type: epic", "priority: high",
            "v2: mappo", "domain: ml", "status: agent-created",
        ],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Version

v2

### Goal

A Multi-Agent PPO implementation with parameter-sharing across homogeneous
agents and a centralized value critic that conditions on global state.
Agents are trained cooperatively in a shared reward setting.

### Motivation & Context

Independent PPO agents in cooperative tasks suffer from non-stationarity.
MAPPO (Yu et al. 2021) addresses this via a centralized critic; parameter
sharing exploits the homogeneous nature of battalion policies.

### Child Issues (Tasks)

- [ ] Implement `models/mappo_policy.py` — shared actor + centralized critic
- [ ] Implement `training/train_mappo.py` — MAPPO training loop with shared rollout buffer
- [ ] Add parameter sharing toggle (shared / separate actor weights)
- [ ] Add experiment config `configs/experiment_mappo_2v2.yaml`
- [ ] Integrate W&B per-agent and aggregate reward logging
- [ ] Add `tests/test_mappo_policy.py`

### Acceptance Criteria

- [ ] MAPPO training loop runs without errors on 2v2 scenario
- [ ] Centralized critic receives global state; actors receive local obs
- [ ] Parameter sharing reduces training memory vs. independent policies
- [ ] W&B run tagged with config dict and per-agent reward curves
- [ ] Win rate against scripted 2v2 opponent exceeds 55 % after training budget

### Priority

high

### Target Milestone

M5: 2v2 MARL""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.3 — 2v2 Curriculum",
        "labels": [
            "type: epic", "priority: medium",
            "v2: curriculum", "domain: ml", "status: agent-created",
        ],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Version

v2

### Goal

A staged curriculum that bootstraps cooperation skills by progressing
from 1v1 (using the v1 checkpoint) through 2v1 (asymmetric advantage)
to 2v2 (full cooperative scenario).

### Motivation & Context

Jumping directly to 2v2 from random initialization is sample-inefficient.
Curriculum transfer from v1 policies seeds cooperative behaviors and
accelerates convergence.

### Child Issues (Tasks)

- [ ] Design curriculum stages: `1v1 (frozen v1) → 2v1 → 2v2`
- [ ] Implement `training/curriculum_scheduler.py` — stage promotion logic
- [ ] Add W&B curriculum stage metric and promotion event logging
- [ ] Create scenario configs: `configs/scenarios/2v1.yaml`, `configs/scenarios/2v2.yaml`
- [ ] Add curriculum transfer tests

### Acceptance Criteria

- [ ] Curriculum progresses automatically when stage win rate ≥ threshold
- [ ] Stage transitions logged to W&B with episode count
- [ ] 2v2 policy initialized from v1 checkpoint parameters (shared portion)
- [ ] Ablation experiment confirms curriculum outperforms random init

### Priority

medium

### Target Milestone

M5: 2v2 MARL""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.4 — Coordination Metrics & Analysis",
        "labels": [
            "type: epic", "priority: medium",
            "v2: metrics", "domain: eval", "status: agent-created",
        ],
        "milestone": "M6: v2 Complete",
        "body": """\
### Version

v2

### Goal

Quantitative metrics that measure emergent coordination behaviors:
flanking, fire concentration, and mutual support.  Metrics are logged
to W&B and visualized in analysis notebooks.

### Motivation & Context

Win rate alone does not reveal whether the policy discovered genuine
tactical coordination.  Domain-specific metrics are necessary to
interpret and guide training.

### Child Issues (Tasks)

- [ ] Implement `envs/metrics/coordination.py` — flanking_ratio, fire_concentration, mutual_support_score
- [ ] Log all coordination metrics to W&B per episode
- [ ] Create `notebooks/v2_coordination_analysis.ipynb`
- [ ] Add `tests/test_coordination_metrics.py`
- [ ] Document metric definitions in `docs/metrics_reference.md`

### Acceptance Criteria

- [ ] All three metrics are logged per episode to W&B
- [ ] `flanking_ratio` > 0.3 indicates meaningful flanking behavior
- [ ] Notebook reproduces metric plots from a sample run
- [ ] Metrics pass unit tests with known episode fixtures

### Priority

medium

### Target Milestone

M6: v2 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.5 — Scale to NvN (up to 6v6)",
        "labels": [
            "type: epic", "priority: medium",
            "v2: scaling", "domain: env", "status: agent-created",
        ],
        "milestone": "M6: v2 Complete",
        "body": """\
### Version

v2

### Goal

Extend the multi-agent environment and training pipeline to support
variable team sizes up to 6v6 without architectural changes.  Benchmark
performance and identify observation-radius sweet spots.

### Motivation & Context

A flexible NvN framework is the prerequisite for v3 hierarchical
command layers.  Scaling also stress-tests MAPPO's centralized critic
under growing global-state dimensionality.

### Child Issues (Tasks)

- [ ] Parameterize `MultiBattalionEnv` with `n_blue` / `n_red` arguments
- [ ] Create scenario configs: `configs/scenarios/3v3.yaml`, `4v4.yaml`, `6v6.yaml`
- [ ] Benchmark wall-clock training time vs. team size (1v1 → 6v6)
- [ ] Sweep observation radius: `[100m, 250m, 500m, full]` at 4v4
- [ ] Add performance regression test (steps/sec must not drop > 20 % vs. 2v2)
- [ ] Document scaling findings in `docs/scaling_notes.md`

### Acceptance Criteria

- [ ] 6v6 environment steps without errors
- [ ] 4v4 trained policy beats scripted opponents with ≥ 55 % win rate
- [ ] Observation radius ablation committed as `[EXP]` issue with conclusions
- [ ] No more than 20 % throughput regression from 2v2 baseline

### Priority

medium

### Target Milestone

M6: v2 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.6 — Multi-Agent Self-Play",
        "labels": [
            "type: epic", "priority: medium",
            "v2: self-play", "domain: ml", "status: agent-created",
        ],
        "milestone": "M6: v2 Complete",
        "body": """\
### Version

v2

### Goal

Port the v1 self-play infrastructure to the multi-agent setting.
Implement team Elo rating, historical snapshot opponent sampling,
and Nash exploitability tracking.

### Motivation & Context

Self-play is essential for producing robust cooperative policies that
don't exploit scripted weaknesses.  Team Elo enables objective comparison
of checkpoint quality across training runs.

### Child Issues (Tasks)

- [ ] Extend `training/self_play.py` for multi-agent team play
- [ ] Implement team Elo rating (`training/elo.py`)
- [ ] Add historical snapshot opponent pool (keep last N checkpoints)
- [ ] Implement Nash exploitability proxy metric
- [ ] Add W&B logging for Elo and exploitability
- [ ] Add `tests/test_team_elo.py`

### Acceptance Criteria

- [ ] Self-play loop stable for ≥ 1M steps without NaN rewards
- [ ] Team Elo increases monotonically over first 500k self-play steps
- [ ] Exploitability proxy decreases or plateaus (not diverging)
- [ ] Snapshot pool correctly rotates out oldest checkpoints

### Priority

medium

### Target Milestone

M6: v2 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E2.7 — v2 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v2: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M6: v2 Complete",
        "body": """\
### Version

v2

### Goal

Complete documentation for all v2 components, publish the v2 release,
and leave the codebase ready for v3 hierarchical RL work.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v2 epics complete
- [ ] Write `docs/multi_agent_guide.md` — MAPPO setup, observation design, curriculum
- [ ] Add v2 architecture diagram to `docs/`
- [ ] Draft `CHANGELOG.md` v2 section
- [ ] Create GitHub release tag `v2.0.0`
- [ ] Seed v3 planning issues (trigger `seed-v3-issues` workflow or merge planning PR)

### Acceptance Criteria

- [ ] All v2 epics marked complete in ROADMAP.md
- [ ] Multi-agent guide reviewed and merged
- [ ] `v2.0.0` release tag exists with release notes
- [ ] v3 epics seeded and triaged

### Priority

low

### Target Milestone

M6: v2 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v2 FEATURE / TASK ISSUES ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

V2_TASKS: list[dict] = [
    # E2.1 tasks
    {
        "title": "Implement `envs/multi_battalion_env.py` — PettingZoo ParallelEnv for NvN",
        "labels": ["type: feature", "priority: high", "v2: environment", "domain: env", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Implement the `MultiBattalionEnv(ParallelEnv)` class wrapping the sim engine. "
                "Support variable `n_blue`/`n_red` team sizes. Expose `observation_spaces`, "
                "`action_spaces`, `step()`, `reset()`, and `state()` methods per PettingZoo API.\n"
                + ATTRIBUTION,
    },
    {
        "title": "Expose global state tensor for centralized critic in MultiBattalionEnv",
        "labels": ["type: feature", "priority: high", "v2: environment", "v2: mappo", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Add a `state()` method to `MultiBattalionEnv` returning a flat tensor of all agents' "
                "positions, headings, strengths, morales, and terrain features. Used by the "
                "centralized critic in MAPPO.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement partial observability (fog of war) in MultiBattalionEnv",
        "labels": ["type: feature", "priority: medium", "v2: environment", "domain: env", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Each agent only observes enemies within its detection radius. "
                "Enemies outside the radius are masked from the observation vector. "
                "Detection radius is a configurable parameter.\n" + ATTRIBUTION,
    },
    # E2.2 tasks
    {
        "title": "Implement `models/mappo_policy.py` — shared actor + centralized critic",
        "labels": ["type: feature", "priority: high", "v2: mappo", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Shared actor network conditioned on local observation. "
                "Centralized critic conditioned on global state tensor. "
                "Compatible with SB3 rollout buffer or custom buffer.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/train_mappo.py` — MAPPO training loop",
        "labels": ["type: feature", "priority: high", "v2: mappo", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Full MAPPO training loop: collect joint rollouts, compute GAE with centralized "
                "value, update shared actor and centralized critic. W&B logging, checkpoint saving.\n"
                + ATTRIBUTION,
    },
    # E2.3 tasks
    {
        "title": "Implement `training/curriculum_scheduler.py` — multi-agent stage promotion",
        "labels": ["type: feature", "priority: medium", "v2: curriculum", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": "Stage-based scheduler: 1v1 → 2v1 → 2v2. Promotes when rolling win rate "
                "exceeds a configurable threshold. Logs stage transitions to W&B.\n" + ATTRIBUTION,
    },
    # E2.4 tasks
    {
        "title": "Implement `envs/metrics/coordination.py` — flanking, fire concentration, mutual support",
        "labels": ["type: feature", "priority: medium", "v2: metrics", "domain: eval", "status: agent-created"],
        "milestone": "M6: v2 Complete",
        "body": "Three metrics: `flanking_ratio` (fraction of episodes where an agent attacks "
                "from rear/flank arc), `fire_concentration` (fraction of volleys where ≥ 2 agents "
                "target the same enemy), `mutual_support_score` (average distance between allies "
                "normalized by map size, lower = tighter support).\n" + ATTRIBUTION,
    },
    # E2.5 tasks
    {
        "title": "Parameterize MultiBattalionEnv with n_blue / n_red for NvN scaling",
        "labels": ["type: feature", "priority: medium", "v2: scaling", "domain: env", "status: agent-created"],
        "milestone": "M6: v2 Complete",
        "body": "Allow the environment to be instantiated with any `n_blue` and `n_red` values "
                "(1–6 per side). All observation/action spaces must resize accordingly. "
                "Add regression test ensuring 6v6 steps without errors.\n" + ATTRIBUTION,
    },
    # E2.6 tasks
    {
        "title": "Implement team Elo rating in `training/elo.py`",
        "labels": ["type: feature", "priority: medium", "v2: self-play", "domain: eval", "status: agent-created"],
        "milestone": "M6: v2 Complete",
        "body": "Team Elo rating system updated after every self-play match. "
                "K-factor configurable (default 32). Ratings persisted between training runs "
                "via checkpoint metadata. Logged to W&B.\n" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v3 EPICS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

V3_EPICS: list[dict] = [
    {
        "title": "[EPIC] E3.1 — SMDP / Options Framework",
        "labels": [
            "type: epic", "priority: high",
            "v3: smdp", "domain: ml", "status: agent-created",
        ],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": """\
### Version

v3

### Goal

Implement a Semi-MDP / Options framework that wraps v2's battalion-level
PettingZoo environment.  Macro-actions (options) are sequences of
primitive actions with learned termination conditions.  This is the
foundation for hierarchical commanders.

### Motivation & Context

HRL requires a formal abstraction layer between echelons.  The SMDP
formulation allows high-level commanders to issue multi-step macro-commands
while low-level battalion policies execute them — matching Black (NPS 2024).

### Child Issues (Tasks)

- [ ] Implement `envs/options.py` — `Option` dataclass (initiation set, policy, termination)
- [ ] Implement `envs/smdp_wrapper.py` — wraps `MultiBattalionEnv` with option selection
- [ ] Define macro-action vocabulary: `advance_sector`, `defend_position`, `flank_left`, `flank_right`, `withdraw`, `concentrate_fire`
- [ ] Implement option termination conditions (time limit, objective reached, routing)
- [ ] Add `tests/test_smdp_wrapper.py`

### Acceptance Criteria

- [ ] `SMDPWrapper(MultiBattalionEnv())` passes PettingZoo API tests
- [ ] Options execute until termination and return aggregate rewards
- [ ] Temporal abstraction ratio (macro-steps / primitive-steps) logged to W&B
- [ ] All six macro-actions trigger distinct primitive-action patterns

### Priority

high

### Target Milestone

M7: HRL Battalion→Brigade""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.2 — Brigade Commander Layer",
        "labels": [
            "type: epic", "priority: high",
            "v3: brigade", "domain: ml", "status: agent-created",
        ],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": """\
### Version

v3

### Goal

A brigade-level RL agent that issues macro-commands (options) to frozen
battalion-level policies.  The brigade commander observes aggregate sector
states and selects which option to execute for each battalion.

### Motivation & Context

Historical doctrine organizes battalions under brigade command.  Training
a brigade-level controller on top of frozen battalion policies matches
the bottom-up curriculum in the HRL literature.

### Child Issues (Tasks)

- [ ] Implement `envs/brigade_env.py` — high-level MDP (brigade observation / macro-action space)
- [ ] Define brigade observation: sector control percentages, battalion strengths/morales, enemy threat vector
- [ ] Implement option dispatcher — translates brigade macro-commands to battalion option selections
- [ ] Implement PPO brigade training loop (`training/train_brigade.py`)
- [ ] Freeze battalion policies during brigade training (load v2 checkpoint, `requires_grad=False`)
- [ ] Add `tests/test_brigade_env.py`

### Acceptance Criteria

- [ ] Brigade PPO converges (reward > random baseline) within 500k steps
- [ ] Battalion policies are frozen (no gradient updates) during brigade training
- [ ] Brigade win rate vs. scripted 2v2 opponent exceeds v2 MAPPO baseline
- [ ] Brigade observation and action shapes documented in `docs/hrl_architecture.md`

### Priority

high

### Target Milestone

M7: HRL Battalion→Brigade""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.3 — Division Commander Layer",
        "labels": [
            "type: epic", "priority: medium",
            "v3: division", "domain: ml", "status: agent-created",
        ],
        "milestone": "M8: v3 Complete",
        "body": """\
### Version

v3

### Goal

A division-level RL agent that issues operational-level commands to
frozen brigade policies.  Enables three-echelon command: battalion →
brigade → division.

### Motivation & Context

Extending to division level tests whether the HRL abstraction scales
to a third echelon and whether operational-level maneuver emerges.

### Child Issues (Tasks)

- [ ] Implement `envs/division_env.py` — top-level MDP (division observation / operational-action space)
- [ ] Define division observation: theatre map sectors, brigade status summaries, objective control
- [ ] Implement command translation: division operational-command → brigade macro-command
- [ ] Implement PPO division training loop (`training/train_division.py`)
- [ ] Freeze brigade policies during division training
- [ ] Add `tests/test_division_env.py`

### Acceptance Criteria

- [ ] Three-echelon hierarchy executes end-to-end: division → brigade → battalion → sim
- [ ] Division policy converges within 1M steps
- [ ] Division win rate vs. flat MARL 4v4 documented in ablation experiment issue

### Priority

medium

### Target Milestone

M8: v3 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.4 — Hierarchical Curriculum (bottom-up training)",
        "labels": [
            "type: epic", "priority: high",
            "v3: hrl-curriculum", "domain: ml", "status: agent-created",
        ],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": """\
### Version

v3

### Goal

A structured bottom-up training curriculum: train battalions first (v2),
freeze them and train brigade, freeze brigade and train division.
Each stage has clear promotion criteria.

### Motivation & Context

Bottom-up training prevents the higher-level agent from exploiting an
unstable lower-level policy, a common failure mode in naive HRL.

### Child Issues (Tasks)

- [ ] Document three-phase training protocol in `docs/hrl_training_protocol.md`
- [ ] Implement policy freezing utility (`training/utils/freeze_policy.py`)
- [ ] Add curriculum stage configs: `configs/hrl/phase1_battalion.yaml`, `phase2_brigade.yaml`, `phase3_division.yaml`
- [ ] Implement curriculum promotion checks (win rate, Elo threshold)
- [ ] Add integration test: full three-phase curriculum on small scenario

### Acceptance Criteria

- [ ] Each phase trains in isolation without modifying frozen-level weights
- [ ] Promotion criteria documented and enforced programmatically
- [ ] Integration test passes on CI (small scenario, 10k steps per phase)

### Priority

high

### Target Milestone

M7: HRL Battalion→Brigade""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.5 — Temporal Abstraction Tuning",
        "labels": [
            "type: epic", "priority: medium",
            "v3: temporal", "domain: ml", "status: agent-created",
        ],
        "milestone": "M8: v3 Complete",
        "body": """\
### Version

v3

### Goal

Systematically experiment with the temporal abstraction ratio between
echelons.  Find the ratio that maximizes brigade win rate while
minimizing catastrophic forgetting at battalion level.

### Motivation & Context

The temporal abstraction ratio (how many battalion steps per brigade
macro-step) is a critical hyperparameter in hierarchical RL.  Too short
and the hierarchy offers no benefit; too long and the commander loses
fine-grained control.

### Child Issues (Tasks)

- [ ] Implement `training/adaptive_temporal.py` — adaptive option length based on episode progress
- [ ] Create sweep config `configs/sweeps/temporal_ratio_sweep.yaml` (ratios: 5, 10, 20, 50)
- [ ] Run temporal ratio sweep; commit results as `[EXP]` issue
- [ ] Add temporal abstraction ratio to W&B run config

### Acceptance Criteria

- [ ] Sweep runs cleanly across all four ratios
- [ ] Results issue identifies optimal ratio with justification
- [ ] Optimal ratio hard-coded as default in `configs/hrl/phase2_brigade.yaml`

### Priority

medium

### Target Milestone

M8: v3 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.6 — Multi-Model Policy Library (per echelon)",
        "labels": [
            "type: epic", "priority: medium",
            "v3: policy-lib", "domain: ml", "status: agent-created",
        ],
        "milestone": "M8: v3 Complete",
        "body": """\
### Version

v3

### Goal

A versioned policy registry that stores and loads policies by echelon
(battalion / brigade / division) and checkpoint version.  Enables
mix-and-match evaluation between echelon versions.

### Motivation & Context

With three trained echelons and multiple checkpoints per echelon, a
systematic registry prevents the "which checkpoint are we evaluating?"
confusion.

### Child Issues (Tasks)

- [ ] Implement `training/policy_registry.py` — register / load by echelon + version
- [ ] Add CLI command `python -m training.policy_registry list`
- [ ] Integrate registry into evaluation pipeline
- [ ] Add `tests/test_policy_registry.py`

### Acceptance Criteria

- [ ] Registry loads any saved echelon checkpoint by name
- [ ] CLI `list` command prints echelon, version, path, W&B run ID
- [ ] Evaluation pipeline accepts `--battalion-policy`, `--brigade-policy`, `--division-policy` flags

### Priority

medium

### Target Milestone

M8: v3 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.7 — HRL Evaluation vs. Flat MARL",
        "labels": [
            "type: epic", "priority: high",
            "v3: evaluation", "domain: eval", "status: agent-created",
        ],
        "milestone": "M8: v3 Complete",
        "body": """\
### Version

v3

### Goal

Rigorous head-to-head evaluation of the three-echelon HRL architecture
against v2's flat MAPPO baseline.  Results answer the primary v3
research question.

### Motivation & Context

Without a controlled comparison, we cannot claim HRL is beneficial.
The evaluation must control for total training compute (same wall-clock
budget), scenario diversity, and number of agents.

### Child Issues (Tasks)

- [ ] Design evaluation protocol: same scenario set, same compute budget, 100-episode tournament
- [ ] Implement `training/evaluate_hrl.py` — end-to-end HRL evaluation harness
- [ ] Run HRL vs. flat MARL tournament at 4v4; commit as `[EXP]` issue
- [ ] Create `notebooks/v3_hrl_analysis.ipynb` with result plots

### Acceptance Criteria

- [ ] Tournament is statistically meaningful (≥ 100 episodes, bootstrapped confidence intervals)
- [ ] Results issue clearly states whether HRL outperforms flat MARL
- [ ] Analysis notebook reproducible from saved run artifacts

### Priority

high

### Target Milestone

M8: v3 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E3.8 — v3 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v3: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M8: v3 Complete",
        "body": """\
### Version

v3

### Goal

Complete documentation for all v3 HRL components, publish the v3 release,
and leave the codebase ready for v4 league training.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v3 epics complete
- [ ] Write `docs/hrl_architecture.md` — SMDP, options, echelon hierarchy
- [ ] Write `docs/hrl_training_protocol.md` — phase sequence and promotion criteria
- [ ] Add v3 architecture diagram to `docs/`
- [ ] Draft `CHANGELOG.md` v3 section
- [ ] Create GitHub release tag `v3.0.0`

### Acceptance Criteria

- [ ] All v3 epics marked complete in ROADMAP.md
- [ ] HRL architecture guide reviewed and merged
- [ ] `v3.0.0` release tag exists with release notes

### Priority

low

### Target Milestone

M8: v3 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v3 FEATURE / TASK ISSUES ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

V3_TASKS: list[dict] = [
    {
        "title": "Implement `envs/options.py` — Option dataclass with initiation, policy, termination",
        "labels": ["type: feature", "priority: high", "v3: smdp", "domain: ml", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": "Define the `Option` dataclass: `initiation_set`, `intra_option_policy`, "
                "`termination_condition`. Implement a standard library of six tactical options: "
                "`advance_sector`, `defend_position`, `flank_left`, `flank_right`, "
                "`withdraw`, `concentrate_fire`.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/smdp_wrapper.py` — Semi-MDP wrapper for MultiBattalionEnv",
        "labels": ["type: feature", "priority: high", "v3: smdp", "domain: env", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": "Wrap `MultiBattalionEnv` so that each `step()` call executes a macro-action (option) "
                "until its termination condition fires. Returns aggregate reward and temporal "
                "abstraction ratio info dict.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/brigade_env.py` — brigade-level MDP",
        "labels": ["type: feature", "priority: high", "v3: brigade", "domain: env", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": "Brigade observation: sector control percentages, battalion strengths/morales, "
                "enemy threat vector. Brigade action: assign an option to each battalion. "
                "Extends `gymnasium.Env`.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement policy freezing utility `training/utils/freeze_policy.py`",
        "labels": ["type: feature", "priority: medium", "v3: hrl-curriculum", "domain: ml", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": "Utility to load a saved policy checkpoint and freeze all parameters "
                "(`requires_grad=False`). Used when training higher-echelon policies "
                "on top of frozen lower-echelon policies.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `envs/division_env.py` — division-level MDP",
        "labels": ["type: feature", "priority: medium", "v3: division", "domain: env", "status: agent-created"],
        "milestone": "M8: v3 Complete",
        "body": "Division observation: theatre-level sector map, brigade status summaries, "
                "objective control metrics. Division action: assign a brigade-level "
                "macro-command to each brigade.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/policy_registry.py` — per-echelon versioned policy store",
        "labels": ["type: feature", "priority: medium", "v3: policy-lib", "domain: infra", "status: agent-created"],
        "milestone": "M8: v3 Complete",
        "body": "Registry backed by a JSON manifest. Supports `register(echelon, version, path, run_id)`, "
                "`load(echelon, version)`, and `list()`. CLI entry point.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/evaluate_hrl.py` — end-to-end HRL evaluation harness",
        "labels": ["type: feature", "priority: high", "v3: evaluation", "domain: eval", "status: agent-created"],
        "milestone": "M8: v3 Complete",
        "body": "Load battalion, brigade, and division policies from registry. "
                "Run a configurable number of evaluation episodes. Log per-episode "
                "results, win rate, and coordination metrics. Compare with flat MARL baseline.\n"
                + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v4 EPICS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

V4_EPICS: list[dict] = [
    {
        "title": "[EPIC] E4.1 — League Infrastructure (agent pool, matchmaking)",
        "labels": [
            "type: epic", "priority: high",
            "v4: league-infra", "domain: infra", "status: agent-created",
        ],
        "milestone": "M9: League Training",
        "body": """\
### Version

v4

### Goal

An AlphaStar-style league infrastructure: a pool of agent snapshots
(main agents, exploiters, league exploiters), a matchmaking algorithm
based on prioritized fictitious self-play (PFSP), and a database of
historical match outcomes.

### Motivation & Context

Self-play with only the latest policy converges to a Nash equilibrium
locally but can forget strategies.  A diverse league with multiple
agent types produces more robust, strategy-diverse policies.

### Child Issues (Tasks)

- [ ] Implement `training/league/agent_pool.py` — add/remove agents, PFSP sampling
- [ ] Implement `training/league/matchmaker.py` — PFSP with win-rate weighting
- [ ] Design and implement league match result database (SQLite or JSON)
- [ ] Implement agent versioning and snapshot storage
- [ ] Add `tests/test_agent_pool.py`, `tests/test_matchmaker.py`

### Acceptance Criteria

- [ ] PFSP matchmaker samples opponents proportional to historical win rates
- [ ] Agent pool supports ≥ 50 concurrent snapshots without memory issues
- [ ] Match results persist across restarts
- [ ] Unit tests cover pool add/remove, PFSP sampling distribution

### Priority

high

### Target Milestone

M9: League Training""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.2 — Main Agent Training Loop",
        "labels": [
            "type: epic", "priority: high",
            "v4: main-agent", "domain: ml", "status: agent-created",
        ],
        "milestone": "M9: League Training",
        "body": """\
### Version

v4

### Goal

Training loop for main agents using PFSP against the full league pool.
Main agents are the primary policies we wish to maximize; they play
against the league's entire historical distribution.

### Motivation & Context

Main agents trained against only self-play variants can develop blind
spots.  PFSP against the league's historical pool forces them to handle
diverse strategies.

### Child Issues (Tasks)

- [ ] Implement `training/league/train_main_agent.py` — MAPPO with PFSP opponent sampling
- [ ] Implement historical snapshot scheduling (with configurable update frequency)
- [ ] Add per-matchup win-rate tracking to W&B
- [ ] Add main-agent Elo tracking across league matches
- [ ] Create `configs/league/main_agent.yaml`

### Acceptance Criteria

- [ ] Main agent training loop stable for 5M steps without NaN
- [ ] Per-matchup win rates logged to W&B per evaluation interval
- [ ] Main agent Elo increases vs. league snapshot pool over time
- [ ] Configurable PFSP temperature parameter

### Priority

high

### Target Milestone

M9: League Training""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.3 — Main Exploiter Agents",
        "labels": [
            "type: epic", "priority: medium",
            "v4: exploiters", "domain: ml", "status: agent-created",
        ],
        "milestone": "M9: League Training",
        "body": """\
### Version

v4

### Goal

Train exploiter agents that target the latest main agent snapshot.
Exploiters maximize win rate against the main agent, exposing
weaknesses that the main agent must then correct.

### Motivation & Context

Main exploiters provide a continuous adversarial pressure that prevents
main agents from over-specializing.  This mirrors AlphaStar's league design.

### Child Issues (Tasks)

- [ ] Implement `training/league/train_exploiter.py` — train against latest main agent snapshot
- [ ] Implement exploiter reset policy (reset when win rate vs. main agent drops below threshold)
- [ ] Add exploiter win-rate vs. main agent to W&B
- [ ] Create `configs/league/main_exploiter.yaml`

### Acceptance Criteria

- [ ] Exploiter consistently exceeds 60 % win rate vs. initial main agent
- [ ] Exploiter is reset when its win rate vs. current main agent drops below 30 %
- [ ] Exploiter training runs in parallel with main agent training (separate process)

### Priority

medium

### Target Milestone

M9: League Training""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.4 — League Exploiter Agents",
        "labels": [
            "type: epic", "priority: medium",
            "v4: exploiters", "domain: ml", "status: agent-created",
        ],
        "milestone": "M9: League Training",
        "body": """\
### Version

v4

### Goal

Train league exploiter agents that target the entire historical league
(not just the main agent).  League exploiters prevent Nash equilibrium
collapse across the full agent pool.

### Motivation & Context

League exploiters ensure no historical strategy goes unchallenged,
maintaining diversity across the league and preventing exploitable
strategy voids.

### Child Issues (Tasks)

- [ ] Implement `training/league/train_league_exploiter.py` — PFSP against full historical pool
- [ ] Implement league-wide Nash exploitability metric
- [ ] Add league exploitability to W&B dashboard
- [ ] Create `configs/league/league_exploiter.yaml`

### Acceptance Criteria

- [ ] League exploitability metric decreases over first 2M league steps
- [ ] League exploiter discovers at least one qualitatively different strategy vs. main agent
- [ ] Strategy discovery documented in a `[EXP]` issue

### Priority

medium

### Target Milestone

M9: League Training""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.5 — Nash Distribution Sampling",
        "labels": [
            "type: epic", "priority: medium",
            "v4: nash", "domain: ml", "status: agent-created",
        ],
        "milestone": "M10: v4 Complete",
        "body": """\
### Version

v4

### Goal

Compute an approximate Nash equilibrium over the league's historical
agent pool and use it to sample training opponents.  This is the core
theoretical advance in AlphaStar-style league training.

### Motivation & Context

PFSP approximates Nash equilibrium but does not guarantee convergence.
Implementing explicit Nash distribution computation (via linear programming
or regret minimization) provides theoretical grounding and better sample efficiency.

### Child Issues (Tasks)

- [ ] Implement `training/league/nash.py` — Nash equilibrium approximation (regret matching or LP)
- [ ] Integrate Nash distribution into matchmaker's sampling weights
- [ ] Add Nash distribution entropy to W&B (high entropy = diverse league)
- [ ] Add `tests/test_nash.py` with known 2×2 game solutions
- [ ] Experiment: PFSP vs Nash sampling on same league; commit as `[EXP]` issue

### Acceptance Criteria

- [ ] `nash.py` correctly solves 2×2 and 3×3 test games
- [ ] Nash entropy logged to W&B per evaluation interval
- [ ] Experiment issue shows whether Nash sampling outperforms PFSP

### Priority

medium

### Target Milestone

M10: v4 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.6 — Strategy Diversity Metrics",
        "labels": [
            "type: epic", "priority: medium",
            "v4: diversity", "domain: eval", "status: agent-created",
        ],
        "milestone": "M10: v4 Complete",
        "body": """\
### Version

v4

### Goal

Quantify behavioral diversity across the league agent pool.  Diverse
leagues produce more robust main agents and prevent premature convergence
to a single strategy archetype.

### Motivation & Context

Win rate alone does not capture strategy diversity.  Without diversity
metrics, the league may converge to a single dominant strategy and
lose its adversarial pressure.

### Child Issues (Tasks)

- [ ] Define behavioral embedding: encode agent trajectory as a fixed-length vector
- [ ] Implement `training/league/diversity.py` — pairwise behavioral distance, diversity score
- [ ] Log diversity score per league evaluation cycle to W&B
- [ ] Create `notebooks/v4_diversity_analysis.ipynb`
- [ ] Add strategy clustering visualization (t-SNE of behavioral embeddings)

### Acceptance Criteria

- [ ] Diversity score is above zero throughout training (strategies differ)
- [ ] t-SNE visualization shows at least 3 distinct strategy clusters after 10M steps
- [ ] Diversity score logged per evaluation cycle to W&B

### Priority

medium

### Target Milestone

M10: v4 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.7 — Distributed Training (Ray / RLlib)",
        "labels": [
            "type: epic", "priority: medium",
            "v4: distributed", "domain: infra", "status: agent-created",
        ],
        "milestone": "M10: v4 Complete",
        "body": """\
### Version

v4

### Goal

Port the league training infrastructure to Ray/RLlib for distributed
multi-node execution.  League training requires high throughput that
exceeds single-machine capacity.

### Motivation & Context

Running 3+ concurrent agent types (main + exploiters + league exploiters)
with a large environment pool requires distributed actor-critic.  Ray is
the standard choice in RL research for this scale.

### Child Issues (Tasks)

- [ ] Add `ray[rllib]` to `requirements.txt` (check for vulnerabilities first)
- [ ] Port `MultiBattalionEnv` to a Ray remote environment
- [ ] Implement `training/league/distributed_runner.py` — Ray actor pool for rollouts
- [ ] Add `configs/distributed/ray_cluster.yaml` — worker count, GPU allocation
- [ ] Benchmark: throughput (steps/sec) single-node vs. 4-worker Ray cluster
- [ ] Add CI smoke test: 2-worker Ray cluster on `ubuntu-latest`

### Acceptance Criteria

- [ ] 4-worker Ray cluster achieves ≥ 3× throughput vs. single process
- [ ] CI smoke test passes in ≤ 5 minutes
- [ ] Checkpoint/restore compatible with non-distributed checkpoints

### Priority

medium

### Target Milestone

M10: v4 Complete""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E4.8 — v4 Documentation & Release",
        "labels": [
            "type: epic", "priority: low",
            "v4: documentation", "domain: infra", "status: agent-created",
        ],
        "milestone": "M10: v4 Complete",
        "body": """\
### Version

v4

### Goal

Complete documentation for all v4 league training components, publish
the v4 release, and prepare the codebase for v5 analysis work.

### Child Issues (Tasks)

- [ ] Update `docs/ROADMAP.md` — mark v4 epics complete
- [ ] Write `docs/league_training_guide.md` — agent types, matchmaking, Nash sampling
- [ ] Add v4 architecture diagram to `docs/`
- [ ] Draft `CHANGELOG.md` v4 section
- [ ] Create GitHub release tag `v4.0.0`

### Acceptance Criteria

- [ ] All v4 epics marked complete in ROADMAP.md
- [ ] League training guide reviewed and merged
- [ ] `v4.0.0` release tag exists with release notes

### Priority

low

### Target Milestone

M10: v4 Complete""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v4 FEATURE / TASK ISSUES ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

V4_TASKS: list[dict] = [
    {
        "title": "Implement `training/league/agent_pool.py` — PFSP opponent pool",
        "labels": ["type: feature", "priority: high", "v4: league-infra", "domain: infra", "status: agent-created"],
        "milestone": "M9: League Training",
        "body": "Agent pool stores policy snapshots with metadata (win rates, Elo, strategy tags). "
                "PFSP sampling weights opponents by inverse win rate against the current agent. "
                "Supports add, remove, and sample operations.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/league/matchmaker.py` — PFSP matchmaking",
        "labels": ["type: feature", "priority: high", "v4: league-infra", "domain: infra", "status: agent-created"],
        "milestone": "M9: League Training",
        "body": "Matchmaker selects opponents for each training step. PFSP weight: "
                "`w_i ∝ (1 - win_rate_vs_i)^temperature`. Configurable temperature and "
                "self-play fraction. Logs match pairings to W&B.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/league/nash.py` — Nash equilibrium approximation",
        "labels": ["type: feature", "priority: medium", "v4: nash", "domain: ml", "status: agent-created"],
        "milestone": "M10: v4 Complete",
        "body": "Compute approximate Nash equilibrium over the win-rate matrix using regret "
                "matching (CFR) or linear programming. Returns a probability distribution over "
                "agent pool for opponent sampling.\n" + ATTRIBUTION,
    },
    {
        "title": "Implement `training/league/diversity.py` — behavioral diversity scoring",
        "labels": ["type: feature", "priority: medium", "v4: diversity", "domain: eval", "status: agent-created"],
        "milestone": "M10: v4 Complete",
        "body": "Encode agent trajectories as fixed-length behavioral vectors (action histograms, "
                "position heatmaps, flanking ratio). Compute pairwise cosine distance and aggregate "
                "diversity score across the pool. Log to W&B.\n" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── v5 / vFuture EPICS ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

V5_EPICS: list[dict] = [
    {
        "title": "[EPIC] E5.1 — Human-Playable Interface (web / desktop)",
        "labels": [
            "type: epic", "priority: low",
            "v5: interface", "domain: viz", "status: agent-created",
        ],
        "milestone": None,
        "body": """\
### Version

v5 / vFuture

### Goal

A real-time human-vs-AI interface where a human player controls one side
and a trained policy controls the other.  Supports web (Pyodide/WebGL)
or desktop (Pygame) rendering.

### Motivation & Context

Making the trained system interactive demonstrates its tactical quality
to non-ML stakeholders and creates a compelling demo for the project.

### Child Issues (Tasks)

- [ ] Design UI wireframe: battle map, action selection, unit status panel
- [ ] Implement `envs/human_env.py` — human action input adapter
- [ ] Implement web renderer (`envs/rendering/web_renderer.py`) or Pygame frontend
- [ ] Integrate v2/v3 trained policy as AI opponent
- [ ] Add game lobby: scenario selection, difficulty (which checkpoint to load)
- [ ] Package as standalone executable or Docker image

### Acceptance Criteria

- [ ] Human can control a battalion via mouse/keyboard
- [ ] AI responds within 100 ms per step
- [ ] Game runs at ≥ 30 FPS on a mid-range laptop
- [ ] Scenario selection covers at least three configurations

### Priority

low

### Target Milestone

TBD (post-v4)""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E5.2 — Course of Action (COA) Generator",
        "labels": [
            "type: epic", "priority: low",
            "v5: coa", "domain: ml", "status: agent-created",
        ],
        "milestone": None,
        "body": """\
### Version

v5 / vFuture

### Goal

Use trained policies to generate and rank multiple courses of action
for a given scenario.  COA generator outputs a set of candidate plans
with predicted outcomes, supporting human decision-making.

### Motivation & Context

Military planners generate multiple COAs and select the best one.
A trained policy can efficiently explore this plan space and estimate
success probabilities — the core applied use case of the project.

### Child Issues (Tasks)

- [ ] Implement `analysis/coa_generator.py` — Monte-Carlo rollout of candidate plans
- [ ] Implement COA scoring: expected win rate, casualties, terrain control
- [ ] Implement COA comparison visualization (`notebooks/coa_analysis.ipynb`)
- [ ] Expose COA generator as a REST API endpoint (`api/coa_endpoint.py`)
- [ ] Validate COA generator output against historical battle decisions

### Acceptance Criteria

- [ ] Generator produces ≥ 5 distinct COAs per scenario within 60 seconds
- [ ] COAs differ in aggregate action sequences (not just stochastic noise)
- [ ] REST API returns scored COA list as JSON

### Priority

low

### Target Milestone

TBD (post-v4)""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E5.3 — Strategy Explainability (attention visualization)",
        "labels": [
            "type: epic", "priority: low",
            "v5: explainability", "domain: ml", "status: agent-created",
        ],
        "milestone": None,
        "body": """\
### Version

v5 / vFuture

### Goal

Interpret and visualize what the trained policies "see" when making
decisions.  Outputs include saliency maps, attention weights (if
transformer policy is used in v2+), and feature importance rankings.

### Motivation & Context

Explainability is critical for operator trust and for identifying
failure modes.  It also supports the historical validation work
in E5.4.

### Child Issues (Tasks)

- [ ] Implement gradient saliency maps for MLP policy (`analysis/saliency.py`)
- [ ] Implement attention visualization for transformer policy (if trained)
- [ ] Implement SHAP feature importance for observation dimensions
- [ ] Create `notebooks/explainability_demo.ipynb`
- [ ] Document observation feature meanings in `docs/observation_reference.md`

### Acceptance Criteria

- [ ] Saliency maps correlate with known tactical importance (e.g., enemy distance is high-salience)
- [ ] Feature importance identifies top-3 most influential observation dimensions
- [ ] Notebook runs end-to-end from a saved checkpoint

### Priority

low

### Target Milestone

TBD (post-v4)""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E5.4 — Historical Scenario Validation",
        "labels": [
            "type: epic", "priority: low",
            "v5: validation", "domain: eval", "status: agent-created",
        ],
        "milestone": None,
        "body": """\
### Version

v5 / vFuture

### Goal

Instantiate historical Napoleonic battle scenarios within the simulation
engine and validate whether trained policies reproduce or improve upon
the historical outcome.

### Motivation & Context

Historical validation is the ultimate stress test: do our agents
discover historically-documented tactics, or do they find novel
superior alternatives?  This connects the project to its wargaming roots.

### Child Issues (Tasks)

- [ ] Implement scenario loader (`envs/scenarios/historical.py`) — YAML-defined initial conditions
- [ ] Create at least three historical scenario YAML files (e.g., Waterloo, Austerlitz, Borodino)
- [ ] Implement outcome comparison: trained policy vs. historical record
- [ ] Create `notebooks/historical_validation.ipynb`
- [ ] Document scenario design methodology in `docs/historical_scenarios.md`

### Acceptance Criteria

- [ ] Scenario loader correctly initializes from YAML (positions, strengths, terrain)
- [ ] At least three scenarios run to completion without errors
- [ ] Outcome comparison documented in a `[EXP]` issue with conclusions

### Priority

low

### Target Milestone

TBD (post-v4)""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] E5.5 — Export Trained Policies for Deployment",
        "labels": [
            "type: epic", "priority: low",
            "v5: deployment", "domain: infra", "status: agent-created",
        ],
        "milestone": None,
        "body": """\
### Version

v5 / vFuture

### Goal

Export trained policies to portable formats (ONNX, TorchScript) for
deployment outside the Python training environment.  Support for
edge devices, browser-based inference (ONNX.js), and standalone
wargame executables.

### Motivation & Context

Trained policies locked in PyTorch/SB3 checkpoints are not reusable
outside the training environment.  Portable export enables integration
with external wargaming systems and demos.

### Child Issues (Tasks)

- [ ] Implement `scripts/export_policy.py` — ONNX and TorchScript export
- [ ] Add `tests/test_policy_export.py` — verify inference parity between PyTorch and ONNX
- [ ] Create Docker image for policy serving (`docker/policy_server/Dockerfile`)
- [ ] Document export procedure in `docs/deployment_guide.md`
- [ ] Benchmark inference latency: PyTorch vs. ONNX runtime vs. TorchScript

### Acceptance Criteria

- [ ] ONNX export produces identical outputs to PyTorch model (within 1e-5)
- [ ] TorchScript export works without tracing errors
- [ ] Docker image serves policy inference via REST API
- [ ] Inference latency < 5 ms per step on CPU

### Priority

low

### Target Milestone

TBD (post-v4)""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── Sprint Planning Issues ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

SPRINT_ISSUES: list[dict] = [
    {
        "title": "[EPIC] Sprint S10 — v2 Kickoff: PettingZoo Environment",
        "labels": ["type: epic", "priority: high", "v2: environment", "domain: env", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Sprint S10 (Weeks 19–20)

**Goal:** PettingZoo `MultiBattalionEnv` running 2v2 battles end-to-end.

### Deliverables

- [ ] `envs/multi_battalion_env.py` — complete PettingZoo `ParallelEnv`
- [ ] Per-agent observation space defined and normalized
- [ ] Global state tensor exposed via `state()` method
- [ ] PettingZoo API test passing in CI
- [ ] `tests/test_multi_battalion_env.py` with ≥ 10 test cases

### Exit Criteria

PettingZoo API test passes; 2v2 episode runs to completion without errors.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S11 — MAPPO Implementation",
        "labels": ["type: epic", "priority: high", "v2: mappo", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Sprint S11 (Weeks 21–22)

**Goal:** MAPPO training loop functional; first 2v2 training run logged to W&B.

### Deliverables

- [ ] `models/mappo_policy.py` — shared actor + centralized critic
- [ ] `training/train_mappo.py` — MAPPO loop
- [ ] `configs/experiment_mappo_2v2.yaml`
- [ ] W&B run with per-agent reward curves

### Exit Criteria

`python training/train_mappo.py` runs 100k steps without NaN; W&B run created.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S12 — 2v2 Curriculum & Coordination Metrics",
        "labels": ["type: epic", "priority: medium", "v2: curriculum", "v2: metrics", "domain: ml", "status: agent-created"],
        "milestone": "M5: 2v2 MARL",
        "body": """\
### Sprint S12 (Weeks 23–24)

**Goal:** Curriculum scheduler functional; coordination metrics logged to W&B.

### Deliverables

- [ ] `training/curriculum_scheduler.py`
- [ ] `envs/metrics/coordination.py` (flanking, fire concentration, mutual support)
- [ ] Scenario configs: `2v1.yaml`, `2v2.yaml`
- [ ] Curriculum ablation `[EXP]` issue filed

### Exit Criteria

Curriculum promotes 1v1 → 2v1 → 2v2 automatically; all coordination metrics logged.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S13 — NvN Scaling & Multi-Agent Self-Play",
        "labels": ["type: epic", "priority: medium", "v2: scaling", "v2: self-play", "domain: ml", "status: agent-created"],
        "milestone": "M6: v2 Complete",
        "body": """\
### Sprint S13 (Weeks 25–26)

**Goal:** Environment scales to 6v6; team Elo implemented; self-play loop running.

### Deliverables

- [ ] `n_blue`/`n_red` parameterization complete
- [ ] `training/elo.py` — team Elo
- [ ] `training/self_play.py` extended for team play
- [ ] 4v4 training run logged to W&B

### Exit Criteria

6v6 env steps without errors; self-play Elo increases monotonically for 500k steps.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S14 — v2 Polish & Release",
        "labels": ["type: epic", "priority: low", "v2: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M6: v2 Complete",
        "body": """\
### Sprint S14 (Weeks 27–28)

**Goal:** v2 documentation complete; `v2.0.0` tag created; v3 issues seeded.

### Deliverables

- [ ] `docs/multi_agent_guide.md`
- [ ] ROADMAP.md v2 section marked complete
- [ ] `CHANGELOG.md` v2 section
- [ ] `v2.0.0` GitHub release
- [ ] v3 epics seeded via `seed-v3-issues` workflow

### Exit Criteria

`v2.0.0` release tag exists; v3 epics visible in GitHub Issues.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S15 — v3 Kickoff: SMDP Framework",
        "labels": ["type: epic", "priority: high", "v3: smdp", "domain: ml", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": """\
### Sprint S15 (Weeks 29–30)

**Goal:** SMDP wrapper and macro-action vocabulary complete; integration test passing.

### Deliverables

- [ ] `envs/options.py` with six macro-actions
- [ ] `envs/smdp_wrapper.py`
- [ ] `tests/test_smdp_wrapper.py`
- [ ] HRL training protocol doc drafted

### Exit Criteria

SMDP wrapper passes PettingZoo API test; six macro-actions produce distinct primitive patterns.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S16 — Brigade Commander Layer",
        "labels": ["type: epic", "priority: high", "v3: brigade", "v3: hrl-curriculum", "domain: ml", "status: agent-created"],
        "milestone": "M7: HRL Battalion→Brigade",
        "body": """\
### Sprint S16 (Weeks 31–32)

**Goal:** Brigade commander trained on frozen v2 battalion policies; beats scripted 2v2 baseline.

### Deliverables

- [ ] `envs/brigade_env.py`
- [ ] `training/train_brigade.py`
- [ ] `training/utils/freeze_policy.py`
- [ ] Brigade win rate > v2 MAPPO baseline (documented in `[EXP]` issue)

### Exit Criteria

Brigade PPO converges; battalion weights unchanged; win rate documented.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S17 — Division Commander & HRL Evaluation",
        "labels": ["type: epic", "priority: medium", "v3: division", "v3: evaluation", "domain: ml", "status: agent-created"],
        "milestone": "M8: v3 Complete",
        "body": """\
### Sprint S17 (Weeks 33–34)

**Goal:** Division layer complete; HRL vs. flat MARL tournament run; results documented.

### Deliverables

- [ ] `envs/division_env.py`
- [ ] `training/train_division.py`
- [ ] `training/evaluate_hrl.py`
- [ ] HRL vs. flat MARL `[EXP]` issue filed with conclusions

### Exit Criteria

Three-echelon chain executes end-to-end; experiment issue merged with conclusion.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S18 — v3 Polish & Release",
        "labels": ["type: epic", "priority: low", "v3: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M8: v3 Complete",
        "body": """\
### Sprint S18 (Weeks 35–36)

**Goal:** v3 documentation complete; `v3.0.0` tag created; v4 issues seeded.

### Deliverables

- [ ] `docs/hrl_architecture.md`
- [ ] `docs/hrl_training_protocol.md`
- [ ] ROADMAP.md v3 section marked complete
- [ ] `v3.0.0` GitHub release
- [ ] v4 epics seeded

### Exit Criteria

`v3.0.0` release tag exists; v4 epics visible in GitHub Issues.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S19 — v4 Kickoff: League Infrastructure",
        "labels": ["type: epic", "priority: high", "v4: league-infra", "domain: infra", "status: agent-created"],
        "milestone": "M9: League Training",
        "body": """\
### Sprint S19 (Weeks 37–38)

**Goal:** League agent pool, PFSP matchmaker, and match database operational.

### Deliverables

- [ ] `training/league/agent_pool.py`
- [ ] `training/league/matchmaker.py`
- [ ] Match result database schema and implementation
- [ ] Unit tests: pool, matchmaker

### Exit Criteria

PFSP matchmaker correctly samples opponents; pool supports ≥ 50 snapshots.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S20 — Main Agent + Exploiter Training",
        "labels": ["type: epic", "priority: high", "v4: main-agent", "v4: exploiters", "domain: ml", "status: agent-created"],
        "milestone": "M9: League Training",
        "body": """\
### Sprint S20 (Weeks 39–40)

**Goal:** Main agent and main exploiter training loops running in parallel; Elo increasing.

### Deliverables

- [ ] `training/league/train_main_agent.py`
- [ ] `training/league/train_exploiter.py`
- [ ] W&B per-matchup win rates + Elo
- [ ] Main agent Elo increases vs. exploiter over 1M steps

### Exit Criteria

Both training loops stable for 1M steps; exploiter achieves > 60 % vs. initial main agent.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S21 — Nash Sampling & Strategy Diversity",
        "labels": ["type: epic", "priority: medium", "v4: nash", "v4: diversity", "domain: ml", "status: agent-created"],
        "milestone": "M10: v4 Complete",
        "body": """\
### Sprint S21 (Weeks 41–42)

**Goal:** Nash equilibrium approximation integrated into matchmaker; diversity scoring operational.

### Deliverables

- [ ] `training/league/nash.py`
- [ ] `training/league/diversity.py`
- [ ] Nash entropy + diversity score logged to W&B
- [ ] PFSP vs. Nash sampling `[EXP]` issue

### Exit Criteria

Nash solver passes unit tests on 2×2 and 3×3 games; diversity score > 0 throughout training.
""" + ATTRIBUTION,
    },
    {
        "title": "[EPIC] Sprint S22 — Distributed Training & v4 Release",
        "labels": ["type: epic", "priority: medium", "v4: distributed", "v4: documentation", "domain: infra", "status: agent-created"],
        "milestone": "M10: v4 Complete",
        "body": """\
### Sprint S22 (Weeks 43–44)

**Goal:** Ray/RLlib distributed training running; `v4.0.0` released.

### Deliverables

- [ ] `training/league/distributed_runner.py`
- [ ] `configs/distributed/ray_cluster.yaml`
- [ ] CI smoke test: 2-worker Ray cluster
- [ ] `docs/league_training_guide.md`
- [ ] `v4.0.0` GitHub release

### Exit Criteria

4-worker Ray cluster achieves ≥ 3× throughput; `v4.0.0` tag exists.
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# Flat list of all issues to create
# ---------------------------------------------------------------------------

ALL_ISSUES: list[dict] = (
    V2_EPICS
    + V2_TASKS
    + V3_EPICS
    + V3_TASKS
    + V4_EPICS
    + V4_TASKS
    + V5_EPICS
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
    """Return a normalized (lower-stripped) set of all non-PR issue titles (open + closed).

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

    # Resolve labels; warn and skip any that are not found on this repository.
    label_objects = []
    for label_name in issue_def.get("labels", []):
        try:
            label_objects.append(repo.get_label(label_name))
        except Exception:
            print(f"    [warn] Label not found: {label_name} — skipping")

    # Resolve milestone
    milestone_obj = None
    if issue_def.get("milestone"):
        milestone_obj = get_milestone_by_title(repo, issue_def["milestone"])
        if milestone_obj is None:
            print(f"    [warn] Milestone not found: {issue_def['milestone']} — issue will have no milestone")

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
        ("── v2 Epics", V2_EPICS),
        ("── v2 Tasks", V2_TASKS),
        ("── v3 Epics", V3_EPICS),
        ("── v3 Tasks", V3_TASKS),
        ("── v4 Epics", V4_EPICS),
        ("── v4 Tasks", V4_TASKS),
        ("── v5/vFuture Epics", V5_EPICS),
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
    raise SystemExit(run())
