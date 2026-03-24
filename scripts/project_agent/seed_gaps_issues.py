"""Seed gap-filling and improvement issues discovered by a full project audit.

Each issue targets a specific gap identified in the codebase: missing CI/CD
pipelines, incomplete integrations, placeholder implementations, security
concerns, documentation shortfalls, and operational blind spots.

Idempotent — skips any issue whose title already exists.

Run via the seed-gaps-issues GitHub Actions workflow or locally with:
    GITHUB_TOKEN=... REPO_NAME=owner/repo python seed_gaps_issues.py
    GITHUB_TOKEN=... REPO_NAME=owner/repo DRY_RUN=true python seed_gaps_issues.py
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
    "> 🤖 *Strategist Forge* created this issue automatically as part of the "
    "gap-filling project audit.\n"
    f"> Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n"
    "> Label `status: agent-created` was requested; it may be absent if the "
    "label does not exist on this repository.\n"
)

# ---------------------------------------------------------------------------
# ── CI / CD GAPS ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

CICD_ISSUES: list[dict] = [
    {
        "title": "[CHORE] Add pytest + coverage CI workflow",
        "labels": [
            "type: chore", "priority: high",
            "domain: infra", "status: agent-created",
        ],
        "milestone": "M0: Project Bootstrap",
        "body": """\
### Summary

There is no GitHub Actions workflow that runs the test suite automatically on
pull requests or pushes to `main`.  The project has ~75 test files but no
automated gate that prevents regressions from landing.

### Gap

`pytest` is the test runner and `tests/` contains comprehensive coverage, but
no `.github/workflows/` file invokes it.  Any contributor breaking existing
tests would not receive CI feedback.

### Proposed Work

- [ ] Add `.github/workflows/ci.yml` that runs on `push` and `pull_request`
- [ ] Execute `pytest tests/ -q --tb=short` in the workflow
- [ ] Upload coverage report using `pytest-cov` + `codecov` (or GitHub artifact)
- [ ] Add a status badge to `README.md`
- [ ] Cache pip dependencies to keep the workflow fast

### Acceptance Criteria

- [ ] Workflow triggers automatically on every PR targeting `main`
- [ ] Workflow fails if any test fails
- [ ] Coverage summary is visible in the PR checks panel
- [ ] Badge in `README.md` reflects current `main` status
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add performance-regression CI check",
        "labels": [
            "type: chore", "priority: medium",
            "domain: infra", "domain: eval", "status: agent-created",
        ],
        "milestone": "M4: v1 Complete",
        "body": """\
### Summary

There is no automated check that detects inference-latency or throughput
regressions between commits.  Slow-downs in the simulation step or policy
forward-pass can go undetected indefinitely.

### Gap

`benchmarks/` contains `wargames_bench.py`, while `training/` contains
`historical_benchmark.py` and `transfer_benchmark.py`, but none are wired
into CI.  There are no published baseline numbers to compare against.

### Proposed Work

- [ ] Add `.github/workflows/benchmark.yml` triggered on `push` to `main`
  and manually via `workflow_dispatch`
- [ ] Run `wargames_bench.py` and record step-time p50/p95 metrics
- [ ] Store results as a GitHub Actions artifact and compare against the
  last saved baseline (fail if p95 regresses > 20 %)
- [ ] Publish benchmark summary as a PR comment via `actions/github-script`

### Acceptance Criteria

- [ ] Workflow produces a JSON artifact with latency statistics per env
- [ ] PR fails the benchmark gate if step-time p95 regresses more than 20 %
- [ ] Baseline can be updated via a manual `workflow_dispatch` with
  `update_baseline=true` input
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add Docker build + push to CI",
        "labels": [
            "type: chore", "priority: medium",
            "domain: infra", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

`docker/` contains Dockerfiles but building and pushing images is a manual
process.  There is no CI workflow that validates Docker builds on PRs or
publishes tagged images on releases.

### Gap

Inconsistent image builds mean the latest `main` may not be reflected in any
published Docker image.  Deployment is entirely manual.

### Proposed Work

- [ ] Add `.github/workflows/docker.yml`
  - On PR: `docker build` (no push) to validate the Dockerfile compiles
  - On push to `main`: build and push `ghcr.io/<owner>/wargames_training:latest`
  - On tag `v*`: push versioned tag as well
- [ ] Use `docker/build-push-action` with layer caching
- [ ] Add `DOCKER_README.md` documenting how to pull and run the image

### Acceptance Criteria

- [ ] PRs receive a Docker build check — broken Dockerfiles fail the PR
- [ ] `ghcr.io/<owner>/wargames_training:latest` is always in sync with `main`
- [ ] Release tags produce a corresponding versioned Docker image
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── SECURITY GAPS ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

SECURITY_ISSUES: list[dict] = [
    {
        "title": "[BUG] REST API endpoints have no authentication layer",
        "labels": [
            "type: bug", "priority: high",
            "domain: infra", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

`api/coa_endpoint.py` and `server/game_server.py` expose HTTP/WebSocket
endpoints with no authentication, rate limiting, or input validation.  Any
client on the network can submit arbitrary requests.

### Gap

- No API key, JWT, or session token check on any route
- No rate limiting (trivial to DoS or exhaust the inference backend)
- No input schema validation beyond basic type checks

### Proposed Work

- [ ] Add an API-key middleware to `api/coa_endpoint.py` (header
  `X-API-Key`) — key loaded from environment variable
- [ ] Add rate limiting with `flask-limiter` (or equivalent)
- [ ] Add request body JSON schema validation using `pydantic` or
  `marshmallow`
- [ ] Document the auth scheme in `docs/deployment.md`
- [ ] Add tests for rejected unauthenticated requests

### Acceptance Criteria

- [ ] Requests without a valid `X-API-Key` header receive HTTP 401
- [ ] Requests exceeding the rate limit receive HTTP 429
- [ ] Malformed request bodies receive HTTP 422 with a descriptive error
- [ ] Existing tests still pass with auth disabled via an env flag
  (`API_AUTH_DISABLED=true`) for local development
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── MODEL / POLICY GAPS ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

MODEL_ISSUES: list[dict] = [
    {
        "title": "[FEATURE] Implement TransformerPolicy (entity-based attention)",
        "labels": [
            "type: feature", "priority: high",
            "domain: ml", "v8: transformer", "status: agent-created",
        ],
        "milestone": "M17: Transformer Policy",
        "body": """\
### Summary

`models/transformer_policy.py` is a placeholder skeleton with no working
implementation.  The `v8` roadmap explicitly requires an entity-based
transformer policy, and `M17: Transformer Policy` is a named milestone, yet
the file contains only structural scaffolding.

### Gap

The file exists but running inference through it would raise
`NotImplementedError` or produce nonsense outputs because the attention
layers and positional embeddings are not implemented.

### Proposed Work

- [ ] Implement entity tokenisation: each unit becomes a feature vector
  (position, orientation, strength, morale, type embedding)
- [ ] Implement a standard multi-head self-attention encoder (4–8 heads,
  2–4 layers, `d_model=128`)
- [ ] Add a recurrent memory cell (GRU) on top of the transformer output
  for partial observability
- [ ] Wire the encoder into the SB3 policy API (matching the interface of
  `MlpPolicy` / `RecurrentPolicy`)
- [ ] Add `tests/test_transformer_policy.py` covering forward-pass shapes,
  batch handling, and reproducibility under fixed seeds
- [ ] Benchmark inference latency vs. `MlpPolicy` baseline

### Acceptance Criteria

- [ ] `TransformerPolicy.predict(obs)` returns correctly shaped actions
- [ ] Latency ≤ 2× that of `MlpPolicy` at batch size 256
- [ ] Policy can be serialized/deserialized via SB3 `save`/`load`
- [ ] All new tests pass in CI
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Integrate PolicyRegistry with W&B Artifacts for model versioning",
        "labels": [
            "type: feature", "priority: medium",
            "domain: ml", "domain: infra", "v5: deployment", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

`training/policy_registry.py` manages local policy checkpoints but has no
integration with W&B Artifacts or any remote model store.  There is no
way to reproduce a specific policy version by name across machines.

### Gap

Policies are referenced by local file path only.  If the checkpoint
directory is deleted or the machine changes, policy history is lost.
League training and historical benchmarks require a stable, addressable
policy archive.

### Proposed Work

- [ ] Extend `PolicyRegistry` with `push_artifact(run_id, path)` and
  `pull_artifact(alias)` methods backed by `wandb.Artifact`
- [ ] Auto-push new best policies from `train.py` callbacks
- [ ] Add a `policy_registry list` CLI sub-command
- [ ] Version policies semantically: `v1-ppo-baseline`, `v2-mappo-2v2`, etc.
- [ ] Add `tests/test_policy_registry_wandb.py` (mock W&B)

### Acceptance Criteria

- [ ] A policy saved in one run can be retrieved by alias in a separate process
- [ ] The registry falls back gracefully when W&B is unavailable (offline mode)
- [ ] Existing `PolicyRegistry` unit tests still pass
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Add batch inference endpoint to policy server",
        "labels": [
            "type: feature", "priority: medium",
            "domain: infra", "v5: deployment", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

`server/game_server.py` supports single-observation inference only.  At
scale (distributed self-play, league training with many workers) this
creates a throughput bottleneck because each worker occupies a full
round-trip for a single timestep prediction.

### Gap

No batch prediction endpoint exists; no async inference queue.

### Proposed Work

- [ ] Add a `/predict_batch` HTTP endpoint that accepts a list of
  observations and returns a list of actions in a single call
- [ ] Add an async prediction queue backed by `asyncio` or `ray`
- [ ] Implement dynamic batching: accumulate requests up to N observations
  or T milliseconds, whichever comes first, then run a single
  `policy.predict` call
- [ ] Document the new endpoint in `docs/deployment.md`
- [ ] Add latency and throughput benchmarks comparing single vs. batched

### Acceptance Criteria

- [ ] `/predict_batch` handles up to 256 observations per call
- [ ] Throughput improvement vs. single-call baseline ≥ 4× at batch=64
- [ ] Single-call endpoint remains unchanged (backward compatible)
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── SIMULATION INTEGRATION GAPS ───────────────────────────────────────────
# ---------------------------------------------------------------------------

SIM_INTEGRATION_ISSUES: list[dict] = [
    {
        "title": "[FEATURE] Integrate supply/logistics into environment reward and unit degradation",
        "labels": [
            "type: feature", "priority: high",
            "domain: sim", "domain: env", "v6: simulation", "status: agent-created",
        ],
        "milestone": "M14: v6 Complete",
        "body": """\
### Summary

`envs/sim/logistics.py` implements logistics effects (e.g., fatigue and ammo
modifiers, wagon damage) that are already applied in `BattalionEnv` behind the
`enable_logistics` flag. `envs/sim/supply_network.py` implements supply
consumption and network routing, but the supply network state is not yet
consistently coupled to environment rewards and observations across envs.

### Gap

Logistics effects are partially integrated at the battalion level, but the
overall supply network is still weakly integrated with the RL training loop.
Agents have only limited incentive to manage supply lines because supply state
is not explicitly reflected in reward terms or exposed as a first-class
observation feature, and higher-level envs do not yet propagate network state.

### Proposed Work

- [ ] Wire `SupplyNetwork.step()` into `BattalionEnv._step_sim()` so that
  units degrade in combat effectiveness when out of supply
- [ ] Add a `supply_penalty` reward term (negative reward proportional to
  undersupply fraction) to `RewardConfig`
- [ ] Expose supply level as an observation feature (with
  `enable_logistics` flag, matching the existing `enable_weather` pattern)
- [ ] Update `BrigadeEnv`/`DivisionEnv`/`CorpsEnv` to propagate supply
  network state through the HRL hierarchy
- [ ] Add integration tests verifying that an unfed battalion loses
  combat effectiveness over time

### Acceptance Criteria

- [ ] `BattalionEnv` with `enable_logistics=True` shows unit strength
  degradation when disconnected from supply
- [ ] Reward includes a configurable supply-penalty term
- [ ] Observation space correctly extends by 3 dims when logistics enabled
  (matching the existing pattern for weather/formations)
- [ ] All existing env tests still pass
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Complete weather visibility and temporal dynamics integration",
        "labels": [
            "type: feature", "priority: medium",
            "domain: sim", "v6: simulation", "status: agent-created",
        ],
        "milestone": "M13: Physics Simulation",
        "body": """\
### Summary

`envs/sim/weather.py` defines weather states (Rain, Fog, Snow, Clear) and
nominal effect magnitudes. These are already wired into movement speed and
accuracy (and associated morale stressors) when `enable_weather=True`, but
visibility radius and temporal dynamics are still only partially integrated.

### Gap

Weather is observable in the agent's observation (when `enable_weather=True`)
and already influences movement and accuracy, but it does not yet affect
fog-of-war / visibility radius and typically remains static over an entire
episode. This leaves weather underutilized as a tactical signal and limits
the richness of scenarios the environment can present.

### Proposed Work

- [ ] Apply `WeatherState.visibility_modifier` to the fog-of-war radius
  in `BattalionEnv._compute_visibility()`
- [ ] Add a weather transition model (e.g., simple Markov chain) so weather
  can change during an episode
- [ ] Ensure weather transitions are consistently reflected in observations
  and any reward components that depend on detection or engagement ranges
- [ ] Add tests verifying that adverse weather (e.g., Rain, Fog, Snow)
  reduces effective visibility compared to Clear and that transitions occur
  according to the configured model

### Acceptance Criteria

- [ ] Fog weather reduces unit visibility radius as configured by
  `WeatherState.visibility_modifier`
- [ ] Weather transitions follow the configured Markov matrix during an
  episode
- [ ] All existing tests pass; weather is still optional
  (`enable_weather=False` keeps existing behaviour)
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── CONCURRENCY / CORRECTNESS GAPS ────────────────────────────────────────
# ---------------------------------------------------------------------------

CORRECTNESS_ISSUES: list[dict] = [
    {
        "title": "[BUG] AgentPool has concurrent-writer race condition on JSON checkpoint file",
        "labels": [
            "type: bug", "priority: high",
            "domain: infra", "v4: league-infra", "status: agent-created",
        ],
        "milestone": "M9: League Training",
        "body": """\
### Summary

`training/agent_pool.py` writes pool state to a JSON file via a `.tmp`
rename pattern, but the pattern is not safe for concurrent writers across
multiple processes (distributed league workers).  Two workers saving
simultaneously can produce a corrupted pool file.

### Gap

The `.tmp`-then-rename write is safe only when a single writer is
guaranteed.  In distributed league training with Ray, multiple workers
may call `save()` simultaneously, leading to lost updates or corrupt JSON.

### Proposed Work

- [ ] Replace the bare rename with a file lock (e.g., `filelock.FileLock`)
  around the read-modify-write cycle
- [ ] Alternatively, migrate pool persistence to a thread-safe/process-safe
  backend: SQLite (via `sqlite3` WAL mode) or Redis (if already available)
- [ ] Add a stress test that spawns 8 processes all writing to the same
  pool file concurrently and verifies final state integrity
- [ ] Document the concurrency model in the `AgentPool` class docstring

### Acceptance Criteria

- [ ] 100 concurrent `save()` calls produce a valid, uncorrupted pool file
- [ ] No data loss when two workers update different agent entries
  simultaneously
- [ ] Existing `AgentPool` unit tests still pass
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── DOCUMENTATION GAPS ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DOCS_ISSUES: list[dict] = [
    {
        "title": "[CHORE] Update README to reflect current project state (v5+, not v1)",
        "labels": [
            "type: chore", "priority: medium",
            "domain: infra", "v1: documentation", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

Portions of `README.md` (versioning, roadmap, and Quick Start sections)
appear out of date relative to the current v5+ training configs, environments,
and automation scripts present in this repository.

### Gap

New contributors and potential collaborators may get an outdated impression of
project maturity and current workflows.  This also increases the risk that the
"Quick Start" instructions and example commands will not match the actual
installed modules and CLI entry-points.

### Proposed Work

- [ ] Update the version badge / header to reflect the current milestone
- [ ] Revise the **Overview** section to list v1–v5 as implemented
- [ ] Update the **Quick Start** commands to match current CLI entry-points
- [ ] Add a **Changelog / Release History** summary table at the top
- [ ] Verify all links (docs, CI badges, W&B project) are not broken
- [ ] Add a license section (see also: add LICENSE file issue)

### Acceptance Criteria

- [ ] README versioning, roadmap, and Quick Start sections accurately
  reflect the current v5+ milestone and implemented features
- [ ] All code snippets in the README execute without error on a fresh
  virtualenv install
- [ ] Docs link checker (if any) reports zero broken links
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add LICENSE file to the repository",
        "labels": [
            "type: chore", "priority: medium",
            "domain: infra", "status: agent-created",
        ],
        "milestone": "M0: Project Bootstrap",
        "body": """\
### Summary

The repository has no `LICENSE` file.  `README.md` contains a placeholder
comment "Add license here."  Without an explicit license, the project is
legally closed-source by default, blocking external contributions and
downstream use.

### Gap

GitHub's "choose a license" tool and SPDX recommend adding a license at
project inception.  The lack of one is a blocker for academic citation,
open-source packaging, and collaboration.

### Proposed Work

- [ ] Decide on the appropriate license (e.g., MIT, Apache-2.0, GPL-3.0)
- [ ] Add the `LICENSE` file to the repository root
- [ ] Add an SPDX identifier comment (`# SPDX-License-Identifier: MIT`) to
  all Python module headers if MIT is chosen
- [ ] Update `README.md` license section with the chosen license name + badge
- [ ] Update `pyproject.toml` `[project] license` field

### Acceptance Criteria

- [ ] `LICENSE` file exists in the repository root
- [ ] `README.md` displays a license badge
- [ ] `pyproject.toml` references the license SPDX identifier
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Write formal API reference documentation (MkDocs)",
        "labels": [
            "type: chore", "priority: medium",
            "domain: infra", "v5: deployment", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

The project has a `docs/` directory with MkDocs configuration, but there is
no auto-generated API reference.  Users must read source code to understand
module APIs.

### Gap

All four top-level packages (`envs`, `models`, `training`, `analysis`) have
comprehensive `__all__` lists and docstrings, making auto-generation
straightforward, but `mkdocs.yml` does not include a `mkdocstrings` plugin
configuration.

### Proposed Work

- [ ] Add `mkdocstrings[python]` to `requirements.txt` (dev extras)
- [ ] Configure `mkdocs.yml` to use `mkdocstrings` with
  `google`-style docstring parsing
- [ ] Create `docs/api/` pages for `envs`, `models`, `training`, `analysis`
- [ ] Wire the `docs.yml` GitHub Actions workflow to build and deploy to
  GitHub Pages
- [ ] Add a "See the API reference" link from `README.md`

### Acceptance Criteria

- [ ] `mkdocs build --strict` produces no warnings about missing symbols
- [ ] Deployed GitHub Pages site includes browsable API reference for all
  four top-level packages
- [ ] Every public class/function in `__all__` has at least a one-line
  summary docstring (CI lint checks this)
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add end-to-end tutorial: league training and strategy diversity",
        "labels": [
            "type: chore", "priority: low",
            "domain: infra", "v4: documentation", "status: agent-created",
        ],
        "milestone": "M10: v4 Complete",
        "body": """\
### Summary

`docs/` has three tutorials (battalion training, self-play, HRL) but no
tutorial covering v4 league training, Nash distribution sampling, or the
strategy diversity metrics.

### Gap

League training is the most complex subsystem and the one most likely to
confuse new researchers.  Without a worked example, the `training/league/`
directory is opaque.

### Proposed Work

- [ ] Write `docs/tutorials/league_training.md` with step-by-step
  instructions: bootstrapping the agent pool, configuring the matchmaker,
  running the main agent + exploiter loop, and reading the diversity report
- [ ] Include a minimal reproducible example that completes in < 5 minutes
  on CPU (small env, few steps)
- [ ] Add a "Troubleshooting" section for common league failures
  (e.g., pool file corruption, Nash solver divergence)
- [ ] Link the tutorial from `README.md` and `mkdocs.yml`

### Acceptance Criteria

- [ ] Tutorial renders correctly in MkDocs (`mkdocs build --strict`)
- [ ] All code snippets in the tutorial execute without errors
- [ ] Tutorial is linked from the main navigation bar
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── ANALYSIS & EVALUATION GAPS ────────────────────────────────────────────
# ---------------------------------------------------------------------------

ANALYSIS_ISSUES: list[dict] = [
    {
        "title": "[FEATURE] Run end-to-end v1 training to convergence and publish baseline results",
        "labels": [
            "type: experiment", "priority: critical",
            "v1: training", "v1: evaluation", "status: agent-created",
        ],
        "milestone": "M1: 1v1 Competence",
        "body": """\
### Summary

No trained policy artifacts exist.  All capability claims are hypothetical.
There is no empirical evidence that the v1 training pipeline produces a
competent agent, nor any published baseline Elo or win-rate numbers.

### Gap

Without a verified trained policy the following are all unvalidated:
- PPO hyperparameters in `configs/default.yaml`
- Reward shaping in `envs/battalion_env.py`
- Self-play curriculum in `training/self_play.py`
- Scripted opponent difficulty levels (`scripted_l1`–`scripted_l5`)

### Proposed Work

- [ ] Run `training/train.py` to completion (≥ 5 M steps) on `M1: 1v1`
  configuration
- [ ] Log all metrics to W&B; link the run in this issue
- [ ] Evaluate the checkpoint with `training/evaluate.py` against all five
  scripted levels; record win rates
- [ ] Commit the best checkpoint to `agent-artifacts/v1/`
- [ ] Add a **Baseline Results** section to `README.md` with a win-rate
  table

### Acceptance Criteria

- [ ] W&B training run link provided in this issue
- [ ] Agent achieves ≥ 80 % win rate against `scripted_l3`
- [ ] `agent-artifacts/v1/best_model.zip` committed and accessible
- [ ] Baseline win-rate table present in `README.md`
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Validate trained policy against historical battle outcomes",
        "labels": [
            "type: experiment", "priority: medium",
            "v5: validation", "domain: eval", "status: agent-created",
        ],
        "milestone": "M11: Interface & Analysis",
        "body": """\
### Summary

`training/historical_benchmark.py` includes an `OutcomeComparator` class
but it has never been applied to an actual trained policy.  There is no
empirical measurement of how well the agent reproduces historical Napoleonic
battle outcomes.

### Gap

Historical validation is a stated project goal (v5 roadmap) and is required
before the project can be cited in any academic or operational context.
Without it, the "historical realism" claim is purely aspirational.

### Proposed Work

- [ ] Select 3 battles from `data/historical/battles.json` with known
  outcomes (win/loss, casualty ratios)
- [ ] Load the best v1 policy and run 50 episodes per battle in the
  corresponding GIS scenario
- [ ] Compare simulated casualty ratios and battle duration to historical
  records using `OutcomeComparator`
- [ ] Document discrepancies and hypothesize causes (missing
  logistics, weather, formation effects)
- [ ] Publish results in `docs/analysis/historical_validation.md`

### Acceptance Criteria

- [ ] Validation report committed to `docs/analysis/`
- [ ] Simulated casualty ratios are within 2× of historical values for
  ≥ 2 of the 3 selected battles
- [ ] Discrepancies are analysed and linked to specific missing sim features
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Publish hyperparameter sweep results and optimal config",
        "labels": [
            "type: experiment", "priority: medium",
            "v1: training", "domain: ml", "status: agent-created",
        ],
        "milestone": "M4: v1 Complete",
        "body": """\
### Summary

`configs/default.yaml` contains PPO hyperparameters but no rationale or
sweep results.  W&B sweeps have never been executed, so there is no
empirical basis for the chosen values.

### Gap

Sub-optimal hyperparameters can easily halve sample efficiency.  Without a
sweep the current config is effectively a guess.

### Proposed Work

- [ ] Define a W&B sweep config (`configs/sweep.yaml`) covering:
  `learning_rate` (log-uniform 1e-5–1e-3), `n_steps` (128/256/512/1024),
  `batch_size` (64/128/256), `gamma` (0.95/0.99/0.995),
  `ent_coef` (0.0/0.005/0.01)
- [ ] Run ≥ 50 sweep trials (Bayesian optimisation) against `scripted_l3`
  eval
- [ ] Commit the best config to `configs/best_sweep.yaml` and link the W&B
  sweep in this issue
- [ ] Add a **Hyperparameter Tuning** section to `docs/training.md`

### Acceptance Criteria

- [ ] W&B sweep link provided in this issue
- [ ] `configs/best_sweep.yaml` committed with the best-found values
- [ ] Best config beats default config by ≥ 5 % win rate against
  `scripted_l3`
- [ ] Sweep methodology documented in `docs/training.md`
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── FRONTEND / UX GAPS ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

FRONTEND_ISSUES: list[dict] = [
    {
        "title": "[FEATURE] Build interactive COA comparison dashboard (frontend)",
        "labels": [
            "type: feature", "priority: medium",
            "domain: viz", "v5: coa", "status: agent-created",
        ],
        "milestone": "M11: Interface & Analysis",
        "body": """\
### Summary

`api/coa_endpoint.py` generates course-of-action (COA) analysis results but
there is no visual front-end to display, compare, or annotate them.  The
React app in `frontend/` has a `COAPanel` component stub but it is not
connected to the API.

### Gap

COA results are JSON blobs accessible only via `curl` or the raw API.
Decision-makers and researchers cannot visually compare alternative courses
of action side-by-side.

### Proposed Work

- [ ] Wire `frontend/src/components/COAPanel.jsx` to the `/api/coa/analyze`
  endpoint
- [ ] Display each COA as an overlay on the map canvas (coloured
  manoeuvre arrows)
- [ ] Add a side-by-side comparison mode (2-column layout)
- [ ] Show key metrics per COA: expected casualties, objective capture
  probability, supply risk
- [ ] Add a "Select COA" button that loads the selected COA into the
  scenario editor

### Acceptance Criteria

- [ ] Dashboard renders ≥ 2 COA options simultaneously on the map
- [ ] Metric cards update in real-time when the scenario is modified
- [ ] "Select COA" correctly populates the order queue
- [ ] Mobile-responsive layout (min width 768 px)
""" + ATTRIBUTION,
    },
    {
        "title": "[FEATURE] Replace Pygame HumanEnv with browser-based WebSocket interface",
        "labels": [
            "type: feature", "priority: medium",
            "domain: viz", "v5: interface", "status: agent-created",
        ],
        "milestone": "M11: Interface & Analysis",
        "body": """\
### Summary

`envs/human_env.py` provides a human-playable interface via Pygame, which
requires a local display and cannot be used in headless server environments
or by remote stakeholders.  The existing `server/game_server.py` provides
a WebSocket game server but the frontend does not fully integrate with it.

### Gap

Playtesters and domain experts cannot evaluate agent behaviour without
installing Pygame and running code locally.  A browser-based interface
would dramatically lower the barrier to feedback collection.

### Proposed Work

- [ ] Implement a `BrowserHumanEnv` wrapper that streams observations to
  the frontend via the existing WebSocket game server
- [ ] Update `frontend/src/App.jsx` to handle human input (click-to-move,
  formation selector, fire order) and send actions back via WebSocket
- [ ] Implement a replay scrubber in `ReplayViewer` component (seek to any
  step, play/pause, speed control)
- [ ] Add a unit info panel: clicking a unit shows its strength, morale,
  supply level, and formation
- [ ] Add a win probability bar that updates each step

### Acceptance Criteria

- [ ] A human can play a full episode in the browser (no Pygame required)
- [ ] Replay viewer allows seeking to any step in a recorded episode
- [ ] Unit info panel displays all six key metrics
- [ ] WebSocket latency < 50 ms on localhost
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ── TESTING GAPS ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

TESTING_ISSUES: list[dict] = [
    {
        "title": "[CHORE] Add end-to-end integration tests for full training pipelines",
        "labels": [
            "type: chore", "priority: high",
            "domain: infra", "v1: training", "status: agent-created",
        ],
        "milestone": "M4: v1 Complete",
        "body": """\
### Summary

The test suite has excellent unit coverage (~75 test files) but no
integration tests that run a full training loop: environment creation →
training for N steps → evaluation → checkpoint save → policy reload.

### Gap

A unit test that mocks the environment cannot catch integration failures
such as: incompatible observation shapes after an env update, broken
checkpoint serialization, or reward normalisation drift.

### Proposed Work

- [ ] Add `tests/integration/test_v1_train_loop.py` that:
  - Creates `BattalionEnv` with `n_envs=2`, runs `train()` for 1 000 steps
  - Saves the checkpoint, reloads it, and runs 10 evaluation episodes
  - Asserts that episode reward is finite and actions are in-bounds
- [ ] Add `tests/integration/test_self_play_loop.py` covering the
  self-play iteration cycle (≥ 2 policy generations)
- [ ] Add `tests/integration/test_hrl_loop.py` covering one battalion →
  brigade HRL handoff
- [ ] Tag these tests with `pytest.mark.slow` so they are skipped by
  default and run only in the nightly CI workflow

### Acceptance Criteria

- [ ] All three integration tests pass in the nightly CI run
- [ ] Tests complete in < 5 minutes total on a 4-core CI runner
- [ ] No mocking of the core sim or env — real step execution required
""" + ATTRIBUTION,
    },
    {
        "title": "[CHORE] Add load tests for the game server and REST API",
        "labels": [
            "type: chore", "priority: low",
            "domain: infra", "v5: deployment", "status: agent-created",
        ],
        "milestone": "M12: v5 Complete",
        "body": """\
### Summary

`server/game_server.py` and `api/coa_endpoint.py` have no load tests.
There is no empirical evidence that they can sustain concurrent connections
from multiple workers or clients.

### Gap

League training with 64 Ray workers all connecting to the game server
simultaneously has never been tested.  A connection-pool exhaustion or
memory leak under load would only surface in a production run.

### Proposed Work

- [ ] Add `tests/load/test_game_server_load.py` using `locust` or
  `websockets` + `asyncio` to simulate 64 concurrent WebSocket clients
- [ ] Add `tests/load/test_api_load.py` using `httpx` async client to
  simulate 100 concurrent `/api/coa/analyze` requests
- [ ] Record peak throughput and p99 latency; store as CI artifacts
- [ ] Define SLA thresholds: game server ≥ 500 steps/s total,
  API p99 < 2 s

### Acceptance Criteria

- [ ] Load test reports committed as CI artifacts
- [ ] Game server meets 500 steps/s SLA under 64 concurrent clients
- [ ] API meets < 2 s p99 SLA under 100 concurrent requests
""" + ATTRIBUTION,
    },
]

# ---------------------------------------------------------------------------
# ALL ISSUES
# ---------------------------------------------------------------------------

ALL_ISSUES: list[dict] = (
    CICD_ISSUES
    + SECURITY_ISSUES
    + MODEL_ISSUES
    + SIM_INTEGRATION_ISSUES
    + CORRECTNESS_ISSUES
    + DOCS_ISSUES
    + ANALYSIS_ISSUES
    + FRONTEND_ISSUES
    + TESTING_ISSUES
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_color(label_name: str) -> str:
    """Return a reasonable hex color for an auto-created label."""
    _COLOR_MAP = {
        "type:": "7057ff",
        "priority:": "fbca04",
        "status:": "0075ca",
        "domain:": "bfd4f2",
        "v1:": "fef2c0",
        "v2:": "d4edda",
        "v3:": "cce5ff",
        "v4:": "f8d7da",
        "v5:": "e2d9f3",
        "v6:": "fff3cd",
        "v8:": "d6d8db",
    }
    for prefix, color in _COLOR_MAP.items():
        if label_name.startswith(prefix):
            return color
    return "ededed"


def get_milestone_by_title(repo, title: str) -> object | None:
    """Return the milestone whose title matches (open or closed), or None."""
    for ms in repo.get_milestones(state="all"):
        if ms.title == title:
            return ms
    return None


def existing_titles(repo) -> set[str]:
    """Return a normalised (lower-stripped) set of all non-PR issue titles."""
    titles: set[str] = set()
    for item in repo.get_issues(state="all"):
        if getattr(item, "pull_request", None) is None:
            titles.add(item.title.strip().lower())
    return titles


def create_issue(repo, issue_def: dict, known: set[str], *, dry_run: bool) -> str:
    """Create a single GitHub issue.  Returns 'created', 'exists', or 'skipped'."""
    title = issue_def["title"]

    if title.strip().lower() in known:
        print(f"  ↩ exists: {title[:80]}")
        return "exists"

    if dry_run:
        print(f"  [dry-run] Would create: {title[:80]}")
        return "skipped"

    # Resolve labels; auto-create any that are missing.
    label_objects = []
    for label_name in issue_def.get("labels", []):
        try:
            label_objects.append(repo.get_label(label_name))
        except Exception:
            try:
                label_objects.append(
                    repo.create_label(
                        name=label_name,
                        color=_label_color(label_name),
                    )
                )
                print(f"    [+] Auto-created label: {label_name}")
            except Exception:
                print(
                    f"    [warn] Label not found and could not be created: "
                    f"{label_name} — skipping"
                )

    # Resolve milestone; auto-create if missing.
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
                    f"    [warn] Milestone not found and could not be created: "
                    f"{milestone_title} — issue will have no milestone"
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
        print(
            "ERROR: PyGithub is not installed.  Run: pip install PyGithub",
            file=sys.stderr,
        )
        return 1

    auth = Auth.Token(token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    print(f"Connected to {repo.full_name}{'  [DRY RUN]' if dry_run else ''}")
    print(f"Total issues to seed: {len(ALL_ISSUES)}")

    print("\nFetching existing issue titles...")
    known = existing_titles(repo)
    print(f"  {len(known)} existing issues found.")

    sections = [
        ("── CI / CD Gaps", CICD_ISSUES),
        ("── Security Gaps", SECURITY_ISSUES),
        ("── Model / Policy Gaps", MODEL_ISSUES),
        ("── Simulation Integration Gaps", SIM_INTEGRATION_ISSUES),
        ("── Concurrency / Correctness Gaps", CORRECTNESS_ISSUES),
        ("── Documentation Gaps", DOCS_ISSUES),
        ("── Analysis & Evaluation Gaps", ANALYSIS_ISSUES),
        ("── Frontend / UX Gaps", FRONTEND_ISSUES),
        ("── Testing Gaps", TESTING_ISSUES),
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
