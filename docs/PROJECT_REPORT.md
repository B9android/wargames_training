# Wargames Training — Comprehensive Project Report

> **Date:** 2026-03-21  
> **Version at time of writing:** 5.0 (v5 code complete, pending roadmap update)  
> **Author:** Copilot project-report agent

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State — What Has Been Built](#2-current-state--what-has-been-built)
   - 2.1 [Simulation Engine](#21-simulation-engine)
   - 2.2 [Environments](#22-environments)
   - 2.3 [Training Pipeline](#23-training-pipeline)
   - 2.4 [Analysis & Explainability](#24-analysis--explainability)
   - 2.5 [Deployment Infrastructure](#25-deployment-infrastructure)
   - 2.6 [GitHub Automation & Operations](#26-github-automation--operations)
   - 2.7 [Test Coverage](#27-test-coverage)
3. [Version-by-Version Milestone Recap](#3-version-by-version-milestone-recap)
4. [What Is Missing — Known Gaps](#4-what-is-missing--known-gaps)
   - 4.1 [Simulation Realism](#41-simulation-realism)
   - 4.2 [Training & Learning Gaps](#42-training--learning-gaps)
   - 4.3 [Deployment & Interface Gaps](#43-deployment--interface-gaps)
   - 4.4 [Operational & Organisational Gaps](#44-operational--organisational-gaps)
5. [Most Logical Next Step](#5-most-logical-next-step)
6. [Long-Term Vision — v6 and Beyond](#6-long-term-vision--v6-and-beyond)
   - 6.1 [v6: Combined Arms & Expanded Realism](#61-v6-combined-arms--expanded-realism)
   - 6.2 [v7: Operational-Level Planning](#62-v7-operational-level-planning)
   - 6.3 [v8: Human-AI Collaborative Wargaming](#63-v8-human-ai-collaborative-wargaming)
7. [Recommended Priorities](#7-recommended-priorities)
8. [Appendix — Component Inventory](#8-appendix--component-inventory)

---

## 1. Executive Summary

**Wargames Training** has reached a significant milestone: all five planned development
versions (v1–v5) are code-complete. The project has grown from a single-battalion
1v1 Gymnasium environment into a full research platform that includes:

- A continuous 2D Napoleonic-era combat simulation engine
- Seven distinct Gymnasium/PettingZoo environments (battalion → brigade → division)
- Hierarchical RL (HRL) enabling multi-echelon command structures
- AlphaStar-style league training with Nash equilibrium sampling and diversity metrics
- Ray-based distributed rollout infrastructure
- Course-of-action (COA) generator and REST API
- SHAP-based strategy explainability
- Historical scenario validation (Waterloo, Austerlitz, Borodino)
- ONNX/TorchScript policy export with a Dockerised inference server
- Human-playable interface (keyboard-driven pygame)
- 50+ unit tests, 8 analysis notebooks, and 27 documentation files

Despite this breadth, the project remains a **research prototype**: no training run
has yet produced a deployable, validated policy artifact, and critical simulation
realism gaps limit its applicability beyond academic exploration.

The most logical immediate next step is **end-to-end validation** — running a
full training experiment through the complete pipeline (battalion → league →
export → COA generation → historical validation) and recording reproducible
results. The next major long-term goal is **v6: Combined Arms & Expanded Realism**,
which adds artillery, cavalry, and logistical constraints to push the agents toward
more doctrinally credible behaviour.

---

## 2. Current State — What Has Been Built

### 2.1 Simulation Engine

**Location:** `envs/sim/` (1,039 lines total)

The core physics simulation is complete and functional:

| Module | Lines | Responsibility |
|---|---|---|
| `battalion.py` | 104 | Battalion dataclass: position, heading, strength, morale |
| `combat.py` | 331 | Damage, morale drain, fire arcs, flanking/rear multipliers |
| `engine.py` | 241 | Simulation loop, state management, casualty application |
| `terrain.py` | 363 | Perlin-noise terrain, elevation speed factors, line-of-sight |

**Key constants:**

| Parameter | Value |
|---|---|
| Map size | 1 000 m × 1 000 m |
| Time step | 0.1 s |
| Base fire damage | 0.05 (fractional strength/step) |
| Morale routing threshold | 0.25 |
| Fire range | 200 m |
| Fire arc | ±45° frontal |

**What the simulation does well:** deterministic physics, morale mechanics,
terrain-induced speed penalties, line-of-sight occlusion, and cover bonuses are
all implemented.

**What the simulation lacks:** see §4.1.

---

### 2.2 Environments

Seven distinct environments exist across two levels of abstraction:

| Environment | API | Agents | Purpose |
|---|---|---|---|
| `BattalionEnv` | Gymnasium | 1 (Blue) | 1v1 vs. scripted Red |
| `MultiBattalionEnv` | PettingZoo Parallel | N (Blue) | NvN MARL |
| `RemoteMultiBattalionEnv` | PettingZoo + Ray | N (Blue) | Distributed MARL |
| `BrigadeEnv` | Gymnasium | 1 (Brigade commander) | HRL over frozen battalion sub-policies |
| `DivisionEnv` | Gymnasium | 1 (Division commander) | HRL over brigade commanders |
| `HumanEnv` | Gymnasium | 1 (Human) | Keyboard-driven human vs. AI |
| SMDP wrappers | Gymnasium | Any | Temporal abstraction / Options framework |

**Observation space (12-dim per battalion):** normalized position `(x, y)`,
heading `(cos θ, sin θ)`, own strength/morale, relative bearing/distance to
nearest enemy, enemy strength/morale, normalised step count.

**Action space (3-dim continuous per battalion):** forward/backward throttle,
rotation rate, fire intensity.

**Scripted opponents:** five curriculum levels (L1 random walk → L5 full
engagement), used as initial training ladders.

---

### 2.3 Training Pipeline

**Location:** `training/` (~9,200 lines)

The training pipeline covers the full RL research lifecycle:

#### Single-Agent (v1)
- PPO with Stable-Baselines3 (`train.py`, 930 lines)
- `BattalionMlpPolicy` — 2-layer MLP (obs → 128 → Tanh → 128 → Tanh)
- Hydra config, W&B logging, checkpoint/resume, Elo tracking
- Evaluation against L1–L5 opponents and self-play pool

#### Multi-Agent / MAPPO (v2)
- Centralized-training decentralized-execution (CTDE) via `MAPPOPolicy`
- `MAPPORolloutBuffer` with per-agent GAE
- 3-stage curriculum: 1v1 → 2v1 → 2v2 with rolling win-rate promotion
- Team Elo registry, coordination metrics (flanking ratio, fire concentration,
  mutual support score)
- NvN scaling tested up to 6v6

#### Hierarchical RL (v3)
- `HRLCurriculumScheduler`: bottom-up training across three phases
  (PHASE\_1\_BATTALION → PHASE\_2\_BRIGADE → PHASE\_3\_DIVISION)
- Options/SMDP framework with five tactical primitives
  (advance, hold, retreat, flank-left, flank-right)
- `PolicyRegistry` (JSON manifest, echelon-aware versioning)
- Freeze utilities (`freeze_mappo_policy`, `freeze_sb3_policy`)
- Adaptive temporal abstraction scheduler
- HRL vs. flat MARL tournament with bootstrapped 95% CIs

#### League Training (v4)
- AlphaStar-style three-tier league:
  - `MainAgentTrainer` (PFSP vs. all, MAPPO + Elo + snapshots)
  - `MainExploiterTrainer` (targets latest main agent, orthogonal reinit on
    high win-rate)
  - `LeagueExploiterTrainer` (PFSP vs. full pool, Nash exploitability)
- `AgentPool` (JSON manifest, atomic writes), `MatchDatabase` (JSONL log)
- `LeagueMatchmaker` (PFSP hard-first, swappable weight function)
- Nash distribution sampling via LP + regret matching (`nash.py`)
- Strategy diversity metrics: action histogram + position heatmap + movement
  stats, L2-normalised embeddings, cosine distance, mean/min/median diversity
- Distributed rollout via Ray (`DistributedRolloutRunner`, `@ray.remote` actors)
- W&B keys: `league/nash_entropy`, `league/diversity_score`

#### Analysis Tools (v5 / completed)
- `SaliencyAnalyser` (SHAP, permutation, gradient-based) — `analysis/saliency.py`
- `COAGenerator` (Monte-Carlo rollouts over 7 tactical archetypes) —
  `analysis/coa_generator.py`
- Historical scenario validation against Waterloo, Austerlitz, Borodino —
  `envs/scenarios/historical.py`

---

### 2.4 Analysis & Explainability

| Component | File | Capability |
|---|---|---|
| Saliency maps | `analysis/saliency.py` (739 lines) | SHAP, permutation, gradient; feature importance |
| COA generator | `analysis/coa_generator.py` (576 lines) | 7 archetypes, Monte-Carlo scoring, ranked COA list |
| COA REST API | `api/coa_endpoint.py` (221 lines) | Flask `/health`, POST `/coas` |
| Historical validation | `envs/scenarios/historical.py` (541 lines) | YAML scenario loader, `OutcomeComparator`, fidelity score |
| Explainability notebook | `notebooks/explainability_demo.ipynb` | SHAP waterfall plots |
| Historical notebook | `notebooks/historical_validation.ipynb` | Fidelity metrics across Napoleonic battles |

**The 7 COA archetypes:** frontal assault, flanking manoeuvre, double envelopment,
defence in depth, feint-and-strike, fire suppression + advance, hold-and-attrit.

---

### 2.5 Deployment Infrastructure

| Component | File | Capability |
|---|---|---|
| Policy export | `scripts/export_policy.py` (674 lines) | ONNX, TorchScript, inference benchmark, CLI |
| Policy server | `docker/policy_server/server.py` (220 lines) | Flask `/predict`, `/health`, `/info` |
| Dockerfile | `docker/policy_server/Dockerfile` | Containerised ONNX + TorchScript serving |
| Deployment guide | `docs/deployment_guide.md` | Container ops, scaling, monitoring |

---

### 2.6 GitHub Automation & Operations

A comprehensive GitHub automation layer lives in `scripts/project_agent/` (~3,000 lines):

| Agent | Responsibility |
|---|---|
| `triage_agent.py` | Auto-label and milestone issues on creation |
| `experiment_lifecycle.py` | Track experiment open → running → complete |
| `progress_reporter.py` | Weekly W&B-based progress reports |
| `training_monitor.py` | CI/CD training run monitoring |
| `sprint_manager.py` / `sprint_assigner.py` | Sprint automation |
| `release_coordinator.py` | Release orchestration |
| `epic_decomposer.py` | Decompose epics into sub-issues |
| `milestone_checker.py` | Validate milestone closure criteria |
| `static_analysis_agent.py` | Code quality checks |

Six GitHub Actions workflows automate triage, governance, label setup, issue
seeding, and Ray smoke tests.

---

### 2.7 Test Coverage

54 test files covering all major components:

| Domain | Key test files |
|---|---|
| Simulation | `test_battalion.py`, `test_combat.py`, `test_terrain.py`, `test_sim.py` |
| Environments | `test_battalion_env.py`, `test_brigade_env.py`, `test_division_env.py`, `test_multi_battalion_env.py` |
| Training | `test_train.py`, `test_train_main_agent.py`, `test_self_play.py`, `test_evaluate.py` |
| League | `test_agent_pool.py`, `test_matchmaker.py`, `test_nash.py`, `test_diversity.py` |
| HRL | `test_hrl_curriculum.py`, `test_evaluate_hrl.py`, `test_smdp_wrapper.py` |
| Analysis | `test_coa_generator.py`, `test_saliency.py`, `test_historical_scenarios.py` |
| Export | `test_policy_export.py` |

---

## 3. Version-by-Version Milestone Recap

| Version | Theme | Status | Key Deliverables |
|---|---|---|---|
| **v1** | Foundation — 1v1 battalion | ✅ Complete | PPO + curriculum, Elo, self-play, visualization |
| **v2** | Multi-Agent — MARL 2v2+ | ✅ Complete | MAPPO, NvN scaling, team Elo, coordination metrics |
| **v3** | Hierarchy — Brigade/Division HRL | ✅ Complete | SMDP/Options, HRL curriculum, policy registry, adaptive temporal abstraction |
| **v4** | League — AlphaStar-style training | ✅ Complete | League infrastructure, Nash sampling, diversity metrics, Ray distributed training |
| **v5** | Real-World Interface & Analysis | ✅ Code complete | Human interface, COA generator, SHAP explainability, historical validation, ONNX export |

> **Note:** The ROADMAP.md currently marks v5 as 🔮 Future. All v5 epics
> (E5.1–E5.5) are code-complete and tested. The roadmap should be updated to
> reflect v5 as complete. See §5 for the immediate action item.

---

## 4. What Is Missing — Known Gaps

### 4.1 Simulation Realism

The current simulation is intentionally simplified for fast RL training. Moving
toward operational credibility requires addressing several realism gaps:

| Gap | Impact | Difficulty |
|---|---|---|
| **Single unit type only** | Agents cannot learn inter-arm cooperation (infantry/cavalry/artillery) | High — requires new action/obs spaces |
| **No artillery or indirect fire** | Eliminates one of the dominant Napoleonic tactical dimensions | High |
| **No cavalry** | Pursuit, screening, and shock action missing | Medium |
| **No supply/logistics** | Agents cannot be penalized for overextension | High |
| **No fog of war beyond visibility radius** | Agents cannot learn intelligence-driven deception | Medium |
| **No unit formation types** | Line, column, square formation switching with stat modifiers is absent | Medium |
| **Fixed map size (1 km²)** | Scales poorly to operational scenarios spanning tens of kilometres | Medium |
| **No weather or time-of-day effects** | Condition-dependent doctrine impossible to learn | Low |
| **Deterministic physics** | No noise → agents may overfit to specific engagement geometries | Low |

### 4.2 Training & Learning Gaps

| Gap | Impact | Difficulty |
|---|---|---|
| **No trained policy artifacts** | Every claim about agent capability is hypothetical — no benchmark results exist | Critical — must run actual experiments |
| **No validation against historical outcomes** | The `OutcomeComparator` exists but has never been used against a policy trained to convergence | High |
| **League training never run end-to-end** | Nash equilibrium and diversity metrics have been unit-tested but never observed in a live league | High |
| **No curriculum continuity across versions** | v1 PPO policies cannot be smoothly continued into v2/v3/v4 training | Medium |
| **No adversarial robustness evaluation** | Policies may be brittle to slight perturbations in initial conditions | Medium |
| **`TransformerPolicy` is a placeholder** | `models/transformer_policy.py` has the scaffold but no implementation | Low–Medium |
| **No hyperparameter sweep results** | W&B sweeps have never been run; no reported optimal hyperparameters | Medium |
| **Concurrent writer risk in `AgentPool`** | JSON manifest uses a single `.tmp` file — not safe for concurrent multi-process writers without external locking | Medium |

### 4.3 Deployment & Interface Gaps

| Gap | Impact | Difficulty |
|---|---|---|
| **No web-based human interface** | `HumanEnv` requires pygame on the same machine; no browser-accessible UI | Medium |
| **No interactive COA dashboard** | COA results are returned via API only; no visual front-end to display and compare courses of action | Medium |
| **No model registry / versioned artifacts** | `PolicyRegistry` exists but there is no integration with MLflow, W&B Artifacts, or a model registry | Medium |
| **ONNX export untested at scale** | Exported models have been tested in unit tests but not benchmarked against a target latency SLA | Low |
| **No authentication on REST API** | Flask policy server and COA endpoint have no auth layer | Low–Medium |
| **No batch inference endpoint** | Policy server supports single predictions; no batched or streaming endpoint | Low |

### 4.4 Operational & Organisational Gaps

| Gap | Impact | Difficulty |
|---|---|---|
| **README still says "Current Version: v1"** | Misleads new contributors about project maturity | Trivial |
| **ROADMAP.md marks v5 as 🔮 Future** | Documentation is out of sync with actual implementation | Trivial |
| **No published benchmarks / paper** | The research value cannot be assessed or cited without results | High |
| **No license file** | `README.md` says "Add license here" — unclear ownership and reuse rights | Low |
| **W&B entity / project not configured** | `configs/default.yaml` has `entity: null` — first-time contributors cannot share results | Low |
| **No CI training smoke test** | CI only tests environment creation and single-step rollouts; no end-to-end training verification | Medium |

---

## 5. Most Logical Next Step

**Run the first end-to-end validated training experiment.**

All the machinery is in place. The critical missing piece is proof that the
pipeline works from start to finish, producing a policy that demonstrably defeats
scripted opponents and passes historical validation. The recommended sequence:

### Step 1 — Documentation Housekeeping (1–2 days)
- Mark v5 as complete in `docs/ROADMAP.md`
- Update `README.md` to reflect "Current Version: v5"
- Add an open-source license (MIT or Apache 2.0)
- Set the W&B `entity` / `project` in `configs/default.yaml`

### Step 2 — Baseline Training Run (1 week)
1. Train a v1 PPO policy to convergence against scripted L5
2. Log full W&B run with all reward breakdown metrics
3. Run `training/evaluate.py` — record win rate vs. L1–L5 and Elo
4. Export to ONNX via `scripts/export_policy.py`
5. Serve with `docker/policy_server/` and verify prediction latency

### Step 3 — Historical Validation (2–3 days)
1. Load the trained policy into `envs/scenarios/historical.py`
2. Run `OutcomeComparator` on Waterloo, Austerlitz, Borodino scenarios
3. Record `fidelity_score` per scenario
4. Publish results in `notebooks/historical_validation.ipynb`

### Step 4 — Self-Play Baseline (3–5 days)
1. Run self-play loop to ≥ 200 k timesteps
2. Record Elo trajectory and pool diversity
3. Compare self-play policy to scripted-L5-trained policy
4. Document in a new GitHub issue `[EXP] v1 Self-Play Baseline`

### Step 5 — League Smoke Run (1 week)
1. Initialise `AgentPool` with the v1 PPO policy as seed main agent
2. Run 1 × `MainAgentTrainer`, 1 × `MainExploiterTrainer`,
   1 × `LeagueExploiterTrainer` for 50 k timesteps each
3. Record Nash entropy and diversity score trajectories in W&B
4. Verify no `AgentPool` concurrency issues

### Step 6 — Fix AgentPool Concurrency (3–5 days)
Replace the `.tmp` single-file atomic write in `training/league/agent_pool.py`
with a proper file-lock (e.g., `filelock`) so concurrent writers cannot corrupt
the manifest.

Completing Steps 1–6 would give the project its first validated, reproducible
baseline — a foundation for all further research claims.

---

## 6. Long-Term Vision — v6 and Beyond

The current foundation (multi-echelon simulation, CTDE MARL, HRL, league training,
COA generation, historical validation, policy export) supports an ambitious
long-term trajectory.

### 6.1 v6: Combined Arms & Expanded Realism

**Theme:** Introduce combined-arms interactions and push simulation fidelity to
the level required for serious operational research.

**Core additions:**

| Epic | Description |
|---|---|
| E6.1 | **Multi-arm unit types** — Infantry, Cavalry, Artillery as distinct unit classes with type-specific action spaces and stat tables |
| E6.2 | **Artillery layer** — indirect fire, suppression, counter-battery mechanics; brigade and division commanders can issue fire missions |
| E6.3 | **Formation system** — line (maximum firepower), column (maximum movement), square (anti-cavalry) with transitions and associated stat modifiers |
| E6.4 | **Cavalry shock and pursuit** — melee combat, rout exploitation, screening missions |
| E6.5 | **Supply and logistics** — ammunition depletion, resupply routes, overextension penalties |
| E6.6 | **Larger maps** — 10 km × 10 km operational theatre, procedural map generation with rivers, roads, and villages |
| E6.7 | **Weather and visibility** — fog of war, rain (fire penalty), night operations |
| E6.8 | **v6 Documentation & Release** |

**Target milestones:** M11 (Combined Arms), M12 (v6 Complete)

**Key research questions:**
- Does MAPPO learn effective combined-arms tactics without explicit reward for
  inter-arm coordination?
- Do formation transitions emerge from unit survival incentives alone?
- Can the HRL commander discover historically documented battle plans for
  multi-arm engagements?

**Estimated effort:** 12–16 weeks

---

### 6.2 v7: Operational-Level Planning

**Theme:** Scale to corps / army-level simulation; introduce strategic and
operational planning layers above the brigade/division tactical layer.

**Core additions:**

| Epic | Description |
|---|---|
| E7.1 | **Corps-level environment** — 3+ division hierarchy; map spanning 50+ km |
| E7.2 | **Planning horizon abstraction** — temporal ratio of 1 000:1 (one corps commander step = 100 battalion steps) |
| E7.3 | **Lines of communication** — model road networks, supply depots, retreat routes |
| E7.4 | **Intelligence layer** — probabilistic enemy position estimates; agents must manage information uncertainty |
| E7.5 | **Coalition forces** — allied AI or human co-commanders with doctrine differences |
| E7.6 | **Asymmetric scenarios** — attacker/defender imbalance; siege mechanics |
| E7.7 | **Historical campaign validation** — Austerlitz campaign (not just battle), Leipzig, Waterloo campaign |
| E7.8 | **v7 Documentation & Release** |

**Key research questions:**
- Does a learned corps commander discover operational manoeuvres (e.g.,
  the strategy of the central position) without being shown historical examples?
- How does uncertainty in enemy intelligence affect Nash equilibrium properties
  of the league?
- Can the system reproduce campaign-level outcomes that are beyond the scope
  of single-battle simulation?

**Estimated effort:** 20–28 weeks

---

### 6.3 v8: Human-AI Collaborative Wargaming

**Theme:** Turn the trained system into a tool for human military planners.
Replace the research prototype with a production-quality collaborative
decision-support system.

**Core additions:**

| Epic | Description |
|---|---|
| E8.1 | **Web-based interactive interface** — Browser-accessible command console; human issues orders to brigade/division AI subordinates |
| E8.2 | **COA comparison dashboard** — Visual side-by-side display of AI-generated courses of action with confidence intervals, risk estimates, and historical analogues |
| E8.3 | **Red team AI** — Adversarial AI that challenges human plans; surfaces exploitable weaknesses automatically |
| E8.4 | **Doctrine injection** — Allow users to encode doctrine constraints (e.g., "hold the ridge until 09:00") as reward shaping or SMDP option constraints |
| E8.5 | **Natural language interface** — LLM front-end that translates plain-English orders into SMDP options and displays AI reasoning in plain language |
| E8.6 | **Audit trail & replay** — Full decision replay, counterfactual ("what if?") scenario branching |
| E8.7 | **Export to standard formats** — OPORD generation, MSEL export, interop with JCATS / OneSAF |
| E8.8 | **v8 Documentation & Release** |

**Key research questions:**
- Does access to AI-generated COAs measurably improve human planner decision
  quality in controlled experiments?
- Can the NLP interface lower the expertise threshold sufficiently for non-RL
  practitioners to use the system?
- Does operator trust in AI-generated plans increase when SHAP explainability is
  surfaced alongside recommendations?

**Estimated effort:** 24–36 weeks (includes significant UX/frontend work)

---

## 7. Recommended Priorities

The table below maps gaps to priority and suggests which version roadmap they
belong to.

| Priority | Item | Version | Effort |
|---|---|---|---|
| 🔴 Critical | Run first end-to-end training experiment | Immediate | 1–2 weeks |
| 🔴 Critical | Fix `AgentPool` concurrent writer vulnerability | v5 patch | 1–2 days |
| 🟠 High | Update ROADMAP.md + README to reflect v5 complete | v5 patch | 1 hour |
| 🟠 High | Add open-source license | v5 patch | 30 minutes |
| 🟠 High | Implement `TransformerPolicy` | v5.1 or v6 | 1 week |
| 🟠 High | Web-based COA dashboard | v5.1 | 2–3 weeks |
| 🟡 Medium | Multi-arm unit types (infantry / cavalry / artillery) | v6 | 4–6 weeks |
| 🟡 Medium | Formation system | v6 | 2–3 weeks |
| 🟡 Medium | Larger maps + procedural generation | v6 | 2–3 weeks |
| 🟡 Medium | CI training smoke test | v5 patch | 1 week |
| 🟢 Low | REST API authentication | v5.1 | 2 days |
| 🟢 Low | W&B entity / project default config | v5 patch | 1 hour |
| 🟢 Low | Batch inference endpoint | v5.1 | 1 day |

---

## 8. Appendix — Component Inventory

### Source files by domain

| Domain | Files | Approx. lines |
|---|---|---|
| Simulation engine | `envs/sim/` (4 files) | ~1,039 |
| Environments | `envs/` (7 env files + helpers) | ~3,500 |
| Models | `models/` (3 files) | ~800 |
| Training (core) | `training/` (11 files) | ~7,000 |
| League training | `training/league/` (8 files) | ~4,500 |
| Analysis | `analysis/` (2 files) | ~1,315 |
| API | `api/` (1 file) | ~221 |
| Scripts / export | `scripts/` (4 files) | ~1,400 |
| GitHub automation | `scripts/project_agent/` (20+ files) | ~6,000 |
| Tests | `tests/` (54 files) | ~12,000 |
| Documentation | `docs/` (27 files) | ~15,000 |
| Configs | `configs/` (20+ YAML files) | ~1,200 |
| **Total** | **~160 files** | **~54,000 lines** |

### Dependency summary

| Category | Key packages |
|---|---|
| RL | `torch`, `stable-baselines3`, `gymnasium`, `pettingzoo` |
| Distributed | `ray[rllib]` |
| Tracking | `wandb` |
| Numerical | `numpy`, `scipy` |
| Explainability | `shap` |
| Visualization | `matplotlib`, `pygame` |
| Config | `hydra-core`, `pyyaml` |
| API | `flask` |
| Export | `onnx`, `onnxruntime` |
| Automation | `PyGithub`, `openai` |
| Dev | `pytest`, `black`, `ruff` |

---

*This report was generated by the Copilot project-report agent on 2026-03-21.
It is intended as a living document — update it as new milestones are reached
and new research directions are confirmed.*
