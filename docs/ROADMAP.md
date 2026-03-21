# Wargames Training — Project Roadmap

> **Living document.** Updated by agents and maintainers.
> Last updated: auto-updated by weekly report agent.

---

## Version Overview

| Version | Theme | Status | Target |
|---|---|---|---|
| **v1** | Foundation — 1v1 battalion | ✅ Complete | M4 |
| **v2** | Multi-Agent — MARL 2v2+ | ✅ Complete | M6 |
| **v3** | Hierarchy — Brigade/Division HRL | ✅ Complete | M8 |
| **v4** | League — AlphaStar-style training | ✅ Complete | M10 |
| **v5** | Real-World Interface & Analysis | ✅ Complete | M12 |

---

## v1: Foundation (Complete ✅)

**Goal:** A single battalion agent that reliably defeats scripted opponents
in 1v1 continuous 2D battles, generalizes across randomized parameters,
and has a working self-play loop.

### Epics
- [x] **E1.1** — Project Bootstrap & Tooling
- [x] **E1.2** — Core Simulation Engine
- [x] **E1.3** — Gymnasium Environment (1v1)
- [x] **E1.4** — Baseline Training Loop (PPO + scripted opponent)
- [x] **E1.5** — Terrain & Environmental Randomization
- [x] **E1.6** — Reward Shaping & Curriculum Design
- [x] **E1.7** — Self-Play Implementation
- [x] **E1.8** — Evaluation Framework & Elo Tracking
- [x] **E1.9** — Visualization & Replay System
- [x] **E1.10** — v1 Documentation & Release

### Milestones
- M0: Project Bootstrap ✅
- M1: 1v1 Competence ✅
- M2: Terrain & Generalization ✅
- M3: Self-Play Baseline ✅
- M4: v1 Complete ✅

---

## v2: Multi-Agent (Complete ✅)

**Goal:** Multiple battalions per side coordinate using MAPPO.
Emergent flanking, fire concentration, and mutual support.

### Epics
- [x] **E2.1** — PettingZoo Multi-Agent Environment
- [x] **E2.2** — MAPPO Implementation (Centralized Critic)
- [x] **E2.3** — 2v2 Curriculum
- [x] **E2.4** — Coordination Metrics & Analysis
- [x] **E2.5** — Scale to NvN (up to 6v6)
- [x] **E2.6** — Multi-Agent Self-Play
- [x] **E2.7** — v2 Documentation & Release

### Milestones
- M5: 2v2 MARL ✅
- M6: v2 Complete ✅

### Key Research Questions
- Does parameter sharing (shared policy across all battalions) outperform
  independent policies?
- What observation radius produces optimal coordination without
  information overload?
- Does emergent flanking behavior appear without explicit reward for it?

### Sprint Schedule (v2)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S10 | 19–20 | PettingZoo environment (2v2) | E2.1 |
| S11 | 21–22 | MAPPO implementation | E2.2 |
| S12 | 23–24 | 2v2 curriculum + coordination metrics | E2.3, E2.4 |
| S13 | 25–26 | NvN scaling + multi-agent self-play | E2.5, E2.6 |
| S14 | 27–28 | v2 polish + release | E2.7 |

---

## v3: Hierarchical RL (Complete ✅)

**Goal:** Brigade and division commanders issue macro-commands to frozen
battalion policies. HRL architecture matching Black (NPS 2024).

### Epics
- [x] **E3.1** — SMDP / Options Framework
- [x] **E3.2** — Brigade Commander Layer
- [x] **E3.3** — Division Commander Layer
- [x] **E3.4** — Hierarchical Curriculum (bottom-up training)
- [x] **E3.5** — Temporal Abstraction Tuning
- [x] **E3.6** — Multi-Model Policy Library (per echelon)
- [x] **E3.7** — HRL Evaluation vs. Flat MARL
- [x] **E3.8** — v3 Documentation & Release

### Milestones
- M7: HRL Battalion→Brigade ✅
- M8: v3 Complete ✅

### Key Research Questions
- Does hierarchical decomposition outperform flat MARL at scale?
- What is the optimal temporal abstraction ratio between echelons?
- Can brigade commanders discover novel operational maneuvers?

### Sprint Schedule (v3)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S15 | 29–30 | SMDP / Options framework | E3.1 |
| S16 | 31–32 | Brigade commander layer + HRL curriculum | E3.2, E3.4 |
| S17 | 33–34 | Division commander + HRL evaluation | E3.3, E3.7 |
| S18 | 35–36 | Temporal abstraction + policy library + v3 release | E3.5, E3.6, E3.8 |

---

## v4: League Training (Complete ✅)

**Goal:** AlphaStar-style league with main agents, exploiters, and
league exploiters. Nash equilibrium sampling. Strategy diversity metrics.

### Epics
- [x] **E4.1** — League Infrastructure (agent pool, matchmaking)
- [x] **E4.2** — Main Agent Training Loop
- [x] **E4.3** — Main Exploiter Agents
- [x] **E4.4** — League Exploiter Agents
- [x] **E4.5** — Nash Distribution Sampling
- [x] **E4.6** — Strategy Diversity Metrics
- [x] **E4.7** — Distributed Training (Ray)
- [x] **E4.8** — v4 Documentation & Release

### Milestones
- M9: League Training ✅
- M10: v4 Complete ✅

### Key Research Questions
- Can we achieve Nash equilibrium sampling in a multi-agent wargame setting?
- Does a diverse league produce demonstrably more robust main agents?
- What level of distribution is required for a commercially viable league?

### Sprint Schedule (v4)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S19 | 37–38 | League infrastructure + matchmaking | E4.1 |
| S20 | 39–40 | Main agent + exploiter training | E4.2, E4.3 |
| S21 | 41–42 | League exploiters + Nash sampling + diversity | E4.4, E4.5, E4.6 |
| S22 | 43–44 | Distributed training (Ray/RLlib) + v4 release | E4.7, E4.8 |

---

## v5: Analysis & Interface (Complete ✅)

**Goal:** Turn the trained system into a useful wargaming tool.
Human-vs-AI play, COA analysis, strategy visualization.

### Epics
- [x] **E5.1** — Human-playable interface (`envs/human_env.py`, `scripts/play.py`)
- [x] **E5.2** — Course of Action (COA) generator (`analysis/coa_generator.py`, `api/coa_endpoint.py`)
- [x] **E5.3** — Strategy explainability (`analysis/saliency.py`, `notebooks/explainability_demo.ipynb`)
- [x] **E5.4** — Historical scenario validation (`envs/scenarios/historical.py`, `configs/scenarios/historical/`)
- [x] **E5.5** — Export trained policies for deployment (`scripts/export_policy.py`, `docker/policy_server/`)

### Milestones
- M11: Interface & Analysis ✅
- M12: v5 Complete ✅

### Key Research Questions
- Do trained policies reproduce historically documented tactics?
- Can the COA generator surface novel strategies missed by human planners?
- How much does explainability improve operator trust in AI-generated COAs?

---

## Sprint Schedule

Sprints are 2 weeks. Sprint planning happens every other Monday.

### v1 Sprints (Active)
| Sprint | Weeks | Focus |
|---|---|---|
| S01 | 1–2 | Bootstrap, env setup, first training run |
| S02 | 3–4 | Sim engine, combat resolution |
| S03 | 5–6 | Reward shaping, curriculum L1–L3 |
| S04 | 7–8 | Terrain system, curriculum L4 |
| S05 | 9–10 | Generalization, randomized params |
| S06 | 11–12 | Self-play infrastructure |
| S07 | 13–14 | Self-play training, Elo tracking |
| S08 | 15–16 | Evaluation, visualization, v1 polish |
| S09 | 17–18 | v1 release, v2 planning |

### v2 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S10 | 19–20 | PettingZoo multi-agent environment (2v2) |
| S11 | 21–22 | MAPPO implementation |
| S12 | 23–24 | 2v2 curriculum + coordination metrics |
| S13 | 25–26 | NvN scaling + multi-agent self-play |
| S14 | 27–28 | v2 polish + release |

### v3 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S15 | 29–30 | SMDP / Options framework |
| S16 | 31–32 | Brigade commander layer + HRL curriculum |
| S17 | 33–34 | Division commander + HRL evaluation |
| S18 | 35–36 | Temporal abstraction + policy library + v3 release |

### v4 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S19 | 37–38 | League infrastructure + matchmaking |
| S20 | 39–40 | Main agent + exploiter training |
| S21 | 41–42 | League exploiters + Nash sampling + diversity |
| S22 | 43–44 | Distributed training (Ray/RLlib) + v4 release |
