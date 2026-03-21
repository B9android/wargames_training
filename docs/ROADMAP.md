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
| **v6** | Physics-Accurate Simulation | 🔲 Planned | M14 |
| **v7** | Operational Scale (Corps / Army) | 🔲 Planned | M16 |
| **v8** | Transformer Policy & Architecture | 🔲 Planned | M18 |
| **v9** | Human-in-the-Loop & Decision Support | 🔲 Planned | M20 |
| **v10** | Multi-Domain & Joint Operations | 🔲 Planned | M22 |
| **v11** | Real-World Data & Transfer | 🔲 Planned | M24 |
| **v12** | Foundation Model & Open Platform | 🔲 Planned | M26 |

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

## v6: Physics-Accurate Simulation (Planned 🔲)

**Goal:** Replace the abstract 2D simulation with a historically-grounded
physics model: terrain elevation, line-of-sight, realistic weapon ranges and
reload cycles, formation mechanics, morale cascades, supply consumption, and
weather effects.  This is the foundation that makes all subsequent versions
tactically meaningful.

### Epics
- [ ] **E6.1** — Terrain Elevation & Line-of-Sight Engine
- [ ] **E6.2** — Realistic Weapon Ranges, Accuracy & Reload Cycles
- [ ] **E6.3** — Morale, Cohesion & Rout Mechanics
- [ ] **E6.4** — Formation System (Line, Column, Square, Skirmish)
- [ ] **E6.5** — Supply, Ammunition & Fatigue Model
- [ ] **E6.6** — Weather & Time-of-Day Effects
- [ ] **E6.7** — v6 Documentation & Release

### Milestones
- M13: Physics Simulation
- M14: v6 Complete

### Key Research Questions
- Does terrain-aware training produce qualitatively different emergent tactics?
- What is the minimum physics fidelity required for historically-plausible agent behaviour?
- Can agents discover fire-and-movement doctrine without explicit reward?

### Sprint Schedule (v6)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S23 | 45–46 | Terrain engine + weapon system | E6.1, E6.2 |
| S24 | 47–48 | Formations + morale + rout | E6.3, E6.4 |
| S25 | 49–50 | Logistics + weather + v6 release | E6.5, E6.6, E6.7 |

---

## v7: Operational Scale — Corps Command (Planned 🔲)

**Goal:** Extend the HRL stack to corps level (3–5 divisions per side on a
20–50 km² map).  Introduce road networks, strategic supply chains, and
operational objectives (capture, interdict, fix-and-flank).

### Epics
- [ ] **E7.1** — Corps-Level Operational Environment
- [ ] **E7.2** — Strategic Supply & Logistics Network
- [ ] **E7.3** — Multi-Corps Self-Play & League Extension
- [ ] **E7.4** — v7 Documentation & Release

### Milestones
- M15: Corps Command
- M16: v7 Complete

### Key Research Questions
- Does corps-level HRL discover Napoleon's corps maneuver system independently?
- What map scale is needed to make supply interdiction a decisive operational factor?
- Does Nash equilibrium sampling still prevent strategy collapse at corps scale?

### Sprint Schedule (v7)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S26 | 51–52 | Corps env + road network + objectives | E7.1 |
| S27 | 53–54 | Strategic supply + corps league | E7.2, E7.3 |
| S28 | 55–56 | v7 release | E7.4 |

---

## v8: Transformer Policy & Attention Architecture (Planned 🔲)

**Goal:** Replace fixed-size concatenation observations with variable-length
entity-token sequences processed by a multi-head self-attention transformer.
Add recurrent memory for fog-of-war scenarios.  Systematic scaling study.

### Epics
- [ ] **E8.1** — Entity-Based Observation & Transformer Policy
- [ ] **E8.2** — Memory Module (LSTM / Temporal Context)
- [ ] **E8.3** — Model Scaling & Hyperparameter Study
- [ ] **E8.4** — v8 Documentation & Release

### Milestones
- M17: Transformer Policy
- M18: v8 Complete

### Key Research Questions
- Does entity-based transformer encoding outperform flat MLP at 8v8+?
- Does recurrent memory provide meaningful advantage under fog-of-war?
- What is the optimal model size for the performance–latency Pareto frontier?

### Sprint Schedule (v8)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S29 | 57–58 | Entity encoder + transformer policy | E8.1 |
| S30 | 59–60 | Memory module + scaling study + v8 release | E8.2, E8.3, E8.4 |

---

## v9: Human-in-the-Loop & Decision Support (Planned 🔲)

**Goal:** Build a web-based wargaming interface, an AI-assisted COA planning
tool at corps scale, and an after-action review system with DAgger-style
human feedback integration.

### Epics
- [ ] **E9.1** — Interactive Web-Based Wargame Interface
- [ ] **E9.2** — AI-Assisted Course of Action (COA) Planning Tool
- [ ] **E9.3** — After-Action Review & Training Feedback Loop
- [ ] **E9.4** — v9 Documentation & Release

### Milestones
- M19: Decision Support
- M20: v9 Complete

### Key Research Questions
- Do AI-generated COAs outperform expert human planners on novel scenarios?
- Does DAgger-style human feedback improve performance on human-designed scenarios?
- What level of explainability is required for operator trust in AI COAs?

### Sprint Schedule (v9)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S31 | 61–62 | Web interface + AI policy server | E9.1 |
| S32 | 63–64 | COA tool + AAR + v9 release | E9.2, E9.3, E9.4 |

---

## v10: Multi-Domain & Joint Operations (Planned 🔲)

**Goal:** Add naval units, elevate cavalry and artillery to independent
operational arms, and enable joint combined-arms operations (land + sea).

### Epics
- [ ] **E10.1** — Naval Unit Type & Coastal Operations
- [ ] **E10.2** — Cavalry Arm as Independent Maneuver Force
- [ ] **E10.3** — Artillery Arm: Grand Battery & Counter-Battery
- [ ] **E10.4** — v10 Documentation & Release

### Milestones
- M21: Joint Operations
- M22: v10 Complete

### Key Research Questions
- Does emergent joint combined-arms doctrine appear without explicit reward?
- Does naval fire support change amphibious assault tactics discovered by agents?
- Does cavalry reconnaissance measurably improve corps-level decision quality?

### Sprint Schedule (v10)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S33 | 65–66 | Naval units + coastal operations | E10.1 |
| S34 | 67–68 | Cavalry corps + grand battery + v10 release | E10.2, E10.3, E10.4 |

---

## v11: Real-World Data & Transfer (Planned 🔲)

**Goal:** Import 50+ historical Napoleonic battle OOBs and GIS terrain for
real battle sites.  Collect expert demonstrations via the v9 interface.
Validate and fine-tune agents on real-world data.

### Epics
- [ ] **E11.1** — Historical Battle Database & Scenario Importer
- [ ] **E11.2** — GIS Terrain Import (real-world maps)
- [ ] **E11.3** — Expert Demonstration Collection & Imitation Learning
- [ ] **E11.4** — v11 Documentation & Release

### Milestones
- M23: Real-World Transfer
- M24: v11 Complete

### Key Research Questions
- Does zero-shot transfer to real terrain lose less than 20 % win rate vs. procedural?
- Does behaviour cloning from expert demonstrations accelerate RL convergence?
- Do agents reproduce historical Napoleonic maneuvers on real battlefield terrain?

### Sprint Schedule (v11)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S35 | 69–70 | Historical database + GIS import | E11.1, E11.2 |
| S36 | 71–72 | Expert demonstrations + BC pre-training + v11 release | E11.3, E11.4 |

---

## v12: Foundation Model & Open Research Platform (Planned 🔲)

**Goal:** Train WFM-1 — a single large transformer policy generalising across
all scenarios, scales, and unit types.  Open-source the full stack as a
reproducible research benchmark (WargamesBench).  Submit the system paper.

### Epics
- [ ] **E12.1** — Wargames Foundation Model (WFM-1)
- [ ] **E12.2** — Open Research Platform & Public Benchmark (WargamesBench)
- [ ] **E12.3** — Modern-Era Extension (v12 Stretch Goal)
- [ ] **E12.4** — v12 Documentation, Paper & Release

### Milestones
- M25: Foundation Model
- M26: v12 Complete

### Key Research Questions
- Can a single foundation model generalise across battalion, brigade, and corps echelons?
- Does multi-task training on all scenario types improve zero-shot generalisation?
- Can WFM-1 transfer to a new historical era (WW1) with < 50k fine-tuning steps?

### Sprint Schedule (v12)
| Sprint | Weeks | Focus | Epics |
|---|---|---|---|
| S37 | 73–74 | WFM-1 architecture + multi-task training | E12.1 |
| S38 | 75–76 | Open platform + paper + v12 LTS release | E12.2, E12.3, E12.4 |

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

### v5 Sprints (Complete ✅)
| Sprint | Weeks | Focus |
|---|---|---|
| S23 (v5) | — | Human interface + COA generator |
| S24 (v5) | — | Explainability + historical validation |
| S25 (v5) | — | Policy export + v5 release |

### v6 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S23 | 45–46 | Terrain engine + weapon system |
| S24 | 47–48 | Formations + morale + rout |
| S25 | 49–50 | Logistics + weather + v6 release |

### v7 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S26 | 51–52 | Corps env + road network + objectives |
| S27 | 53–54 | Strategic supply + corps league |
| S28 | 55–56 | v7 release |

### v8 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S29 | 57–58 | Entity encoder + transformer policy |
| S30 | 59–60 | Memory module + scaling study + v8 release |

### v9 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S31 | 61–62 | Web interface + AI policy server |
| S32 | 63–64 | COA tool + AAR + v9 release |

### v10 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S33 | 65–66 | Naval units + coastal operations |
| S34 | 67–68 | Cavalry corps + grand battery + v10 release |

### v11 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S35 | 69–70 | Historical database + GIS import |
| S36 | 71–72 | Expert demonstrations + BC pre-training + v11 release |

### v12 Sprints (Planned)
| Sprint | Weeks | Focus |
|---|---|---|
| S37 | 73–74 | WFM-1 architecture + multi-task training |
| S38 | 75–76 | Open platform + paper + v12 LTS release |
