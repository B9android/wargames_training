# Wargames Training — Project Roadmap

> **Living document.** Updated by agents and maintainers.
> Last updated: auto-updated by weekly report agent.

---

## Version Overview

| Version | Theme | Status | Target |
|---|---|---|---|
| **v1** | Foundation — 1v1 battalion | 🔨 Active | M4 |
| **v2** | Multi-Agent — MARL 2v2+ | 📋 Planned | M6 |
| **v3** | Hierarchy — Brigade/Division HRL | 📋 Planned | M8 |
| **v4** | League — AlphaStar-style training | 📋 Planned | M10 |
| **v5** | Real-World Interface & Analysis | 🔮 Future | TBD |

---

## v1: Foundation (Active)

**Goal:** A single battalion agent that reliably defeats scripted opponents
in 1v1 continuous 2D battles, generalizes across randomized parameters,
and has a working self-play loop.

### Epics
- [ ] **E1.1** — Project Bootstrap & Tooling
- [ ] **E1.2** — Core Simulation Engine
- [ ] **E1.3** — Gymnasium Environment (1v1)
- [ ] **E1.4** — Baseline Training Loop (PPO + scripted opponent)
- [ ] **E1.5** — Terrain & Environmental Randomization
- [ ] **E1.6** — Reward Shaping & Curriculum Design
- [ ] **E1.7** — Self-Play Implementation
- [ ] **E1.8** — Evaluation Framework & Elo Tracking
- [ ] **E1.9** — Visualization & Replay System
- [ ] **E1.10** — v1 Documentation & Release

### Milestones
- M0: Project Bootstrap
- M1: 1v1 Competence
- M2: Terrain & Generalization
- M3: Self-Play Baseline
- M4: v1 Complete

---

## v2: Multi-Agent (Planned)

**Goal:** Multiple battalions per side coordinate using MAPPO.
Emergent flanking, fire concentration, and mutual support.

### Epics
- [ ] **E2.1** — PettingZoo Multi-Agent Environment
- [ ] **E2.2** — MAPPO Implementation (Centralized Critic)
- [ ] **E2.3** — 2v2 Curriculum
- [ ] **E2.4** — Coordination Metrics & Analysis
- [ ] **E2.5** — Scale to NvN (up to 6v6)
- [ ] **E2.6** — Multi-Agent Self-Play
- [ ] **E2.7** — v2 Documentation & Release

### Milestones
- M5: 2v2 MARL
- M6: v2 Complete

### Key Research Questions
- Does parameter sharing (shared policy across all battalions) outperform
  independent policies?
- What observation radius produces optimal coordination without
  information overload?
- Does emergent flanking behavior appear without explicit reward for it?

---

## v3: Hierarchical RL (Planned)

**Goal:** Brigade and division commanders issue macro-commands to frozen
battalion policies. HRL architecture matching Black (NPS 2024).

### Epics
- [ ] **E3.1** — SMDP / Options Framework
- [ ] **E3.2** — Brigade Commander Layer
- [ ] **E3.3** — Division Commander Layer
- [ ] **E3.4** — Hierarchical Curriculum (bottom-up training)
- [ ] **E3.5** — Temporal Abstraction Tuning
- [ ] **E3.6** — Multi-Model Policy Library (per echelon)
- [ ] **E3.7** — HRL Evaluation vs. Flat MARL
- [ ] **E3.8** — v3 Documentation & Release

### Milestones
- M7: HRL Battalion→Brigade
- M8: v3 Complete

### Key Research Questions
- Does hierarchical decomposition outperform flat MARL at scale?
- What is the optimal temporal abstraction ratio between echelons?
- Can brigade commanders discover novel operational maneuvers?

---

## v4: League Training (Planned)

**Goal:** AlphaStar-style league with main agents, exploiters, and
league exploiters. Nash equilibrium sampling. Strategy diversity metrics.

### Epics
- [ ] **E4.1** — League Infrastructure (agent pool, matchmaking)
- [ ] **E4.2** — Main Agent Training Loop
- [ ] **E4.3** — Main Exploiter Agents
- [ ] **E4.4** — League Exploiter Agents
- [ ] **E4.5** — Nash Distribution Sampling
- [ ] **E4.6** — Strategy Diversity Metrics
- [ ] **E4.7** — Distributed Training (Ray/RLlib)
- [ ] **E4.8** — v4 Documentation & Release

### Milestones
- M9: League Training
- M10: v4 Complete

---

## v5: Analysis & Interface (Future)

**Goal:** Turn the trained system into a useful wargaming tool.
Human-vs-AI play, COA analysis, strategy visualization.

### Potential Epics
- [ ] **E5.1** — Human-playable interface (web or desktop)
- [ ] **E5.2** — Course of Action (COA) generator
- [ ] **E5.3** — Strategy explainability (attention visualization)
- [ ] **E5.4** — Historical scenario validation
- [ ] **E5.5** — Export trained policies for deployment

---

## Sprint Schedule

Sprints are 2 weeks. Sprint planning happens every other Monday.

| Sprint | Dates | Focus |
|---|---|---|
| S01 | Week 1–2 | Bootstrap, env setup, first training run |
| S02 | Week 3–4 | Sim engine, combat resolution |
| S03 | Week 5–6 | Reward shaping, curriculum L1–L3 |
| S04 | Week 7–8 | Terrain system, curriculum L4 |
| S05 | Week 9–10 | Generalization, randomized params |
| S06 | Week 11–12 | Self-play infrastructure |
| S07 | Week 13–14 | Self-play training, Elo tracking |
| S08 | Week 15–16 | Evaluation, visualization, v1 polish |
| S09 | Week 17–18 | v1 release, v2 planning |
| ... | ... | ... |
