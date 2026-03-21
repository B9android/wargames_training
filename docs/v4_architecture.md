# v4 Architecture — League Training System

> **Version:** v4.0.0  
> **Theme:** AlphaStar-style League Training

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LEAGUE TRAINING SYSTEM                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │  MAIN AGENT  │    │  MAIN EXPLOITER  │    │ LEAGUE EXPLOITER  │ │
│  │              │    │                  │    │                   │ │
│  │ PFSP vs all  │    │ Targets latest   │    │ PFSP vs all       │ │
│  │ league       │    │ main agent only  │    │ league            │ │
│  │ members      │    │ Resets on high   │    │ members           │ │
│  │              │    │ win rate         │    │                   │ │
│  └──────┬───────┘    └────────┬─────────┘    └─────────┬─────────┘ │
│         │  snapshots          │  snapshots              │ snapshots │
│         ▼                     ▼                         ▼           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      AGENT POOL                             │   │
│  │  AgentRecord × N  (JSON manifest, persisted to disk)        │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│         ┌─────────────────────┴──────────────────────┐             │
│         ▼                                             ▼             │
│  ┌─────────────────┐                   ┌──────────────────────┐    │
│  │  MATCH DATABASE │                   │  NASH SOLVER         │    │
│  │  (JSONL log of  │──── win_rates ───▶│  build_payoff_matrix │    │
│  │  match results) │                   │  compute_nash_distribution() │    │
│  └─────────────────┘                   └──────────┬───────────┘    │
│                                                    │ nash_weights   │
│                                                    ▼               │
│                                         ┌──────────────────────┐   │
│                                         │  LEAGUE MATCHMAKER   │   │
│                                         │  (PFSP or Nash)      │   │
│                                         │  select_opponent()   │   │
│                                         └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

```
training/league/
│
├── agent_pool.py          AgentPool ─── AgentRecord ─── AgentType enum
│                          (JSON manifest, atomic writes via .tmp)
│
├── match_database.py      MatchDatabase ─── JSONL append-log
│                          win_rate(), win_rates_for() (single-pass)
│
├── matchmaker.py          LeagueMatchmaker
│                          ├── select_opponent()   ← PFSP or Nash
│                          ├── set_weight_function()
│                          ├── set_nash_weights()
│                          └── opponent_probabilities()
│
├── nash.py                build_payoff_matrix()
│                          compute_nash_distribution()  ← LP + regret matching
│                          nash_entropy()               ← Shannon entropy (nats)
│
├── diversity.py           TrajectoryBatch
│                          embed_trajectory()    ← histogram + heatmap + stats
│                          pairwise_cosine_distances()
│                          diversity_score()     ← mean / min / median
│                          DiversityTracker
│
├── distributed_runner.py  RemoteMultiBattalionEnv  (@ray.remote)
│                          make_remote_envs(n)
│                          DistributedRolloutRunner
│                          RolloutResult
│                          benchmark()
│
├── train_main_agent.py    MainAgentTrainer
│                          ├── PFSP matchmaking
│                          ├── MAPPO training loop
│                          ├── Elo rating updates
│                          └── Pool snapshot saving
│
├── train_exploiter.py     MainExploiterTrainer
│                          ├── Targets latest MAIN_AGENT snapshot
│                          ├── Rolling win-rate tracking
│                          ├── Orthogonal re-initialisation on reset
│                          └── MAIN_EXPLOITER pool snapshots
│
└── train_league_exploiter.py  LeagueExploiterTrainer
                               ├── PFSP vs full pool
                               ├── Nash exploitability computation
                               └── LEAGUE_EXPLOITER pool snapshots
```

---

## Data Flow

```
                     ┌──────────────┐
                     │  Rollout     │
      policy ───────▶│  Collection  │◀─── opponent policy (from pool)
                     │  (env step)  │
                     └──────┬───────┘
                            │  trajectory
                            ▼
               ┌─────────────────────────┐
               │   Policy Update (MAPPO) │
               └─────────────┬───────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
       MatchDatabase    AgentPool       DiversityTracker
       (record result)  (snapshot)      (embed trajectory)
             │               │               │
             └───────────────▼───────────────┘
                      W&B Logging
                  league/nash_entropy
                  league/diversity_score
                  elo/main_agent
                  exploiter/rolling_win_rate_vs_main
```

---

## Training Roles & Interaction

```
  MAIN EXPLOITER                    LEAGUE EXPLOITER
       │                                   │
       │  forces main agent to             │  ensures no league
       │  defend against targeted          │  strategy is safe
       │  weaknesses                       │  from exploitation
       │                                   │
       └──────────────┬────────────────────┘
                      │
                      ▼
              MAIN AGENT  ◀──── learns to be robust
                      │         against diverse attacks
                      │
                      ▼
              AGENT POOL snapshots
              (historical strategies)
                      │
                      ▼
              PFSP / Nash sampling
              (prioritises hard opponents)
```

---

## Distributed Execution (Ray)

```
  ┌─────────────────────────────────────────────────────────┐
  │                  Ray Cluster                            │
  │                                                         │
  │   Driver Process (DistributedRolloutRunner)             │
  │        │                                                │
  │        ├─▶  RemoteMultiBattalionEnv actor 0             │
  │        ├─▶  RemoteMultiBattalionEnv actor 1             │
  │        ├─▶  RemoteMultiBattalionEnv actor 2             │
  │        │           …                                    │
  │        └─▶  RemoteMultiBattalionEnv actor N-1           │
  │                                                         │
  │   Each actor runs an independent MultiBattalionEnv      │
  │   episode.  Results are gathered as RolloutResult       │
  │   objects and aggregated by the driver.                 │
  └─────────────────────────────────────────────────────────┘
```

---

## Relationship to v1–v3 Architecture

```
v1  BattalionEnv (1v1, PPO, scripted opponent)
        │
v2  MultiBattalionEnv (NvN, MAPPO, shared policy)
        │
v3  BrigadeEnv / DivisionEnv (HRL, frozen sub-policies)
        │
v4  League (AgentPool + MatchDB + Matchmaker + Nash)
        │ wraps the MAPPO policies from v2/v3
        ▼
    Nash-robust main agent policy
```

---

## Key Files

| File | Description |
|---|---|
| `training/league/agent_pool.py` | Agent registry with JSON persistence |
| `training/league/match_database.py` | JSONL match outcome log |
| `training/league/matchmaker.py` | PFSP / Nash opponent selector |
| `training/league/nash.py` | Nash distribution solver and entropy |
| `training/league/diversity.py` | Trajectory embedding and diversity metrics |
| `training/league/distributed_runner.py` | Ray-based parallel rollout collection |
| `training/league/train_main_agent.py` | Main agent PFSP+MAPPO training loop |
| `training/league/train_exploiter.py` | Main exploiter trainer |
| `training/league/train_league_exploiter.py` | League exploiter trainer |
| `configs/league/main_agent.yaml` | Main agent hyperparameters |
| `configs/league/main_exploiter.yaml` | Main exploiter hyperparameters |
| `configs/league/league_exploiter.yaml` | League exploiter hyperparameters |
| `configs/distributed/ray_cluster.yaml` | Ray cluster configuration |
