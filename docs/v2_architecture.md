# v2 System Architecture

This document describes the v2 multi-agent system architecture for
wargames_training.  The v2 system implements Multi-Agent Proximal Policy
Optimization (MAPPO) with Centralized Training, Decentralized Execution
(CTDE) for NvN battalion combat.

---

## High-Level Component Diagram

```mermaid
graph TD
    subgraph Configs["⚙️ Configuration (Hydra)"]
        CFG_MAPPO["experiment_mappo_2v2.yaml"]
        CFG_CURR["curriculum_2v2.yaml"]
        CFG_SCENE["scenarios/*.yaml"]
    end

    subgraph Env["🗺️ Environment Layer"]
        MBE["MultiBattalionEnv\n(PettingZoo ParallelEnv)"]
        SIM["Simulation Engine\n(envs/sim/)"]
        FOW["Fog of War\n(visibility_radius)"]
        METRICS["Coordination Metrics\n(envs/metrics/coordination.py)"]
        MBE --> SIM
        MBE --> FOW
        MBE --> METRICS
    end

    subgraph Policy["🧠 Policy Layer (models/)"]
        ACTOR["MAPPOActor\nLocal obs → Gaussian action\nobs_dim → [128,64] → action_dim×2"]
        CRITIC["MAPPOCritic\nGlobal state → value\nstate_dim → [128,64] → 1"]
        MAPPO_POL["MAPPOPolicy\n(wraps Actor + Critic)"]
        ACTOR --> MAPPO_POL
        CRITIC --> MAPPO_POL
    end

    subgraph Training["🏋️ Training Layer (training/)"]
        TRAINER["MAPPOTrainer\n(train_mappo.py)"]
        BUFFER["MAPPORolloutBuffer\n(per-agent GAE)"]
        CURR_SCHED["CurriculumScheduler\n1v1 → 2v1 → 2v2"]
        SELF_PLAY["TeamOpponentPool\n(self_play.py)"]
        ELO["TeamEloRegistry\n(elo.py)"]
        TRAINER --> BUFFER
        TRAINER --> CURR_SCHED
        TRAINER --> SELF_PLAY
        SELF_PLAY --> ELO
    end

    subgraph Logging["📊 Logging & Checkpoints"]
        WANDB["W&B\n(per-agent rewards,\ncoordination metrics,\nElo ratings)"]
        CKPT["Checkpoints\n(.pt snapshots\nfor opponent pool)"]
    end

    CFG_MAPPO --> TRAINER
    CFG_CURR  --> CURR_SCHED
    CFG_SCENE --> MBE

    MBE -- "obs[agent_id], state()" --> TRAINER
    MAPPO_POL -- "action[agent_id]" --> MBE
    TRAINER --> WANDB
    TRAINER --> CKPT
    CKPT --> SELF_PLAY
```

---

## Data Flow: Single Training Step

```mermaid
sequenceDiagram
    participant T as MAPPOTrainer
    participant E as MultiBattalionEnv
    participant A as MAPPOActor (shared)
    participant C as MAPPOCritic
    participant B as MAPPORolloutBuffer

    T->>E: reset()
    loop Rollout collection (n_steps)
        E-->>T: obs[blue_0], obs[blue_1], state()
        T->>A: forward(obs[blue_0])
        A-->>T: action[blue_0], log_prob[blue_0]
        T->>A: forward(obs[blue_1])
        A-->>T: action[blue_1], log_prob[blue_1]
        T->>C: forward(state())
        C-->>T: value
        T->>E: step(actions)
        E-->>T: rewards, terminated, truncated
        T->>B: add(obs, action, reward, value, log_prob)
    end
    T->>B: compute_returns_and_advantages() [GAE-λ]
    loop PPO epochs (n_epochs)
        T->>B: sample_minibatch(batch_size)
        B-->>T: batch
        T->>A: evaluate_actions(obs_batch)
        T->>C: evaluate(state_batch)
        Note over T: Compute L_CLIP + v_f coef × L_VF − ent_coef × H
        T->>T: optimizer.step()
    end
```

---

## Observation & State Tensors

```
Per-Agent Local Observation (obs_dim = 6 + 5*(n_total-1) + 1)
┌──────────────────────────────────────────────────────────────┐
│ Self state (6)   │ x/W │ y/H │ cos θ │ sin θ │ hp │ morale  │
├──────────────────────────────────────────────────────────────┤
│ Other units      │ For each of (n_total - 1) other units:   │
│ (5 per unit)     │ Δx/W │ Δy/H │ cos θ │ sin θ │ hp         │
│                  │ (zeroed if outside visibility_radius)     │
├──────────────────────────────────────────────────────────────┤
│ Terrain (1)      │ cover value at agent position [0, 1]     │
└──────────────────────────────────────────────────────────────┘

Global State for Centralized Critic (state_dim = 6*n_total + 1)
┌──────────────────────────────────────────────────────────────┐
│ All units (6 per unit, unobscured, ordered Blue then Red)   │
│  x/W │ y/H │ cos θ │ sin θ │ hp │ morale                   │
├──────────────────────────────────────────────────────────────┤
│ Step (1)  │ normalized step count [0, 1]                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Curriculum Stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — 1v1                                              │
│  Env: BattalionEnv (v1)                                     │
│  Policy: PPO (or MAPPO with n_blue=1)                       │
│  Opponent: scripted Red (curriculum levels 1–5)             │
│                                                             │
│  Promote when: rolling win-rate ≥ 70% over 100 episodes     │
└───────────────────────┬─────────────────────────────────────┘
                        │  load_v1_weights_into_mappo()
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — 2v1                                              │
│  Env: MultiBattalionEnv (n_blue=2, n_red=1)                 │
│  Policy: MAPPO (shared actor)                               │
│  Opponent: stationary Red                                   │
│                                                             │
│  Promote when: rolling win-rate ≥ 70% over 100 episodes     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — 2v2                                              │
│  Env: MultiBattalionEnv (n_blue=2, n_red=2)                 │
│  Policy: MAPPO (shared actor)                               │
│  Opponent: scripted Red → self-play pool                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-Play Loop

```
┌───────────────────────────────────────────────────────────┐
│                     TeamOpponentPool                      │
│                                                           │
│   Snapshot archive (up to pool_max_size .pt files)        │
│   ┌──────┐  ┌──────┐  ┌──────┐  ...  ┌──────┐           │
│   │ 50k  │  │ 100k │  │ 150k │       │ Nk   │ ← latest  │
│   └──────┘  └──────┘  └──────┘       └──────┘           │
│                                                           │
│   Sampling: uniform (or Elo-weighted when enabled)        │
└───────────────────┬───────────────────────────────────────┘
                    │ sample opponent
                    ▼
          ┌──────────────────┐        ┌──────────────────┐
          │  Blue (MAPPO)    │◄──────►│  Red (snapshot)  │
          │  current policy  │ battle │  frozen policy   │
          └────────┬─────────┘        └──────────────────┘
                   │ episode result
                   ▼
          ┌──────────────────┐
          │  TeamEloRegistry │ ← updates Elo ratings
          └──────────────────┘
                   │ every snapshot_freq steps
                   ▼
          ┌──────────────────┐
          │ Save snapshot to │
          │ opponent pool    │
          └──────────────────┘
```

---

## File Map

```
wargames_training/
├── envs/
│   ├── multi_battalion_env.py    # PettingZoo ParallelEnv (NvN)
│   ├── battalion_env.py          # Gymnasium 1v1 env (v1, used in curriculum)
│   ├── metrics/
│   │   └── coordination.py       # flanking_ratio, fire_concentration, mutual_support
│   └── sim/
│       ├── battalion.py          # Battalion state, movement, morale
│       ├── combat.py             # Fire resolution, damage, routing
│       ├── engine.py             # Step orchestration
│       └── terrain.py            # Terrain cover and elevation
├── models/
│   ├── mappo_policy.py           # MAPPOActor, MAPPOCritic, MAPPOPolicy
│   └── mlp_policy.py             # BattalionMlpPolicy (v1 PPO)
├── training/
│   ├── train_mappo.py            # MAPPOTrainer, MAPPORolloutBuffer, Hydra entry-point
│   ├── curriculum_scheduler.py   # CurriculumScheduler, load_v1_weights_into_mappo
│   ├── self_play.py              # OpponentPool, TeamOpponentPool, TeamEloRegistry
│   ├── elo.py                    # EloRegistry, TeamEloRegistry
│   ├── train.py                  # v1 PPO training pipeline
│   └── evaluate.py               # CLI evaluation script
├── configs/
│   ├── experiment_mappo_2v2.yaml # Reference 2v2 MAPPO experiment
│   ├── curriculum_2v2.yaml       # Three-stage curriculum config
│   └── scenarios/
│       ├── 2v1.yaml
│       ├── 2v2.yaml
│       ├── 3v3.yaml
│       ├── 4v4.yaml
│       └── 6v6.yaml
└── docs/
    ├── multi_agent_guide.md      # MAPPO setup and usage guide (this doc's companion)
    ├── v2_architecture.md        # This document
    └── scaling_notes.md          # NvN dimensionality and performance analysis
```

---

## See Also

- `docs/multi_agent_guide.md` — Step-by-step setup and training guide
- `docs/scaling_notes.md` — NvN scaling analysis
- `docs/ENVIRONMENT_SPEC.md` — Full v1 environment specification
