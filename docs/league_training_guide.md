# League Training Guide

> **v4 — AlphaStar-style League Training for Wargames**
> Covers agent types, matchmaking, Nash equilibrium sampling, strategy
> diversity, and distributed execution.

---

## Overview

v4 introduces an AlphaStar-inspired league training system.  Instead of a
single policy iterating against itself, a *league* of specialised agents
co-evolves so that each role applies selection pressure on the others, driving
the main agent toward a robust Nash equilibrium strategy.

The league is built on the v2 MAPPO foundation, using `MultiBattalionEnv`
with MAPPO policies directly at the battalion level.  It is compatible with
the v3 HRL stack as a source of frozen battalion sub-policies.  The league
infrastructure sits above that, managing which agent checkpoint plays against
which during training rollouts.

---

## Agent Types

Three AlphaStar-style roles are defined in `training/league/agent_pool.py`
via the `AgentType` enum.

### MAIN_AGENT

The primary policy that the league is designed to strengthen.  Trains via
PFSP (see below) against *all* league members — main agents, main exploiters,
and league exploiters.  Periodically snapshots itself to the shared pool so
that exploiters can target its past strategies.

Training entry-point: `training/league/train_main_agent.py`  
Config: `configs/league/main_agent.yaml`  
W&B metric namespaces: `elo/main_agent`, `matchup/win_rate/*`, `train/*`, `league/*`

### MAIN_EXPLOITER

A specialist agent that trains **exclusively against the latest main agent
snapshot**.  Its goal is to expose weaknesses in the current main agent
strategy.  When its performance against the main agent deteriorates (rolling
win rate drops below a reset threshold) it is reset via orthogonal
re-initialisation, forcing it to search for a different exploit.  This cycling
pressure prevents the main agent from converging to a strategy that is merely
robust to a fixed set of stale or underperforming exploiters.

Training entry-point: `training/league/train_exploiter.py`  
Config: `configs/league/main_exploiter.yaml`  
W&B project key prefix: `exploiter/`

### LEAGUE_EXPLOITER

A generalist exploiter that trains against *all* current league members
(same matchup rules as MAIN_AGENT).  It tracks which league strategies are
exploitable at a broad level and feeds exploitability metrics back into the
league through Nash exploitability computation.

Training entry-point: `training/league/train_league_exploiter.py`  
Config: `configs/league/league_exploiter.yaml`  
W&B project key prefix: `league_exploiter/`

---

## Agent Pool

`training/league/agent_pool.py` — `AgentPool`

The pool maintains a JSON manifest on disk (`checkpoints/league/*/pool.json`
by default).  Each entry is an `AgentRecord` with the following fields:

| Field | Type | Description |
|---|---|---|
| `agent_id` | `str` | Unique identifier (UUID4 by default) |
| `agent_type` | `AgentType` | `main_agent`, `main_exploiter`, or `league_exploiter` |
| `snapshot_path` | `str` | Path to the saved agent snapshot (e.g., `.pt` / `.zip` or directory) |
| `version` | `int` | Monotonically increasing version number within the pool |
| `created_at` | `float` | Unix timestamp when this snapshot was added to the pool |
| `metadata` | `dict` | Arbitrary extra metadata (e.g., training step, W&B run ID) |

Pool mutations (`add`, `update`, `remove`) rewrite the manifest atomically
via a `.tmp` file.  The pool is **not** safe for concurrent multi-process
writes without external locking.

---

## Match Database

`training/league/match_database.py` — `MatchDatabase`

Stores match outcomes as JSONL (one JSON object per line) so the file can be
appended without rewriting.  Each line is a `MatchResult` record with fields:
`match_id`, `agent_id`, `opponent_id`, `outcome`, `timestamp`, and `metadata`,
where `outcome` is from the perspective of `agent_id` (a float in `[0, 1]`:
`1.0` = win, `0.5` = draw, `0.0` = loss).  The `win_rates_for(agent_id)`
method returns a mapping of `{opponent_id: win_rate}` computed in a single
pass over the match history, avoiding repeated O(n) scans.

---

## Matchmaking — PFSP

`training/league/matchmaker.py` — `LeagueMatchmaker`

**Prioritized Fictitious Self-Play (PFSP)** is the default opponent sampling
strategy.  For a focal agent *A* the probability of selecting opponent *O* is:

```
P(O | A) ∝ f(win_rate(A, O))
```

The default (hard-first) weight function is `f(w) = 1 − w`, which biases
sampling toward opponents the focal agent currently struggles against.

When no match history exists for a `(focal, opponent)` pair the win rate is
assumed to be `0.5`, giving a neutral weight of `0.5` under the default
function.

### Matchup Rules

| Agent Role | Eligible Opponents |
|---|---|
| MAIN_AGENT | All league members |
| MAIN_EXPLOITER | MAIN_AGENT only |
| LEAGUE_EXPLOITER | All league members |

### Customising the Weight Function

The weight function can be swapped at runtime:

```python
from training.league.matchmaker import LeagueMatchmaker

def soft_first(win_rate: float) -> float:
    """Prefer easy opponents (curriculum-style warm-up)."""
    return win_rate

matchmaker.set_weight_function(soft_first)

# Revert to hard-first default:
matchmaker.set_weight_function(None)
```

A temperature-scaled version is used during main agent training:

```python
from training.league.train_main_agent import make_pfsp_weight_fn

matchmaker.set_weight_function(make_pfsp_weight_fn(T=1.0))
```

---

## Nash Equilibrium Sampling

`training/league/nash.py`

### Why Nash Sampling?

PFSP adapts opponent selection to the *current* focal agent's win rates, but
it does not account for the global strategic landscape across the whole league.
Nash equilibrium sampling assigns each agent a probability proportional to its
strategic importance in the league payoff matrix, ensuring the main agent
practices against the full distribution of relevant strategies — not just the
ones it currently loses to.

### Computing the Nash Distribution

```python
from training.league.nash import compute_nash_distribution, build_payoff_matrix

# Build the payoff matrix from match history
payoff = build_payoff_matrix(agent_ids, win_rate_callable)

# Solve for the Nash distribution (LP + regret matching)
# Returns a 1-D NumPy probability array of shape (N,)
nash_probs = compute_nash_distribution(payoff)

# Convert to {agent_id: probability} mapping for use with the matchmaker
nash_dist = dict(zip(agent_ids, nash_probs.tolist()))
```

`build_payoff_matrix` accepts a callable `win_rate(agent_a, agent_b) → float`
(typically `match_database.win_rate`) and returns a square NumPy array indexed
by position in `agent_ids`.

`compute_nash_distribution` uses linear programming (via SciPy) followed by
regret-matching to find the mixed Nash equilibrium of the symmetric two-player
zero-sum game defined by the payoff matrix.  It returns a normalised 1-D
NumPy array of shape `(N,)`.  Zip with `agent_ids` to form an
`{agent_id: probability}` mapping.

### Nash Entropy

```python
from training.league.nash import nash_entropy

entropy = nash_entropy(nash_probs)  # nats
```

A high Nash entropy means the league has a diverse equilibrium — no single
strategy dominates.  This value is logged to W&B as `league/nash_entropy`.

### Activating Nash Sampling in the Matchmaker

```python
# nash_dist is {agent_id: probability} — see "Computing the Nash Distribution" above
matchmaker.set_nash_weights(nash_dist)
# Revert to PFSP:
matchmaker.set_nash_weights(None)
```

When Nash weights are set, `LeagueMatchmaker.select_opponent` draws opponents
from the Nash distribution (filtered to eligible candidates) instead of PFSP.

---

## Strategy Diversity Metrics

`training/league/diversity.py`

Diversity is measured by embedding each agent's rollout trajectory and
computing pairwise distances in embedding space.

### Trajectory Embedding

`embed_trajectory(batch: TrajectoryBatch) → np.ndarray`

Each trajectory is embedded as a concatenation of:
1. **Action histogram** — normalised frequency of discrete action buckets.
2. **Position heatmap** — 2D occupancy grid (normalised by map dimensions).
3. **Movement statistics** — mean speed, heading variance, formation spread.

The resulting vector is L2-normalised before distance computation.

### Diversity Score

```python
from training.league.diversity import diversity_score, pairwise_cosine_distances

embeddings = [embed_trajectory(b) for b in batches]
distances = pairwise_cosine_distances(embeddings)
score = diversity_score(distances)  # returns mean, min, and median
```

The `DiversityTracker` class wraps the above into a stateful tracker that
accumulates trajectory batches over training and logs `league/diversity_score`
to W&B.

---

## Distributed Training

`training/league/distributed_runner.py`

v4 supports parallel rollout collection across multiple workers via Ray.

### Components

| Class / Function | Description |
|---|---|
| `RemoteMultiBattalionEnv` | `@ray.remote` actor wrapping `MultiBattalionEnv` |
| `make_remote_envs(n)` | Factory that spawns *n* remote environment actors |
| `DistributedRolloutRunner` | Manages a pool of remote envs, collects `RolloutResult`s |
| `benchmark(n_envs, n_steps)` | Throughput benchmark utility |

Ray cluster configuration: `configs/distributed/ray_cluster.yaml`  
Smoke test: `.github/workflows/ray_smoke_test.yml`

### Usage

```python
from training.league.distributed_runner import DistributedRolloutRunner, make_remote_envs

envs = make_remote_envs(n=8)
runner = DistributedRolloutRunner(envs)
results = runner.collect(policy, n_steps=512)
```

---

## Configuration Reference

All league training configs live in `configs/league/`.

### `main_agent.yaml` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | `5_000_000` | Total training steps |
| `pfsp_temperature` | `1.0` | Temperature for PFSP weight function |
| `pool_max_size` | `200` | Maximum snapshots retained in pool |
| `league.snapshot_freq` | `100_000` | Steps between pool snapshots |

### `main_exploiter.yaml` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | `3_000_000` | Total training steps |
| `reset_win_rate_threshold` | `0.30` | Rolling WR below which exploiter resets |
| `reset_window_size` | `5` | Window size for rolling win-rate check |

### `league_exploiter.yaml` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | `3_000_000` | Total training steps |
| `pfsp_temperature` | `1.0` | Temperature for PFSP weight function |
| `reset_win_rate_threshold` | `0.30` | Rolling WR below which exploiter resets |
| `pool_max_size` | `200` | Maximum snapshots retained in pool |

---

## W&B Metrics

| Metric | Source |
|---|---|
| `elo/main_agent` | Elo rating updated after each evaluation episode |
| `exploiter/rolling_win_rate_vs_main` | Rolling win rate of main exploiter vs. main agent |
| `league_exploiter/exploitability` | Nash exploitability score |
| `league/nash_entropy` | Shannon entropy (nats) of the Nash distribution |
| `league/diversity_score` | Mean cosine distance between agent trajectory embeddings |

---

## Further Reading

- `docs/v4_architecture.md` — Component diagram for the v4 league system
- `docs/hrl_architecture.md` — Underlying v3 HRL battalion/brigade/division hierarchy
- `docs/multi_agent_guide.md` — v2 MAPPO multi-agent foundation
- [AlphaStar: Mastering the Real-Time Strategy Game StarCraft II](https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii)
