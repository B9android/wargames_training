# League Training Tutorial

This tutorial walks through running v4 AlphaStar-style league training end-to-end:
bootstrapping the agent pool, configuring the matchmaker, running the main agent
and exploiter training loops, and reading the strategy diversity report.

!!! note "Prerequisites"
    Complete the [Training Guide](training.md) first.  You should have a working
    PPO checkpoint and a passing `pytest tests/ -q` run before continuing.

---

## Overview

League training replaces a single self-play loop with a *league* of
co-evolving specialist agents.  Three roles apply complementary selection
pressure on one another, driving the main agent toward a robust Nash equilibrium
strategy.

| Agent role | Trains against | Purpose |
|---|---|---|
| `MAIN_AGENT` | All league members (PFSP) | Primary policy to strengthen |
| `MAIN_EXPLOITER` | Latest main agent only | Expose current-policy weaknesses |
| `LEAGUE_EXPLOITER` | All league members (PFSP) | Measure & report broad exploitability |

All league infrastructure lives under `training/league/`.  The full reference
is in [`docs/league_training_guide.md`](../league_training_guide.md).

---

## Step 1 — Bootstrap the Agent Pool

The agent pool is a JSON manifest that records every snapshot ever added to the
league.  You must initialise it with at least one starting checkpoint before
any trainer can run.

```python
import tempfile, pathlib
from training.league import AgentPool, AgentType

# Choose a directory for league artefacts.
# In production use: "checkpoints/league/main_agent"
pool_dir = pathlib.Path("checkpoints/league/main_agent")
pool_dir.mkdir(parents=True, exist_ok=True)

pool = AgentPool(pool_manifest=str(pool_dir / "pool.json"))

# Register your pre-trained starting checkpoint.
# The snapshot_path points to a .zip produced by SB3 or a PyTorch .pt file.
record = pool.add(
    snapshot_path="checkpoints/pretrained/battalion_ppo_v1.zip",
    agent_type=AgentType.MAIN_AGENT,
    metadata={"note": "bootstrapped from PPO v1"},
)
print(f"Pool initialised — agent_id: {record.agent_id}, version: {record.version}")
```

The manifest is written atomically to `checkpoints/league/main_agent/pool.json` by the
main agent trainer. Other league trainers maintain their own `pool_manifest` files and
typically read from the main agent pool via a `main_agent_pool_manifest` config setting.
---

## Step 2 — Configure the Matchmaker

`LeagueMatchmaker` selects opponents for each rollout using **Prioritized
Fictitious Self-Play (PFSP)** — it biases sampling toward opponents the focal
agent currently struggles against.

```python
from training.league import LeagueMatchmaker, MatchDatabase, AgentPool, AgentType

pool = AgentPool(pool_manifest="checkpoints/league/main_agent/pool.json")
match_db = MatchDatabase(db_path="checkpoints/league/main_agent/matches.jsonl")

matchmaker = LeagueMatchmaker(agent_pool=pool, match_database=match_db)

# Confirm the matchmaker sees the pool.
agents = pool.list()
print(f"Pool size: {len(agents)} agent(s)")

# Optional: switch to a curriculum warm-up weight function (easy-first).
def prefer_easy_opponents(win_rate: float) -> float:
    """Prefer opponents the focal agent already beats — good for warm-up."""
    return win_rate

matchmaker.set_weight_function(prefer_easy_opponents)

# Revert to hard-first default (recommended for main training):
matchmaker.set_weight_function(None)
```

The matchmaker selects an opponent each time a trainer calls
`matchmaker.select_opponent(focal_agent_id)`.

---

## Step 3 — Run the Main Agent Loop

Launch `MainAgentTrainer` to start main agent training.  It uses PFSP to sample
opponents from the full pool and periodically snapshots itself back into the
pool so that exploiters can target its latest strategy.

### CLI (recommended for full runs)

```bash
python training/league/train_main_agent.py
```

Override any config key via Hydra:

```bash
python training/league/train_main_agent.py \
    training.total_timesteps=5_000_000 \
    league.pfsp_temperature=1.0 \
    wandb.tags='["v4","league","main_agent"]'
```

The configuration file is `configs/league/main_agent.yaml`.  Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `training.total_timesteps` | `5_000_000` | Total training steps |
| `league.pfsp_temperature` | `1.0` | PFSP temperature (1.0 = hard-first) |
| `league.snapshot_freq` | `100_000` | Steps between pool snapshots |
| `league.pool_max_size` | `200` | Maximum snapshots retained |

### Python API

```python
from training.league.train_main_agent import MainAgentTrainer

trainer = MainAgentTrainer(config_path="configs/league/main_agent.yaml")
trainer.train()
```

---

## Step 4 — Run the Exploiter Loops

Run the exploiter processes **in parallel** with the main agent (separate
terminals or processes).  They read the main agent's pool manifest to discover
new snapshots as training progresses.

### Main Exploiter

Targets the *latest* main agent snapshot to expose policy weaknesses.  Resets
itself when its rolling win rate drops below `reset_win_rate_threshold`.

```bash
python training/league/train_exploiter.py
```

Configuration: `configs/league/main_exploiter.yaml`

| Parameter | Default | Description |
|---|---|---|
| `training.total_timesteps` | `3_000_000` | Total training steps |
| `league.reset_win_rate_threshold` | `0.30` | Reset trigger (rolling WR < 30 %) |
| `league.reset_window_size` | `5` | Evaluation window for rolling WR |

### League Exploiter

Trains against *all* pool members (PFSP) and logs a broad exploitability score.

```bash
python training/league/train_league_exploiter.py
```

Configuration: `configs/league/league_exploiter.yaml`

---

## Step 5 — Activate Nash Distribution Sampling

Once the pool has ≥ 3 agents and enough match history, activate Nash sampling so
the matchmaker draws opponents according to the theoretical equilibrium
distribution.

```python
import numpy as np
from training.league import (
    AgentPool, MatchDatabase, LeagueMatchmaker,
    build_payoff_matrix, compute_nash_distribution, nash_entropy,
)

pool = AgentPool(pool_manifest="checkpoints/league/main_agent/pool.json")
match_db = MatchDatabase(db_path="checkpoints/league/main_agent/matches.jsonl")
matchmaker = LeagueMatchmaker(agent_pool=pool, match_database=match_db)

agent_ids = [r.agent_id for r in pool.list()]

# win_rate(a, b) -> float: fraction of games won by a against b.
payoff = build_payoff_matrix(agent_ids, win_rate_fn=match_db.win_rate)

nash_probs = compute_nash_distribution(payoff)           # shape (N,)
nash_dist = dict(zip(agent_ids, nash_probs.tolist()))

entropy = nash_entropy(nash_probs)                        # nats
print(f"Nash entropy: {entropy:.3f} nats  (higher = more diverse equilibrium)")

# Activate Nash-weighted sampling in the matchmaker.
matchmaker.set_nash_weights(nash_dist)

# Revert to PFSP at any time:
# matchmaker.set_nash_weights(None)
```

`nash_entropy` is logged automatically to W&B as `league/nash_entropy` when
using `MainAgentTrainer`.

---

## Step 6 — Read the Diversity Report

The `DiversityTracker` accumulates trajectory batches from each agent and
computes pairwise cosine distances in behavioral embedding space.

```python
import numpy as np
from training.league import DiversityTracker, TrajectoryBatch

tracker = DiversityTracker()

# After collecting evaluation rollouts for each agent, track them:
for agent_id, actions, positions in eval_rollouts:
    batch = TrajectoryBatch(
        actions=actions,       # np.ndarray of shape (T, action_dim)
        positions=positions,   # np.ndarray of shape (T, 2), normalised to [0,1]
        agent_id=agent_id,
    )
    tracker.update(agent_id, batch)

score = tracker.diversity_score()
print(f"League diversity score: {score:.4f}  (0=identical, ~1=highly diverse)")

# Log to W&B manually if needed:
import wandb
wandb.log({"league/diversity_score": score})
```

A diversity score above **0.3** generally indicates a healthy league with
multiple distinct strategy archetypes.  Values below **0.1** suggest premature
convergence — consider lowering `ent_coef` or adding more exploiter restarts.

---

## Minimal Reproducible Example

The following script bootstraps a tiny league and runs a short training round
that completes in **under 5 minutes on CPU**.  Use it to verify your
installation before committing to a full run.

```python
"""
league_smoke_test.py — minimal end-to-end league training check.
Run: python league_smoke_test.py
Expected: prints "Smoke test PASSED" with diversity score and Nash entropy.
"""
import pathlib, tempfile, numpy as np
from training.league import (
    AgentPool, AgentType, MatchDatabase, LeagueMatchmaker,
    build_payoff_matrix, compute_nash_distribution, nash_entropy,
    DiversityTracker, TrajectoryBatch,
)

# ── 1. Bootstrap pool with three synthetic agents ─────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    pool_path  = pathlib.Path(tmp) / "pool.json"
    match_path = pathlib.Path(tmp) / "matches.jsonl"

    pool = AgentPool(pool_manifest=str(pool_path))
    match_db = MatchDatabase(db_path=str(match_path))

    a = pool.add(snapshot_path="/dev/null", agent_type=AgentType.MAIN_AGENT,
                 metadata={"note": "seed"})
    b = pool.add(snapshot_path="/dev/null", agent_type=AgentType.MAIN_EXPLOITER)
    c = pool.add(snapshot_path="/dev/null", agent_type=AgentType.LEAGUE_EXPLOITER)

    assert len(pool.list()) == 3, "Pool should contain 3 agents"

    # ── 2. Record some synthetic match outcomes ───────────────────────────
    pairs = [(a.agent_id, b.agent_id, 0.7),
             (a.agent_id, c.agent_id, 0.6),
             (b.agent_id, c.agent_id, 0.55)]
    for src, dst, outcome in pairs:
        match_db.record(src, dst, outcome)
        match_db.record(dst, src, 1.0 - outcome)

    # ── 3. Matchmaker selects an opponent ─────────────────────────────────
    matchmaker = LeagueMatchmaker(agent_pool=pool, match_database=match_db)
    opponent = matchmaker.select_opponent(a.agent_id)
    assert opponent is not None, "Matchmaker should return an opponent"

    # ── 4. Nash distribution ──────────────────────────────────────────────
    ids = [r.agent_id for r in pool.list()]
    payoff = build_payoff_matrix(ids, win_rate_fn=match_db.win_rate)
    nash_probs = compute_nash_distribution(payoff)
    entropy = nash_entropy(nash_probs)
    assert abs(nash_probs.sum() - 1.0) < 1e-6, "Nash probs must sum to 1"

    # ── 5. Diversity tracking ─────────────────────────────────────────────
    tracker = DiversityTracker()
    rng = np.random.default_rng(0)
    for agent_id in ids:
        batch = TrajectoryBatch(
            actions=rng.standard_normal((50, 4)),
            positions=rng.uniform(0, 1, (50, 2)),
            agent_id=agent_id,
        )
        tracker.update(agent_id, batch)

    div_score = tracker.diversity_score()

    print(f"Nash entropy    : {entropy:.4f} nats")
    print(f"Diversity score : {div_score:.4f}")
    print("Smoke test PASSED ✓")
```

Save as `/tmp/league_smoke_test.py` and run:

```bash
python /tmp/league_smoke_test.py
```

Expected output (exact values will vary slightly):

```
Nash entropy    : 0.9xxx nats
Diversity score : 0.xxxx
Smoke test PASSED ✓
```

---

## Troubleshooting

### Pool file corruption

**Symptom:** `JSONDecodeError` or `KeyError` when loading `pool.json`.

**Cause:** The pool manifest was partially written (e.g., process killed mid-write).

**Fix:** `AgentPool` writes atomically via a `.tmp` file so corruption should be
rare.  If it occurs, inspect the last valid `.tmp` backup (`pool.tmp`):

```bash
ls checkpoints/league/main_agent/pool*
# If pool.tmp exists and is valid JSON, restore it:
mv checkpoints/league/main_agent/pool.tmp \
   checkpoints/league/main_agent/pool.json
```

To validate a manifest programmatically:

```python
import json, pathlib
data = json.loads(pathlib.Path("checkpoints/league/main_agent/pool.json").read_text())
print(f"Manifest OK — {len(data)} agent(s)")
```

---

### Nash solver divergence

**Symptom:** `compute_nash_distribution` returns a uniform distribution and logs
a warning like `"LP solver failed; falling back to regret matching"`.

**Cause:** The payoff matrix contains many `0.5` entries (missing match data),
making the linear programme degenerate.  This is normal early in training.

**Fix:** The function automatically falls back to regret matching, which
converges more slowly but is always stable.  Allow more match history to
accumulate (at least **N²/2** matches for an N-agent pool) before relying on
Nash weights.  You can also increase `n_iterations`:

```python
nash_probs = compute_nash_distribution(payoff, n_iterations=50_000)
```

---

### Exploiter never resets

**Symptom:** The main exploiter's rolling win rate stays high and it never
resets, causing the main agent to over-fit against a single exploit.

**Fix:** Lower `reset_win_rate_threshold` in `configs/league/main_exploiter.yaml`:

```yaml
league:
  reset_win_rate_threshold: 0.40   # reset sooner (default 0.30)
  reset_window_size: 3             # shorter window (default 5)
```

---

### Low diversity score

**Symptom:** `league/diversity_score` stays below 0.1 throughout training.

**Cause:** All agents converge to the same strategy archetype, usually due to a
low entropy bonus or too few exploiter resets.

**Fix options:**

1. Increase `ent_coef` in all league configs (e.g. from `0.01` to `0.03`).
2. Reduce `reset_win_rate_threshold` so exploiters reset more often.
3. Seed the pool with multiple pre-trained checkpoints from different runs.

---

### Processes writing to the same pool concurrently

**Symptom:** `AssertionError` or corrupted match records when running main agent
and exploiters in parallel.

**Cause:** `AgentPool` writes are atomic but not multi-process safe.  The main
agent and exploiters **must not** share the same `pool_manifest` path.  Each
process owns its own pool file; the exploiter uses a separate
`main_agent_pool_manifest` pointer as a *read-only* reference.

**Fix:** Check that `pool_manifest` in `main_exploiter.yaml` differs from the
one in `main_agent.yaml` — this is already the default.

---

## What's Next?

- **[League Training Guide](../league_training_guide.md)** — full API reference
  for every league module
- **[v4 Architecture](../v4_architecture.md)** — component diagram and design
  rationale
- **[Metrics Reference](../metrics_reference.md)** — all W&B metric definitions
- **[HRL Architecture](../hrl_architecture.md)** — using HRL sub-policies inside
  the league
