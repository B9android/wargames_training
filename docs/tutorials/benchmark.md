# WargamesBench Guide

WargamesBench is the standardised 20-scenario evaluation suite for
reproducible comparison of wargame AI methods.

## The 20 Canonical Scenarios

The scenarios are fixed and must not be modified.  They span:

| Category | Scenarios |
|----------|-----------|
| Symmetric (all weather) | 5 |
| Asymmetric force ratios | 4 |
| Hilly terrain | 2 |
| Dense forest | 2 |
| Large-scale engagement | 2 |
| Small skirmish | 2 |
| Defender's advantage | 2 |
| Low visibility | 1 |

## Reproducibility

Results are reproducible within ± 2 % win rate when:

1. `n_eval_episodes ≥ 100`
2. The same `seed` values in `BENCH_SCENARIOS` are used without modification.
3. Your policy is deterministic (`policy.predict(obs, deterministic=True)`).

## Submitting to the Leaderboard

1. Run the canonical benchmark:

```bash
wargames-bench --episodes 100 --label "MyPolicy v1.0"
```

2. The leaderboard report is written to `docs/wargames_bench_leaderboard.md`.

3. Open a PR appending your results row to that file.

4. Include in the PR: Git SHA of your checkpoint, W&B run URL, and
   the WargamesBench version (`git describe --tags`).

## Programmatic API

```python
from benchmarks import WargamesBench, BenchConfig, BenchSummary

# Evaluate a custom policy
def my_policy(obs):
    return obs[:3]   # toy: return first 3 elements as action

cfg = BenchConfig(n_eval_episodes=100, baseline_label="my_policy_v1")
bench = WargamesBench(cfg)
summary: BenchSummary = bench.run(policy=my_policy)
print(f"Mean win rate: {summary.mean_win_rate:.1%}")

# Check reproducibility against a second run
summary2 = bench.run(policy=my_policy)
assert summary.is_reproducible(summary2), "Results differ by > 2%!"
```

## Baseline Policies

The built-in scripted baseline (`policy=None`) advances straight forward
each step.  It provides a lower bound for comparison.
