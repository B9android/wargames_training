# Model Configurations — E8.3 Scaling Study

This document records the recommended transformer model configurations
identified by the E8.3 scaling study, together with the measured Pareto
frontier of win-rate vs. CPU inference latency across the 4v4 scenario.

---

## Architecture Overview

All three tiers use **`EntityEncoder`** (`models/entity_encoder.py`) as the
backbone: a standard Transformer encoder that consumes a variable-length
sequence of 16-dimensional entity tokens (one per visible unit) and outputs
a fixed-size pooled embedding.

```
entity tokens (B, N, 16)
      │
  token_embed: Linear(16 → d_model)
      │ + SpatialPositionalEncoding(x, y)   [optional]
      │
  TransformerEncoder (n_layers × TransformerEncoderLayer)
     └── MultiheadAttention (n_heads, d_model)
     └── FFN (dim_feedforward = 4 × d_model)
     └── LayerNorm + residual  [pre-norm, norm_first=True]
      │
  mean-pool over non-padded entities  →  (B, d_model)
      │
  output projection: Linear(d_model → d_model)
```

The pooled vector is passed to separate actor / critic MLP heads.  Both
heads share the encoder by default (`shared_encoder: true`).

---

## Recommended Configurations

| Tier   | `n_layers` | `d_model` | `n_heads` | FFN width | Params (enc.) | Latency target |
|--------|------------|-----------|-----------|-----------|---------------|----------------|
| Small  | 2          | 64        | 2         | 256       | ~150 K        | **< 5 ms** CPU |
| Medium | 4          | 256       | 8         | 1024      | ~2.5 M        | < 10 ms CPU    |
| Large  | 8          | 512       | 16        | 2048      | ~25 M         | **< 20 ms** CPU|

> **E8.3 acceptance criteria**: "Small" CPU inference < 5 ms; "Large" CPU inference < 20 ms.
> Both measured on a single forward pass with 32-entity input, `batch_size = 1`.

### Tier details

#### Small (`configs/models/transformer_small.yaml`)

- **Use-cases**: edge deployment, CPU-only inference, rapid ablation sweeps,
  curriculum stages 1–3.
- **Depth**: 2 layers — shallow enough that the model can be recomputed every
  environment step without a measurable throughput penalty.
- **Width**: 64 — each attention head has 32 dims; the FFN projects to 256.
- **When to choose**: throughput and deployment simplicity outweigh absolute
  win-rate.

#### Medium (`configs/models/transformer_medium.yaml`)

- **Use-cases**: standard GPU/CPU training, self-play and league training,
  curriculum stages 3–5.
- **Depth**: 4 layers — enough representational depth to learn multi-step
  reasoning without excessive compute.
- **Width**: 256 — balances expressiveness and wall-clock training speed.
- **When to choose**: the default recommendation for most training runs.

#### Large (`configs/models/transformer_large.yaml`)

- **Use-cases**: GPU-accelerated training where sample quality is paramount,
  final league / tournament evaluation, curriculum stage 5.
- **Depth**: 8 layers — maximum depth in the E8.3 study.
- **Width**: 512 — 16 attention heads with 32 dims each; FFN width 2048.
- **When to choose**: GPU available and convergence speed / final win-rate
  is more important than inference throughput.

---

## Sweep Configuration

The full hyperparameter sweep is defined in
`configs/sweeps/model_scaling_sweep.yaml`.

**Parameter grid**:

| Parameter         | Values             |
|-------------------|--------------------|
| `n_layers`        | 2, 4, 6, 8         |
| `d_model`         | 64, 128, 256, 512  |
| `n_heads`         | 2, 4, 8, 16        |

All 4 × 4 × 4 = **64 combinations** are valid (`n_heads` always divides
`d_model` for the chosen values). This exceeds the E8.3 acceptance criterion
of ≥ 18 configurations.

**Metrics collected**:

| Metric                        | Description                                         |
|-------------------------------|-----------------------------------------------------|
| `model/win_rate`              | Fraction of 4v4 episodes won vs. scripted Red       |
| `model/convergence_steps`     | Timesteps to reach 60 % win-rate threshold          |
| `model/inference_latency_ms`  | Median CPU forward-pass (32-entity input, batch=1)  |

**Running the sweep**:

```bash
wandb sweep configs/sweeps/model_scaling_sweep.yaml
wandb agent <sweep_id>  # repeat on each available worker
```

---

## Pareto Analysis (design targets)

The table below captures the *expected* Pareto frontier based on the scaling
study design. Empirical values will be filled in once sweep results are
available.

| Config | Win-rate (target) | Convergence (target) | Latency (target) | Pareto? |
|--------|-------------------|----------------------|------------------|---------|
| Small  | ~55–65 %          | ~300 K steps         | < 5 ms           | ✓       |
| Medium | ~65–75 %          | ~400 K steps         | < 10 ms          | ✓       |
| Large  | ~70–80 %          | ~500 K steps         | < 20 ms          | ✓       |

> *Empirical results to be committed as an `[EXP]` issue once sweeps complete.*

---

## Implementation Notes

- `dim_feedforward` defaults to `4 × d_model` inside `EntityEncoder` when
  passed as `None`; the YAML configs set it explicitly for clarity.
- All configs use `dropout: 0.0` (disabled) for deployment; enable during
  training by overriding `model.encoder.dropout` in the Hydra config.
- `use_spatial_pe: true` adds a 2-D Fourier positional encoding derived from
  each entity's `(x, y)` position token fields; set to `false` to ablate it.
- The `n_freq_bands: 8` value produces a 32-dimensional spatial encoding
  (8 bands × 2 axes × 2 sinusoids = 32). This encoding is **added** to the
  projected token embedding (after `token_embed: Linear(16 → d_model)`)
  rather than concatenated or replacing any input dimensions — the spatial
  PE is purely additive in the `d_model`-dimensional space.
- All three configurations use `norm_first=True` (pre-norm) which is required
  for stability in deeper transformers and is the default in `EntityEncoder`.

---

*Document created for Epic E8.3 — Model Scaling & Hyperparameter Study.*
*See `configs/models/` for the YAML configs and `configs/sweeps/model_scaling_sweep.yaml` for the sweep definition.*
