# Deployment Guide — Exporting & Serving Trained Policies

> **Epic:** E5.5 — Export Trained Policies for Deployment
> **Version:** v5 (post-v4 milestone)

This guide describes how to export a trained policy checkpoint to portable
formats (**ONNX** and **TorchScript**), verify inference parity, serve the
policy via the Docker REST API, and benchmark latency.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Exporting a Policy](#exporting-a-policy)
   - [MAPPOActor / MAPPOCritic](#mappoactoractor--mappocriticcritic)
   - [SB3 MLP Policy (BattalionMlpPolicy)](#sb3-mlp-policy-battalionmlppolicy)
   - [Export formats](#export-formats)
3. [Verifying Inference Parity](#verifying-inference-parity)
4. [Serving the Policy via Docker](#serving-the-policy-via-docker)
   - [Build the image](#build-the-image)
   - [Run the container](#run-the-container)
   - [API reference](#api-reference)
5. [Benchmarking Inference Latency](#benchmarking-inference-latency)
6. [Edge Device & Browser Deployment](#edge-device--browser-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Install the optional export / serving dependencies:

```bash
# ONNX export and runtime
pip install onnx onnxruntime

# Already required by the project
pip install torch stable-baselines3 flask
```

These packages are **not** in the core `requirements.txt` because they are
only needed for deployment, not for training.

---

## Exporting a Policy

The main export script is `scripts/export_policy.py`.  It supports three
model types and two output formats.

### MAPPOActor / MAPPOCritic

Save the actor (or critic) state-dict first:

```python
import torch
from models.mappo_policy import MAPPOActor

actor = MAPPOActor(obs_dim=22, action_dim=3)
# … train or load weights …
torch.save(actor.state_dict(), "checkpoints/actor.pt")
```

Then export:

```bash
# Export to both ONNX and TorchScript
python scripts/export_policy.py \
    --checkpoint checkpoints/actor.pt \
    --model-type mappo_actor \
    --obs-dim 22 \
    --action-dim 3 \
    --output-dir exports/

# Export critic (uses global state as input)
python scripts/export_policy.py \
    --checkpoint checkpoints/critic.pt \
    --model-type mappo_critic \
    --obs-dim 22 \
    --state-dim 25 \
    --output-dir exports/
```

### SB3 MLP Policy (BattalionMlpPolicy)

SB3 models are saved as `.zip` archives by `model.save()`:

```python
from stable_baselines3 import PPO
from models.mlp_policy import BattalionMlpPolicy
from envs.battalion_env import BattalionEnv

env = BattalionEnv()
model = PPO(BattalionMlpPolicy, env)
model.learn(total_timesteps=100_000)
model.save("checkpoints/ppo_battalion")  # writes ppo_battalion.zip
```

Export the actor head:

```bash
python scripts/export_policy.py \
    --checkpoint checkpoints/ppo_battalion.zip \
    --model-type sb3_mlp \
    --obs-dim 12 \
    --action-dim 3 \
    --output-dir exports/
```

### Export formats

| Flag | Output file | Use-case |
|---|---|---|
| `--formats onnx` | `<stem>.onnx` | ONNX.js, ONNX Runtime, edge devices |
| `--formats torchscript` | `<stem>.torchscript.pt` | C++ libtorch, mobile, standalone |
| `--formats onnx torchscript` | both | default — produce both formats |

Additional options:

```
--batch-size N      Dummy-input batch size for tracing (default: 1)
--opset-version N   ONNX opset version (default: 17)
--benchmark         Print CPU latency statistics after export
```

---

## Verifying Inference Parity

Run the dedicated test suite to confirm that exported models produce
outputs identical to the original PyTorch model within **1 × 10⁻⁵**:

```bash
pytest tests/test_policy_export.py -v
```

Key test classes:

| Class | What is verified |
|---|---|
| `TestExportToTorchScript` | File creation, shape, PyTorch parity ≤ 1e-5 |
| `TestExportToOnnx` | File creation, ONNX graph validity, PyTorch parity ≤ 1e-5, dynamic batch |
| `TestExportPolicyOrchestrator` | End-to-end orchestrator for all model types and formats |
| `TestBenchmarkInference` | Latency measurement utility |
| `TestCLI` | Command-line interface |

> **Note:** ONNX tests are automatically skipped when `onnx` /
> `onnxruntime` are not installed.

---

## Serving the Policy via Docker

The `docker/policy_server/` directory contains a lightweight Flask REST API
that loads an exported policy and serves inference requests.

### Build the image

```bash
docker build -t policy_server docker/policy_server/
```

### Run the container

**ONNX (recommended — lighter runtime):**

```bash
docker run --rm \
    -e POLICY_PATH=/models/actor.onnx \
    -v $(pwd)/exports:/models \
    -p 8080:8080 \
    policy_server
```

**TorchScript:**

```bash
docker run --rm \
    -e POLICY_PATH=/models/actor.torchscript.pt \
    -e POLICY_FORMAT=torchscript \
    -v $(pwd)/exports:/models \
    -p 8080:8080 \
    policy_server
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `POLICY_PATH` | *(required)* | Absolute path to the exported model file |
| `POLICY_FORMAT` | auto-detected | `onnx` or `torchscript` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8080` | Server bind port |

### API reference

#### `GET /health`

Liveness probe — always returns HTTP 200.

```json
{"status": "ok"}
```

#### `GET /info`

Returns metadata about the loaded policy.

```json
{
  "policy_path": "/models/actor.onnx",
  "policy_format": "onnx",
  "input_name": "obs",
  "input_shape": [-1, 22],
  "output_shape": [-1, 3]
}
```

#### `POST /predict`

Run a forward pass.

**Request:**

```json
{
  "obs": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
}
```

`obs` is a 2-D array of shape `(batch, obs_dim)`.  A single observation
can be passed as a 1-D array and will be automatically unsqueezed.

**Response:**

```json
{
  "output": [[0.032, -0.118, 0.551]],
  "latency_ms": 0.2341
}
```

**Example with `curl`:**

```bash
curl -s -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"obs": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}'
```

---

## Benchmarking Inference Latency

Use the `--benchmark` flag to measure CPU inference latency immediately
after export:

```bash
python scripts/export_policy.py \
    --checkpoint checkpoints/actor.pt \
    --model-type mappo_actor \
    --obs-dim 22 \
    --output-dir exports/ \
    --benchmark
```

Sample output:

```
[export] TorchScript → exports/actor.torchscript.pt
[export] ONNX       → exports/actor.onnx

[benchmark] PyTorch CPU inference:
  mean=0.142 ms  min=0.118 ms  max=0.391 ms
[benchmark] ONNX Runtime CPU inference:
  mean=0.085 ms  min=0.071 ms  max=0.192 ms
  ONNX speedup vs PyTorch: 1.67x
[benchmark] TorchScript CPU inference:
  mean=0.131 ms  min=0.109 ms  max=0.303 ms
```

**Target:** < 5 ms per step on CPU (from acceptance criteria).  All
three runtimes comfortably meet this target for the default MAPPOActor
architecture (22-D input → [128, 64] hidden → 3-D output).

You can also call the benchmark helpers programmatically:

```python
from scripts.export_policy import benchmark_inference, benchmark_onnx_inference
import torch

model = ...  # loaded nn.Module in eval mode
stats = benchmark_inference(model, torch.zeros(1, 22), n_warmup=20, n_runs=500)
print(f"mean: {stats['mean_ms']:.3f} ms")

onnx_stats = benchmark_onnx_inference("exports/actor.onnx", torch.zeros(1, 22))
print(f"ONNX mean: {onnx_stats['mean_ms']:.3f} ms")
```

---

## Edge Device & Browser Deployment

### ONNX.js / ONNX Web Runtime

Exported `.onnx` files can be loaded directly in the browser using
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/):

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script>
  async function runPolicy() {
    const session = await ort.InferenceSession.create('actor.onnx');
    const obs = new Float32Array(22).fill(0);
    const tensor = new ort.Tensor('float32', obs, [1, 22]);
    const results = await session.run({ obs: tensor });
    console.log(results['output'].data);  // action mean
  }
  runPolicy();
</script>
```

### C++ / libtorch (TorchScript)

TorchScript `.pt` files can be loaded in C++ without Python:

```cpp
#include <torch/script.h>

auto module = torch::jit::load("actor.torchscript.pt");
module.eval();

torch::Tensor obs = torch::zeros({1, 22});
auto outputs = module.forward({obs}).toTuple();
auto action_mean = outputs->elements()[0].toTensor();
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named 'onnx'` | `onnx` not installed | `pip install onnx` |
| `ImportError: No module named 'onnxruntime'` | `onnxruntime` not installed | `pip install onnxruntime` |
| `RuntimeError: POLICY_PATH environment variable is not set` | Docker container missing env var | Pass `-e POLICY_PATH=/models/actor.onnx` |
| `FileNotFoundError: Policy file not found` | Wrong path or missing volume mount | Check `-v` mount and `POLICY_PATH` |
| ONNX parity > 1e-5 | Floating-point accumulation with large hidden layers | Use `opset_version=17`; ensure model is in `eval()` mode before export |
| TorchScript tracing warning about data-dependent branches | Model uses Python control flow based on tensor values | Refactor to use `torch.where` or scripting instead of tracing |
