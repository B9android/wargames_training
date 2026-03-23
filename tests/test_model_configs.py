# tests/test_model_configs.py
"""Tests for E8.3 model configuration files and latency acceptance criteria.

Coverage
--------
* YAML structure validation for all three tier configs (small / medium / large)
* Required keys present and correctly typed
* Transformer constraint: n_heads evenly divides d_model
* dim_feedforward matches 4 × d_model
* Latency acceptance criteria (small < 5 ms CPU; large < 20 ms CPU)
  measured via a real EntityEncoder forward pass on 32-entity input (batch=1)
* Sweep config (model_scaling_sweep.yaml) covers ≥ 18 combinations
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_CONFIGS_DIR = PROJECT_ROOT / "configs" / "models"
_SWEEPS_DIR = PROJECT_ROOT / "configs" / "sweeps"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _encoder_cfg(data: dict) -> dict:
    return data["model"]["encoder"]


# ---------------------------------------------------------------------------
# YAML structure tests
# ---------------------------------------------------------------------------


class TestTransformerSmallConfig(unittest.TestCase):

    def setUp(self):
        self.data = _load_yaml(_CONFIGS_DIR / "transformer_small.yaml")

    def test_top_level_model_key(self):
        self.assertIn("model", self.data)

    def test_name(self):
        self.assertEqual(self.data["model"]["name"], "transformer_small")

    def test_encoder_section_present(self):
        self.assertIn("encoder", self.data["model"])

    def test_required_encoder_keys(self):
        enc = _encoder_cfg(self.data)
        for key in ("d_model", "n_heads", "n_layers", "dim_feedforward",
                    "dropout", "use_spatial_pe", "n_freq_bands"):
            with self.subTest(key=key):
                self.assertIn(key, enc)

    def test_d_model_is_int(self):
        self.assertIsInstance(_encoder_cfg(self.data)["d_model"], int)

    def test_n_heads_divides_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["d_model"] % enc["n_heads"], 0,
                         "n_heads must evenly divide d_model")

    def test_dim_feedforward_equals_4x_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["dim_feedforward"], 4 * enc["d_model"])

    def test_latency_target_present(self):
        self.assertIn("latency_target_ms", self.data["model"])

    def test_latency_target_small_lte_5ms(self):
        self.assertLessEqual(self.data["model"]["latency_target_ms"], 5)

    def test_actor_hidden_sizes_present(self):
        self.assertIn("actor_hidden_sizes", self.data["model"])

    def test_critic_hidden_sizes_present(self):
        self.assertIn("critic_hidden_sizes", self.data["model"])

    def test_shared_encoder_key(self):
        self.assertIn("shared_encoder", self.data["model"])

    def test_n_layers_value(self):
        self.assertEqual(_encoder_cfg(self.data)["n_layers"], 2)

    def test_d_model_value(self):
        self.assertEqual(_encoder_cfg(self.data)["d_model"], 64)


class TestTransformerMediumConfig(unittest.TestCase):

    def setUp(self):
        self.data = _load_yaml(_CONFIGS_DIR / "transformer_medium.yaml")

    def test_name(self):
        self.assertEqual(self.data["model"]["name"], "transformer_medium")

    def test_n_heads_divides_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["d_model"] % enc["n_heads"], 0)

    def test_dim_feedforward_equals_4x_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["dim_feedforward"], 4 * enc["d_model"])

    def test_n_layers_value(self):
        self.assertEqual(_encoder_cfg(self.data)["n_layers"], 4)

    def test_d_model_value(self):
        self.assertEqual(_encoder_cfg(self.data)["d_model"], 256)

    def test_latency_target_present(self):
        self.assertIn("latency_target_ms", self.data["model"])

    def test_latency_target_strictly_between_small_and_large(self):
        target = self.data["model"]["latency_target_ms"]
        self.assertGreater(target, 5)
        self.assertLess(target, 20)


class TestTransformerLargeConfig(unittest.TestCase):

    def setUp(self):
        self.data = _load_yaml(_CONFIGS_DIR / "transformer_large.yaml")

    def test_name(self):
        self.assertEqual(self.data["model"]["name"], "transformer_large")

    def test_n_heads_divides_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["d_model"] % enc["n_heads"], 0)

    def test_dim_feedforward_equals_4x_d_model(self):
        enc = _encoder_cfg(self.data)
        self.assertEqual(enc["dim_feedforward"], 4 * enc["d_model"])

    def test_n_layers_value(self):
        self.assertEqual(_encoder_cfg(self.data)["n_layers"], 8)

    def test_d_model_value(self):
        self.assertEqual(_encoder_cfg(self.data)["d_model"], 512)

    def test_latency_target_present(self):
        self.assertIn("latency_target_ms", self.data["model"])

    def test_latency_target_large_lte_20ms(self):
        self.assertLessEqual(self.data["model"]["latency_target_ms"], 20)


# ---------------------------------------------------------------------------
# Cross-config ordering constraints
# ---------------------------------------------------------------------------


class TestConfigOrdering(unittest.TestCase):
    """Small ⊂ Medium ⊂ Large in depth, width, and latency budget."""

    def setUp(self):
        self.small = _load_yaml(_CONFIGS_DIR / "transformer_small.yaml")
        self.medium = _load_yaml(_CONFIGS_DIR / "transformer_medium.yaml")
        self.large = _load_yaml(_CONFIGS_DIR / "transformer_large.yaml")

    def _enc(self, data):
        return _encoder_cfg(data)

    def test_d_model_ordering(self):
        self.assertLess(self._enc(self.small)["d_model"],
                        self._enc(self.medium)["d_model"])
        self.assertLess(self._enc(self.medium)["d_model"],
                        self._enc(self.large)["d_model"])

    def test_n_layers_ordering(self):
        self.assertLess(self._enc(self.small)["n_layers"],
                        self._enc(self.medium)["n_layers"])
        self.assertLess(self._enc(self.medium)["n_layers"],
                        self._enc(self.large)["n_layers"])

    def test_latency_target_ordering(self):
        self.assertLess(self.small["model"]["latency_target_ms"],
                        self.medium["model"]["latency_target_ms"])
        self.assertLess(self.medium["model"]["latency_target_ms"],
                        self.large["model"]["latency_target_ms"])


# ---------------------------------------------------------------------------
# Sweep config coverage
# ---------------------------------------------------------------------------


class TestModelScalingSweepConfig(unittest.TestCase):

    def setUp(self):
        self.data = _load_yaml(_SWEEPS_DIR / "model_scaling_sweep.yaml")

    def test_top_level_keys(self):
        for key in ("program", "method", "project", "metric", "parameters"):
            with self.subTest(key=key):
                self.assertIn(key, self.data)

    def test_metric_name(self):
        self.assertIn("name", self.data["metric"])

    def test_required_parameters_present(self):
        params = self.data["parameters"]
        for key in ("model.encoder.n_layers", "model.encoder.d_model",
                    "model.encoder.n_heads"):
            with self.subTest(key=key):
                self.assertIn(key, params)

    def test_at_least_18_configurations(self):
        """Grid product must cover ≥ 18 configurations (E8.3 acceptance)."""
        params = self.data["parameters"]
        n_layers = params["model.encoder.n_layers"]["values"]
        d_model = params["model.encoder.d_model"]["values"]
        n_heads = params["model.encoder.n_heads"]["values"]
        n_configs = len(n_layers) * len(d_model) * len(n_heads)
        self.assertGreaterEqual(n_configs, 18,
                                f"Sweep has only {n_configs} configurations; need ≥ 18")

    def test_all_n_heads_divide_all_d_models(self):
        """Every (d_model, n_heads) pair in the sweep must be valid."""
        params = self.data["parameters"]
        for d in params["model.encoder.d_model"]["values"]:
            for h in params["model.encoder.n_heads"]["values"]:
                with self.subTest(d_model=d, n_heads=h):
                    self.assertEqual(d % h, 0,
                                     f"d_model={d} not divisible by n_heads={h}")


# ---------------------------------------------------------------------------
# Latency acceptance criteria (requires torch)
# ---------------------------------------------------------------------------


try:
    import torch
    from models.entity_encoder import EntityEncoder

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

#: Number of warm-up forward passes before timing.
_WARMUP = 5
#: Number of timed forward passes; median is used.
_N_TIMED = 20
#: Entity count matching the E8.3 acceptance criterion definition.
_N_ENTITIES = 32


def _median_latency_ms(enc: "EntityEncoder") -> float:
    """Return median CPU forward-pass latency in ms for a single sample."""
    enc.eval()
    tokens = torch.zeros(1, _N_ENTITIES, enc.token_embed.in_features)
    with torch.no_grad():
        for _ in range(_WARMUP):
            enc(tokens)
        times = []
        for _ in range(_N_TIMED):
            t0 = time.perf_counter()
            enc(tokens)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mid = _N_TIMED // 2
    # True median: average the two middle elements for even-length lists.
    return (times[mid - 1] + times[mid]) / 2.0


_RUN_LATENCY = os.environ.get("RUN_LATENCY_TESTS")
_LATENCY_SKIP_MSG = (
    "Skipped by default to avoid CI flakiness. "
    "Set RUN_LATENCY_TESTS to any non-empty value to enable."
)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
@unittest.skipUnless(_RUN_LATENCY, _LATENCY_SKIP_MSG)
class TestSmallModelLatency(unittest.TestCase):
    """Small model CPU inference must be < 5 ms (E8.3 acceptance criterion)."""

    def test_small_latency_under_5ms(self):
        data = _load_yaml(_CONFIGS_DIR / "transformer_small.yaml")
        enc_cfg = _encoder_cfg(data)
        enc = EntityEncoder(
            d_model=enc_cfg["d_model"],
            n_heads=enc_cfg["n_heads"],
            n_layers=enc_cfg["n_layers"],
            dim_feedforward=enc_cfg["dim_feedforward"],
            dropout=enc_cfg["dropout"],
            use_spatial_pe=enc_cfg["use_spatial_pe"],
            n_freq_bands=enc_cfg["n_freq_bands"],
        )
        latency = _median_latency_ms(enc)
        self.assertLessEqual(
            latency, 5.0,
            f"Small model latency {latency:.2f} ms exceeds 5 ms target",
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
@unittest.skipUnless(_RUN_LATENCY, _LATENCY_SKIP_MSG)
class TestLargeModelLatency(unittest.TestCase):
    """Large model CPU inference must be < 20 ms (E8.3 acceptance criterion)."""

    def test_large_latency_under_20ms(self):
        data = _load_yaml(_CONFIGS_DIR / "transformer_large.yaml")
        enc_cfg = _encoder_cfg(data)
        enc = EntityEncoder(
            d_model=enc_cfg["d_model"],
            n_heads=enc_cfg["n_heads"],
            n_layers=enc_cfg["n_layers"],
            dim_feedforward=enc_cfg["dim_feedforward"],
            dropout=enc_cfg["dropout"],
            use_spatial_pe=enc_cfg["use_spatial_pe"],
            n_freq_bands=enc_cfg["n_freq_bands"],
        )
        latency = _median_latency_ms(enc)
        self.assertLessEqual(
            latency, 20.0,
            f"Large model latency {latency:.2f} ms exceeds 20 ms target",
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
@unittest.skipUnless(_RUN_LATENCY, _LATENCY_SKIP_MSG)
class TestMediumModelLatency(unittest.TestCase):
    """Medium model CPU inference must be within [5 ms, 20 ms]."""

    def test_medium_latency_bounds(self):
        data = _load_yaml(_CONFIGS_DIR / "transformer_medium.yaml")
        enc_cfg = _encoder_cfg(data)
        enc = EntityEncoder(
            d_model=enc_cfg["d_model"],
            n_heads=enc_cfg["n_heads"],
            n_layers=enc_cfg["n_layers"],
            dim_feedforward=enc_cfg["dim_feedforward"],
            dropout=enc_cfg["dropout"],
            use_spatial_pe=enc_cfg["use_spatial_pe"],
            n_freq_bands=enc_cfg["n_freq_bands"],
        )
        latency = _median_latency_ms(enc)
        self.assertGreater(
            latency, 5.0,
            f"Medium model latency {latency:.2f} ms is unexpectedly fast (≤ small tier bound of 5 ms)",
        )
        self.assertLessEqual(
            latency, 20.0,
            f"Medium model latency {latency:.2f} ms exceeds 20 ms upper bound",
        )


if __name__ == "__main__":
    unittest.main()
