# tests/test_entity_encoder.py
"""Tests for models/entity_encoder.py — E8.1 Entity-Based Observation & Transformer Policy.

Coverage
--------
* Entity token schema constants and slice correctness
* SpatialPositionalEncoding — output shapes, no-NaN
* EntityEncoder — forward pass shapes for variable N (1–64 entities)
* EntityEncoder — padding mask correctness
* EntityEncoder — attention weight extraction (return_attention=True)
* EntityEncoder — make_padding_mask helper
* EntityEncoder — inference latency < 8 ms on CPU for 32-entity input
* EntityActorCriticPolicy — act(), get_value(), evaluate_actions()
* EntityActorCriticPolicy — shared vs. separate encoder parameter counts
* EntityActorCriticPolicy — deterministic action reproducibility
* EntityActorCriticPolicy — no-NaN outputs with random inputs
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.entity_encoder import (
    ENTITY_TOKEN_DIM,
    UNIT_TYPE_INFANTRY,
    UNIT_TYPE_CAVALRY,
    UNIT_TYPE_ARTILLERY,
    TEAM_BLUE,
    TEAM_RED,
    SpatialPositionalEncoding,
    EntityEncoder,
    EntityActorCriticPolicy,
)

BATCH = 4
ACTION_DIM = 3
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2

#: Maximum acceptable median CPU inference latency for 32-entity input (ms).
#: Derived from the E8.1 acceptance criteria: < 8 ms on CPU for 32-entity input.
LATENCY_THRESHOLD_MS: float = 8.0


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestEntityTokenSchema(unittest.TestCase):

    def test_token_dim_value(self):
        self.assertEqual(ENTITY_TOKEN_DIM, 16)

    def test_unit_type_indices(self):
        self.assertEqual(UNIT_TYPE_INFANTRY, 0)
        self.assertEqual(UNIT_TYPE_CAVALRY, 1)
        self.assertEqual(UNIT_TYPE_ARTILLERY, 2)

    def test_team_indices(self):
        self.assertEqual(TEAM_BLUE, 0)
        self.assertEqual(TEAM_RED, 1)

    def test_unit_type_one_hot_valid(self):
        """unit_type dims 0-2 should be one-hot (sum to 1)."""
        token = torch.zeros(ENTITY_TOKEN_DIM)
        token[UNIT_TYPE_INFANTRY] = 1.0
        self.assertAlmostEqual(token[:3].sum().item(), 1.0)

    def test_team_one_hot_valid(self):
        """team dims 14-15 should be one-hot (sum to 1)."""
        token = torch.zeros(ENTITY_TOKEN_DIM)
        token[14 + TEAM_BLUE] = 1.0
        self.assertAlmostEqual(token[14:16].sum().item(), 1.0)


# ---------------------------------------------------------------------------
# SpatialPositionalEncoding
# ---------------------------------------------------------------------------


class TestSpatialPositionalEncoding(unittest.TestCase):

    def setUp(self):
        self.pe = SpatialPositionalEncoding(d_model=D_MODEL, n_freqs=4)

    def test_output_shape_2d(self):
        """(B, N, 2) → (B, N, d_model)"""
        xy = torch.rand(BATCH, 10, 2)
        out = self.pe(xy)
        self.assertEqual(out.shape, (BATCH, 10, D_MODEL))

    def test_output_shape_1d(self):
        """(N, 2) → (N, d_model)"""
        xy = torch.rand(10, 2)
        out = self.pe(xy)
        self.assertEqual(out.shape, (10, D_MODEL))

    def test_no_nan(self):
        xy = torch.rand(BATCH, 10, 2)
        out = self.pe(xy)
        self.assertFalse(torch.isnan(out).any().item())

    def test_different_positions_different_encoding(self):
        """Two different positions should (almost certainly) differ."""
        xy1 = torch.tensor([[[0.1, 0.2]]])
        xy2 = torch.tensor([[[0.9, 0.7]]])
        pe1 = self.pe(xy1)
        pe2 = self.pe(xy2)
        self.assertFalse(torch.allclose(pe1, pe2))


# ---------------------------------------------------------------------------
# EntityEncoder — shapes and basic behaviour
# ---------------------------------------------------------------------------


class TestEntityEncoderShapes(unittest.TestCase):

    def setUp(self):
        self.encoder = EntityEncoder(
            token_dim=ENTITY_TOKEN_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            use_spatial_pe=True,
        )

    def _make_tokens(self, batch: int, n: int) -> torch.Tensor:
        tokens = torch.zeros(batch, n, ENTITY_TOKEN_DIM)
        # Set position fields to [0,1] range
        tokens[..., 3:5] = torch.rand(batch, n, 2)
        return tokens

    def test_output_shape(self):
        tokens = self._make_tokens(BATCH, 8)
        out = self.encoder(tokens)
        self.assertEqual(out.shape, (BATCH, D_MODEL))

    def test_output_dim_property(self):
        self.assertEqual(self.encoder.output_dim, D_MODEL)

    def test_variable_n_shapes(self):
        """Encoder must handle any N from 1 to 64."""
        for n in [1, 2, 4, 8, 16, 32, 64]:
            with self.subTest(n=n):
                tokens = self._make_tokens(BATCH, n)
                out = self.encoder(tokens)
                self.assertEqual(out.shape, (BATCH, D_MODEL),
                                 msg=f"Shape mismatch for N={n}")

    def test_no_nan_output(self):
        tokens = torch.randn(BATCH, 12, ENTITY_TOKEN_DIM)
        # Keep positions in [0,1]
        tokens[..., 3:5] = torch.rand(BATCH, 12, 2)
        out = self.encoder(tokens)
        self.assertFalse(torch.isnan(out).any().item())

    def test_no_pad_mask(self):
        """Passing pad_mask=None should work identically to all-False mask."""
        tokens = self._make_tokens(BATCH, 8)
        out_no_mask = self.encoder(tokens, pad_mask=None)
        pad_mask = torch.zeros(BATCH, 8, dtype=torch.bool)
        out_all_valid = self.encoder(tokens, pad_mask=pad_mask)
        self.assertTrue(torch.allclose(out_no_mask, out_all_valid, atol=1e-5))

    def test_single_sample(self):
        """Batch size of 1 should work."""
        tokens = self._make_tokens(1, 4)
        out = self.encoder(tokens)
        self.assertEqual(out.shape, (1, D_MODEL))


# ---------------------------------------------------------------------------
# EntityEncoder — padding mask
# ---------------------------------------------------------------------------


class TestEntityEncoderPaddingMask(unittest.TestCase):

    def setUp(self):
        self.encoder = EntityEncoder(
            token_dim=ENTITY_TOKEN_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            use_spatial_pe=False,
        )
        self.encoder.eval()

    def _make_tokens(self, batch: int, n: int) -> torch.Tensor:
        return torch.randn(batch, n, ENTITY_TOKEN_DIM)

    def test_padded_positions_ignored(self):
        """Changing padded tokens should not change the encoder output."""
        tokens = self._make_tokens(1, 8)
        pad_mask = torch.zeros(1, 8, dtype=torch.bool)
        pad_mask[0, 4:] = True  # last 4 positions are padding

        with torch.no_grad():
            out1 = self.encoder(tokens, pad_mask)
            # Scramble padded positions
            tokens2 = tokens.clone()
            tokens2[0, 4:] = torch.randn(4, ENTITY_TOKEN_DIM) * 100
            out2 = self.encoder(tokens2, pad_mask)

        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-4),
            msg="Padded positions should not affect output",
        )

    def test_make_padding_mask_shape(self):
        n_valid = torch.tensor([4, 8, 2])
        mask = EntityEncoder.make_padding_mask(n_valid, max_n=10)
        self.assertEqual(mask.shape, (3, 10))
        self.assertEqual(mask.dtype, torch.bool)

    def test_make_padding_mask_values(self):
        n_valid = torch.tensor([3])
        mask = EntityEncoder.make_padding_mask(n_valid, max_n=5)
        expected = torch.tensor([[False, False, False, True, True]])
        self.assertTrue(torch.equal(mask, expected))

    def test_all_padded_does_not_crash(self):
        """Edge case: encoder should not crash when only 1 valid entity."""
        tokens = self._make_tokens(2, 6)
        pad_mask = torch.ones(2, 6, dtype=torch.bool)
        pad_mask[:, 0] = False  # only first entity is valid
        with torch.no_grad():
            out = self.encoder(tokens, pad_mask)
        self.assertEqual(out.shape, (2, D_MODEL))
        self.assertFalse(torch.isnan(out).any().item())

    def test_variable_n_with_padding(self):
        """Simulate a batch with different numbers of valid entities."""
        max_n = 16
        tokens = self._make_tokens(BATCH, max_n)
        n_valid = torch.tensor([16, 8, 4, 1])
        pad_mask = EntityEncoder.make_padding_mask(n_valid, max_n)
        with torch.no_grad():
            out = self.encoder(tokens, pad_mask)
        self.assertEqual(out.shape, (BATCH, D_MODEL))
        self.assertFalse(torch.isnan(out).any().item())


# ---------------------------------------------------------------------------
# EntityEncoder — attention weight extraction
# ---------------------------------------------------------------------------


class TestEntityEncoderAttention(unittest.TestCase):

    def setUp(self):
        self.encoder = EntityEncoder(
            token_dim=ENTITY_TOKEN_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            use_spatial_pe=False,
        )
        self.encoder.eval()

    def test_return_attention_shapes(self):
        tokens = torch.randn(BATCH, 8, ENTITY_TOKEN_DIM)
        with torch.no_grad():
            enc, attn = self.encoder(tokens, return_attention=True)
        self.assertEqual(enc.shape, (BATCH, D_MODEL))
        self.assertEqual(attn.shape, (BATCH, 8, 8))

    def test_attention_weights_no_nan(self):
        tokens = torch.randn(BATCH, 8, ENTITY_TOKEN_DIM)
        with torch.no_grad():
            _, attn = self.encoder(tokens, return_attention=True)
        self.assertFalse(torch.isnan(attn).any().item())

    def test_attention_weights_sum_to_one_per_row(self):
        """Each attention row should sum to ~1 (softmax)."""
        tokens = torch.randn(2, 6, ENTITY_TOKEN_DIM)
        with torch.no_grad():
            _, attn = self.encoder(tokens, return_attention=True)
        row_sums = attn.sum(dim=-1)  # (B, N)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4),
                        msg="Attention rows must sum to 1")

    def test_return_encoding_consistent(self):
        """Encoding returned with return_attention should match without."""
        tokens = torch.randn(BATCH, 8, ENTITY_TOKEN_DIM)
        self.encoder.eval()
        with torch.no_grad():
            enc_no_attn = self.encoder(tokens, return_attention=False)
            enc_with_attn, _ = self.encoder(tokens, return_attention=True)
        self.assertTrue(torch.allclose(enc_no_attn, enc_with_attn, atol=1e-5))


# ---------------------------------------------------------------------------
# EntityEncoder — inference latency
# ---------------------------------------------------------------------------


class TestEntityEncoderLatency(unittest.TestCase):

    @unittest.skipUnless(
        os.environ.get("RUN_LATENCY_TESTS"),
        "Skipped by default to avoid CI flakiness. Set RUN_LATENCY_TESTS to any non-empty value to enable.",
    )
    def test_latency_32_entities_under_8ms(self):
        """Inference for 32-entity input should complete in < 8 ms on CPU."""
        encoder = EntityEncoder(
            token_dim=ENTITY_TOKEN_DIM,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            use_spatial_pe=True,
        )
        encoder.eval()

        tokens = torch.zeros(1, 32, ENTITY_TOKEN_DIM)
        tokens[..., 3:5] = torch.rand(1, 32, 2)  # valid position range

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                encoder(tokens)

        # Measure median over 20 runs
        latencies = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                encoder(tokens)
                latencies.append((time.perf_counter() - t0) * 1e3)

        median_ms = float(np.median(latencies))
        self.assertLess(
            median_ms, LATENCY_THRESHOLD_MS,
            msg=f"Median CPU latency {median_ms:.2f} ms exceeds {LATENCY_THRESHOLD_MS} ms limit",
        )


# ---------------------------------------------------------------------------
# EntityActorCriticPolicy — basic usage
# ---------------------------------------------------------------------------


class TestEntityActorCriticPolicyBasic(unittest.TestCase):

    def setUp(self):
        self.policy = EntityActorCriticPolicy(
            token_dim=ENTITY_TOKEN_DIM,
            action_dim=ACTION_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            actor_hidden_sizes=(32, 16),
            critic_hidden_sizes=(32, 16),
            shared_encoder=True,
            dropout=0.0,
            use_spatial_pe=True,
        )

    def _make_tokens(self, batch: int, n: int) -> torch.Tensor:
        tokens = torch.zeros(batch, n, ENTITY_TOKEN_DIM)
        tokens[..., 3:5] = torch.rand(batch, n, 2)
        return tokens

    def test_act_shapes(self):
        tokens = self._make_tokens(BATCH, 8)
        actions, log_probs = self.policy.act(tokens)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))
        self.assertEqual(log_probs.shape, (BATCH,))

    def test_act_single_sample(self):
        """Single sample (no batch dim) should return (1, action_dim), (1,)."""
        tokens = self._make_tokens(1, 8).squeeze(0)  # (N, token_dim)
        actions, log_probs = self.policy.act(tokens)
        self.assertEqual(actions.shape, (1, ACTION_DIM))
        self.assertEqual(log_probs.shape, (1,))

    def test_get_value_shape(self):
        tokens = self._make_tokens(BATCH, 8)
        values = self.policy.get_value(tokens)
        self.assertEqual(values.shape, (BATCH,))

    def test_evaluate_actions_shapes(self):
        tokens = self._make_tokens(BATCH, 8)
        actions = torch.zeros(BATCH, ACTION_DIM)
        log_probs, entropy = self.policy.evaluate_actions(tokens, actions)
        self.assertEqual(log_probs.shape, (BATCH,))
        self.assertEqual(entropy.shape, (BATCH,))

    def test_deterministic_reproducible(self):
        tokens = self._make_tokens(BATCH, 8)
        a1, _ = self.policy.act(tokens, deterministic=True)
        a2, _ = self.policy.act(tokens, deterministic=True)
        self.assertTrue(torch.allclose(a1, a2))

    def test_no_nan_act(self):
        tokens = torch.randn(BATCH, 8, ENTITY_TOKEN_DIM)
        tokens[..., 3:5] = torch.rand(BATCH, 8, 2)
        actions, log_probs = self.policy.act(tokens)
        self.assertFalse(torch.isnan(actions).any().item())
        self.assertFalse(torch.isnan(log_probs).any().item())

    def test_no_nan_value(self):
        tokens = torch.randn(BATCH, 8, ENTITY_TOKEN_DIM)
        tokens[..., 3:5] = torch.rand(BATCH, 8, 2)
        values = self.policy.get_value(tokens)
        self.assertFalse(torch.isnan(values).any().item())

    def test_variable_n_act(self):
        """Policy should handle variable sequence lengths."""
        for n in [1, 4, 16, 32, 64]:
            with self.subTest(n=n):
                tokens = self._make_tokens(BATCH, n)
                actions, log_probs = self.policy.act(tokens)
                self.assertEqual(actions.shape, (BATCH, ACTION_DIM))

    def test_with_padding_mask(self):
        tokens = self._make_tokens(BATCH, 16)
        n_valid = torch.tensor([16, 8, 4, 2])
        pad_mask = EntityEncoder.make_padding_mask(n_valid, max_n=16)
        actions, log_probs = self.policy.act(tokens, pad_mask)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))
        self.assertFalse(torch.isnan(actions).any().item())


# ---------------------------------------------------------------------------
# EntityActorCriticPolicy — shared vs. separate encoder
# ---------------------------------------------------------------------------


class TestEntityActorCriticPolicySharedEncoder(unittest.TestCase):

    def _make_policy(self, shared: bool) -> EntityActorCriticPolicy:
        return EntityActorCriticPolicy(
            token_dim=ENTITY_TOKEN_DIM,
            action_dim=ACTION_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
            shared_encoder=shared,
            dropout=0.0,
        )

    def test_shared_encoder_is_same_object(self):
        policy = self._make_policy(shared=True)
        self.assertIs(policy.actor_encoder, policy.critic_encoder)

    def test_separate_encoder_different_objects(self):
        policy = self._make_policy(shared=False)
        self.assertIsNot(policy.actor_encoder, policy.critic_encoder)

    def test_parameter_count_dict_keys(self):
        policy = self._make_policy(shared=True)
        counts = policy.parameter_count()
        self.assertIn("actor", counts)
        self.assertIn("critic", counts)
        self.assertIn("total", counts)
        self.assertGreater(counts["total"], 0)

    def test_separate_more_params_than_shared(self):
        shared = self._make_policy(shared=True)
        separate = self._make_policy(shared=False)
        # Separate encoder adds an extra encoder worth of params to critic
        self.assertGreater(
            separate.parameter_count()["total"],
            shared.parameter_count()["total"],
        )


# ---------------------------------------------------------------------------
# EntityActorCriticPolicy — gradient flow
# ---------------------------------------------------------------------------


class TestEntityActorCriticPolicyGradients(unittest.TestCase):

    def setUp(self):
        self.policy = EntityActorCriticPolicy(
            token_dim=ENTITY_TOKEN_DIM,
            action_dim=ACTION_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            actor_hidden_sizes=(32, 16),
            critic_hidden_sizes=(32, 16),
            dropout=0.0,
        )

    def _make_tokens(self, batch: int, n: int) -> torch.Tensor:
        tokens = torch.zeros(batch, n, ENTITY_TOKEN_DIM)
        tokens[..., 3:5] = torch.rand(batch, n, 2)
        return tokens

    def test_policy_loss_backward(self):
        tokens = self._make_tokens(BATCH, 8)
        actions_ref = torch.zeros(BATCH, ACTION_DIM)
        log_probs, entropy = self.policy.evaluate_actions(tokens, actions_ref)
        loss = -log_probs.mean() - 0.01 * entropy.mean()
        loss.backward()
        # Check that some gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in self.policy.parameters()
        )
        self.assertTrue(has_grad, "Backward pass should produce non-zero gradients")

    def test_value_loss_backward(self):
        tokens = self._make_tokens(BATCH, 8)
        values = self.policy.get_value(tokens)
        targets = torch.ones(BATCH)
        loss = (values - targets).pow(2).mean()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in self.policy.critic_head.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
