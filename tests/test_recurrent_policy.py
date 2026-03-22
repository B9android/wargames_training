# tests/test_recurrent_policy.py
"""Tests for models/recurrent_policy.py — E8.2 Memory Module (LSTM / Temporal Context).

Coverage
--------
* LSTMHiddenState — zeros(), detach(), to(), reset_at(), as_tuple()
* RecurrentEntityEncoder — output shapes, LSTM state threading, sequence forward
* RecurrentActorCriticPolicy — act(), get_value(), evaluate_actions()
* RecurrentActorCriticPolicy — initial_state(), episode-boundary reset
* RecurrentActorCriticPolicy — shared vs. separate encoder parameter counts
* RecurrentActorCriticPolicy — deterministic action reproducibility
* RecurrentActorCriticPolicy — no-NaN outputs with random inputs
* RecurrentActorCriticPolicy — save_checkpoint() / load_checkpoint()
* RecurrentRolloutBuffer — add(), len, full/reset guard
* RecurrentRolloutBuffer — compute_returns_and_advantages() (GAE)
* RecurrentRolloutBuffer — get_sequences() shapes and normalised advantages
* RecurrentRolloutBuffer — memory_bytes() overhead < 20 % vs. non-recurrent baseline
* RecurrentRolloutBuffer — hidden state correctly zeroed at episode boundaries
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.entity_encoder import ENTITY_TOKEN_DIM
from models.recurrent_policy import (
    LSTMHiddenState,
    RecurrentActorCriticPolicy,
    RecurrentEntityEncoder,
    RecurrentRolloutBuffer,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

BATCH = 4
N_ENTITIES = 8
TOKEN_DIM = ENTITY_TOKEN_DIM  # 16
ACTION_DIM = 3
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
LSTM_HIDDEN = 64
LSTM_LAYERS = 2


def _make_tokens(batch: int = BATCH, n: int = N_ENTITIES) -> torch.Tensor:
    """Random entity token batch of shape (batch, n, TOKEN_DIM)."""
    return torch.randn(batch, n, TOKEN_DIM)


def _make_policy(**kw) -> RecurrentActorCriticPolicy:
    defaults = dict(
        token_dim=TOKEN_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        lstm_hidden_size=LSTM_HIDDEN,
        lstm_num_layers=LSTM_LAYERS,
    )
    defaults.update(kw)
    return RecurrentActorCriticPolicy(**defaults)


# ---------------------------------------------------------------------------
# LSTMHiddenState
# ---------------------------------------------------------------------------


class TestLSTMHiddenState(unittest.TestCase):

    def test_zeros_shape(self):
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, BATCH)
        self.assertEqual(hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))
        self.assertEqual(hx.c.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_zeros_are_zero(self):
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, BATCH)
        self.assertTrue(hx.h.eq(0).all())
        self.assertTrue(hx.c.eq(0).all())

    def test_detach_no_grad(self):
        # Create hx with grad
        h = torch.randn(LSTM_LAYERS, BATCH, LSTM_HIDDEN, requires_grad=True)
        c = torch.randn(LSTM_LAYERS, BATCH, LSTM_HIDDEN, requires_grad=True)
        hx = LSTMHiddenState(h=h, c=c)
        hx_d = hx.detach()
        self.assertFalse(hx_d.h.requires_grad)
        self.assertFalse(hx_d.c.requires_grad)

    def test_to_device(self):
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, BATCH)
        hx2 = hx.to(torch.device("cpu"))
        self.assertEqual(hx2.h.device.type, "cpu")

    def test_reset_at_zeroes_done_entries(self):
        hx = LSTMHiddenState(
            h=torch.ones(LSTM_LAYERS, BATCH, LSTM_HIDDEN),
            c=torch.ones(LSTM_LAYERS, BATCH, LSTM_HIDDEN),
        )
        done = torch.tensor([True, False, True, False])
        hx_new = hx.reset_at(done)

        # Indices 0 and 2 should be zeros
        for b in [0, 2]:
            self.assertTrue(hx_new.h[:, b, :].eq(0).all(), f"h not zeroed at b={b}")
            self.assertTrue(hx_new.c[:, b, :].eq(0).all(), f"c not zeroed at b={b}")
        # Indices 1 and 3 should remain ones
        for b in [1, 3]:
            self.assertTrue(hx_new.h[:, b, :].eq(1).all(), f"h wrongly zeroed at b={b}")

    def test_reset_at_preserves_original(self):
        """reset_at must not mutate the original state."""
        hx = LSTMHiddenState(
            h=torch.ones(LSTM_LAYERS, BATCH, LSTM_HIDDEN),
            c=torch.ones(LSTM_LAYERS, BATCH, LSTM_HIDDEN),
        )
        done = torch.ones(BATCH, dtype=torch.bool)
        _ = hx.reset_at(done)
        self.assertTrue(hx.h.eq(1).all())

    def test_as_tuple(self):
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, BATCH)
        t = hx.as_tuple()
        self.assertIsInstance(t, tuple)
        self.assertEqual(len(t), 2)
        self.assertIs(t[0], hx.h)
        self.assertIs(t[1], hx.c)


# ---------------------------------------------------------------------------
# RecurrentEntityEncoder
# ---------------------------------------------------------------------------


class TestRecurrentEntityEncoder(unittest.TestCase):

    def setUp(self):
        self.enc = RecurrentEntityEncoder(
            token_dim=TOKEN_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lstm_hidden_size=LSTM_HIDDEN,
            lstm_num_layers=LSTM_LAYERS,
        )

    def test_output_dim_property(self):
        self.assertEqual(self.enc.output_dim, LSTM_HIDDEN)

    def test_forward_single_step_shape(self):
        tokens = _make_tokens(BATCH, N_ENTITIES)
        hx = self.enc.initial_state(BATCH)
        out, new_hx = self.enc(tokens, hx)
        self.assertEqual(out.shape, (BATCH, LSTM_HIDDEN))
        self.assertEqual(new_hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))
        self.assertEqual(new_hx.c.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_forward_no_nan(self):
        tokens = _make_tokens(BATCH, N_ENTITIES)
        hx = self.enc.initial_state(BATCH)
        out, _ = self.enc(tokens, hx)
        self.assertFalse(torch.isnan(out).any())

    def test_state_changes_after_forward(self):
        tokens = _make_tokens(BATCH, N_ENTITIES)
        hx = self.enc.initial_state(BATCH)
        _, new_hx = self.enc(tokens, hx)
        # State should generally not remain zero after a forward pass
        self.assertFalse(new_hx.h.eq(0).all())

    def test_forward_sequence_shape(self):
        T = 5
        tokens_seq = torch.randn(BATCH, T, N_ENTITIES, TOKEN_DIM)
        hx = self.enc.initial_state(BATCH)
        out_seq, new_hx = self.enc.forward_sequence(tokens_seq, hx)
        self.assertEqual(out_seq.shape, (BATCH, T, LSTM_HIDDEN))
        self.assertEqual(new_hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_sequence_consistent_with_step_by_step(self):
        """forward_sequence output must match step-by-step forward calls."""
        torch.manual_seed(0)
        B, T = 2, 4
        tokens_seq = torch.randn(B, T, N_ENTITIES, TOKEN_DIM)
        hx0 = self.enc.initial_state(B)

        # Step-by-step
        hx = hx0
        step_outs = []
        with torch.no_grad():
            for t in range(T):
                out, hx = self.enc(tokens_seq[:, t], hx)
                step_outs.append(out)
        step_out = torch.stack(step_outs, dim=1)  # (B, T, hidden)

        # Sequence forward
        with torch.no_grad():
            seq_out, _ = self.enc.forward_sequence(tokens_seq, hx0)

        self.assertTrue(torch.allclose(step_out, seq_out, atol=1e-5))

    def test_with_padding_mask(self):
        tokens = _make_tokens(BATCH, N_ENTITIES)
        pad_mask = torch.zeros(BATCH, N_ENTITIES, dtype=torch.bool)
        pad_mask[:, 4:] = True  # pad the last half
        hx = self.enc.initial_state(BATCH)
        out, _ = self.enc(tokens, hx, pad_mask=pad_mask)
        self.assertEqual(out.shape, (BATCH, LSTM_HIDDEN))
        self.assertFalse(torch.isnan(out).any())

    def test_initial_state_device(self):
        hx = self.enc.initial_state(BATCH, device=torch.device("cpu"))
        self.assertEqual(hx.h.device.type, "cpu")


# ---------------------------------------------------------------------------
# RecurrentActorCriticPolicy
# ---------------------------------------------------------------------------


class TestRecurrentActorCriticPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = _make_policy()

    def test_initial_state_shape(self):
        hx = self.policy.initial_state(BATCH)
        self.assertEqual(hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_act_shapes(self):
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        actions, log_probs, new_hx = self.policy.act(tokens, hx)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))
        self.assertEqual(log_probs.shape, (BATCH,))
        self.assertEqual(new_hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_act_single_sample(self):
        """Single (N, token_dim) input should be handled (batch dim added)."""
        tokens = torch.randn(N_ENTITIES, TOKEN_DIM)
        hx = self.policy.initial_state(1)
        actions, log_probs, new_hx = self.policy.act(tokens, hx)
        self.assertEqual(actions.shape, (1, ACTION_DIM))
        self.assertEqual(log_probs.shape, (1,))

    def test_act_deterministic_reproducible(self):
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        a1, _, _ = self.policy.act(tokens, hx, deterministic=True)
        a2, _, _ = self.policy.act(tokens, hx, deterministic=True)
        self.assertTrue(torch.allclose(a1, a2))

    def test_act_stochastic_differs(self):
        torch.manual_seed(0)
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        a1, _, _ = self.policy.act(tokens, hx, deterministic=False)
        torch.manual_seed(99)
        a2, _, _ = self.policy.act(tokens, hx, deterministic=False)
        self.assertFalse(torch.allclose(a1, a2))

    def test_act_no_nan(self):
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        actions, log_probs, _ = self.policy.act(tokens, hx)
        self.assertFalse(torch.isnan(actions).any())
        self.assertFalse(torch.isnan(log_probs).any())

    def test_get_value_shapes(self):
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        values, new_hx = self.policy.get_value(tokens, hx)
        self.assertEqual(values.shape, (BATCH,))
        self.assertEqual(new_hx.h.shape, (LSTM_LAYERS, BATCH, LSTM_HIDDEN))

    def test_get_value_single_sample(self):
        tokens = torch.randn(N_ENTITIES, TOKEN_DIM)
        hx = self.policy.initial_state(1)
        values, _ = self.policy.get_value(tokens, hx)
        self.assertEqual(values.shape, (1,))

    def test_get_value_no_nan(self):
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        values, _ = self.policy.get_value(tokens, hx)
        self.assertFalse(torch.isnan(values).any())

    def test_evaluate_actions_shapes(self):
        B, T = 2, 8
        tokens_seq = torch.randn(B, T, N_ENTITIES, TOKEN_DIM)
        hx = self.policy.initial_state(B)
        actions_seq = torch.randn(B, T, ACTION_DIM)
        log_probs, entropy, values = self.policy.evaluate_actions(
            tokens_seq, hx, actions_seq
        )
        self.assertEqual(log_probs.shape, (B, T))
        self.assertEqual(entropy.shape, (B, T))
        self.assertEqual(values.shape, (B, T))

    def test_evaluate_actions_no_nan(self):
        B, T = 2, 4
        tokens_seq = torch.randn(B, T, N_ENTITIES, TOKEN_DIM)
        hx = self.policy.initial_state(B)
        actions_seq = torch.randn(B, T, ACTION_DIM)
        log_probs, entropy, values = self.policy.evaluate_actions(
            tokens_seq, hx, actions_seq
        )
        self.assertFalse(torch.isnan(log_probs).any())
        self.assertFalse(torch.isnan(entropy).any())
        self.assertFalse(torch.isnan(values).any())

    def test_evaluate_actions_entropy_positive(self):
        B, T = 2, 4
        tokens_seq = torch.randn(B, T, N_ENTITIES, TOKEN_DIM)
        hx = self.policy.initial_state(B)
        actions_seq = torch.randn(B, T, ACTION_DIM)
        _, entropy, _ = self.policy.evaluate_actions(tokens_seq, hx, actions_seq)
        self.assertTrue((entropy > 0).all())

    def test_shared_vs_separate_encoder_param_counts(self):
        shared = _make_policy(shared_encoder=True)
        separate = _make_policy(shared_encoder=False)
        pc_shared = shared.parameter_count()
        pc_separate = separate.parameter_count()
        self.assertLess(pc_shared["total"], pc_separate["total"])

    def test_parameter_count_keys(self):
        pc = self.policy.parameter_count()
        self.assertIn("actor", pc)
        self.assertIn("critic", pc)
        self.assertIn("total", pc)

    def test_hidden_state_reset_at_episode_boundary(self):
        """After reset_at, hidden state should be zero for done episodes."""
        tokens = _make_tokens(BATCH)
        hx = self.policy.initial_state(BATCH)
        # Run a forward pass to get non-zero state
        _, _, hx = self.policy.act(tokens, hx)
        self.assertFalse(hx.h.eq(0).all(), "State should be non-zero after forward")

        # Reset all episodes
        done = torch.ones(BATCH, dtype=torch.bool)
        hx_reset = hx.reset_at(done)
        self.assertTrue(hx_reset.h.eq(0).all())
        self.assertTrue(hx_reset.c.eq(0).all())

    def test_hidden_state_threads_across_steps(self):
        """Hidden state fed back in should influence subsequent outputs."""
        tokens = _make_tokens(1)
        hx = self.policy.initial_state(1)
        _, _, hx1 = self.policy.act(tokens, hx, deterministic=True)
        a_step2_fresh, _, _ = self.policy.act(tokens, hx, deterministic=True)
        a_step2_carried, _, _ = self.policy.act(tokens, hx1, deterministic=True)
        # Different hidden states → different outputs
        self.assertFalse(torch.allclose(a_step2_fresh, a_step2_carried))


# ---------------------------------------------------------------------------
# RecurrentActorCriticPolicy — checkpointing
# ---------------------------------------------------------------------------


class TestRecurrentPolicyCheckpointing(unittest.TestCase):

    def test_save_and_load_weights_match(self):
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "policy.pt"
            policy.save_checkpoint(ckpt_path)

            loaded = RecurrentActorCriticPolicy.load_checkpoint(
                ckpt_path,
                token_dim=TOKEN_DIM,
                action_dim=ACTION_DIM,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                lstm_hidden_size=LSTM_HIDDEN,
                lstm_num_layers=LSTM_LAYERS,
            )

        # Weights should be identical
        for (n1, p1), (n2, p2) in zip(
            policy.named_parameters(), loaded.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(
                torch.allclose(p1, p2), f"Parameter mismatch: {n1}"
            )

    def test_save_creates_parent_dirs(self):
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "deep" / "nested" / "policy.pt"
            policy.save_checkpoint(nested)
            self.assertTrue(nested.exists())

    def test_loaded_policy_produces_same_actions(self):
        policy = _make_policy()
        tokens = _make_tokens(1)
        hx = policy.initial_state(1)
        actions_orig, _, _ = policy.act(tokens, hx, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            policy.save_checkpoint(ckpt_path)
            loaded = RecurrentActorCriticPolicy.load_checkpoint(
                ckpt_path,
                token_dim=TOKEN_DIM,
                action_dim=ACTION_DIM,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                lstm_hidden_size=LSTM_HIDDEN,
                lstm_num_layers=LSTM_LAYERS,
            )

        hx2 = loaded.initial_state(1)
        actions_loaded, _, _ = loaded.act(tokens, hx2, deterministic=True)
        self.assertTrue(torch.allclose(actions_orig, actions_loaded))


# ---------------------------------------------------------------------------
# RecurrentRolloutBuffer
# ---------------------------------------------------------------------------


def _make_buffer(n_steps: int = 32, **kw) -> RecurrentRolloutBuffer:
    defaults = dict(
        n_steps=n_steps,
        max_entities=N_ENTITIES,
        token_dim=TOKEN_DIM,
        action_dim=ACTION_DIM,
        lstm_hidden_size=LSTM_HIDDEN,
        lstm_num_layers=LSTM_LAYERS,
    )
    defaults.update(kw)
    return RecurrentRolloutBuffer(**defaults)


def _fill_buffer(
    buf: RecurrentRolloutBuffer,
    lstm_layers: int = LSTM_LAYERS,
    lstm_hidden: int = LSTM_HIDDEN,
    episode_len: int | None = None,
) -> None:
    """Fill *buf* with dummy data.  Optionally simulate episode resets."""
    for i in range(buf.n_steps):
        tokens = np.random.randn(N_ENTITIES, TOKEN_DIM).astype(np.float32)
        hx = LSTMHiddenState.zeros(lstm_layers, lstm_hidden, batch_size=1)
        action = np.random.randn(ACTION_DIM).astype(np.float32)
        log_prob = float(np.random.randn())
        reward = float(np.random.randn())
        done = bool(episode_len and (i + 1) % episode_len == 0)
        value = float(np.random.randn())
        buf.add(tokens, hx, action, log_prob, reward, done, value)


class TestRecurrentRolloutBufferBasic(unittest.TestCase):

    def test_len_increases(self):
        buf = _make_buffer(32)
        tokens = np.zeros((N_ENTITIES, TOKEN_DIM), dtype=np.float32)
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, 1)
        buf.add(tokens, hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0)
        self.assertEqual(len(buf), 1)

    def test_full_raises(self):
        buf = _make_buffer(4)
        _fill_buffer(buf)
        self.assertTrue(buf._full)
        with self.assertRaises(RuntimeError):
            hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, 1)
            buf.add(
                np.zeros((N_ENTITIES, TOKEN_DIM), dtype=np.float32),
                hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0,
            )

    def test_reset_clears_buffer(self):
        buf = _make_buffer(4)
        _fill_buffer(buf)
        buf.reset()
        self.assertEqual(len(buf), 0)
        self.assertFalse(buf._full)

    def test_tokens_padded_to_max_entities(self):
        buf = _make_buffer(4)
        short_tokens = np.ones((3, TOKEN_DIM), dtype=np.float32)
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, 1)
        buf.add(short_tokens, hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0)
        stored = buf.tokens[0]
        self.assertEqual(stored.shape, (N_ENTITIES, TOKEN_DIM))
        np.testing.assert_array_equal(stored[:3], short_tokens)
        np.testing.assert_array_equal(stored[3:], 0.0)

    def test_pad_mask_set_for_short_observations(self):
        buf = _make_buffer(4)
        short_tokens = np.ones((3, TOKEN_DIM), dtype=np.float32)
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, 1)
        buf.add(short_tokens, hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0)
        mask = buf.pad_masks[0]
        self.assertFalse(mask[:3].any(), "Valid positions should not be masked")
        self.assertTrue(mask[3:].all(), "Padded positions should be masked")

    def test_hx_stored_correctly(self):
        buf = _make_buffer(4)
        h_val = torch.full((LSTM_LAYERS, 1, LSTM_HIDDEN), 3.14)
        c_val = torch.full((LSTM_LAYERS, 1, LSTM_HIDDEN), 2.72)
        hx = LSTMHiddenState(h=h_val, c=c_val)
        tokens = np.zeros((N_ENTITIES, TOKEN_DIM), dtype=np.float32)
        buf.add(tokens, hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0)
        np.testing.assert_allclose(buf.hx_h[0], h_val[:, 0, :].numpy(), rtol=1e-5)
        np.testing.assert_allclose(buf.hx_c[0], c_val[:, 0, :].numpy(), rtol=1e-5)


class TestRecurrentRolloutBufferGAE(unittest.TestCase):

    def test_compute_returns_and_advantages_runs(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        self.assertFalse(np.isnan(buf.advantages).any())
        self.assertFalse(np.isnan(buf.returns).any())

    def test_advantages_shape(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        self.assertEqual(buf.advantages.shape, (16,))

    def test_terminal_bootstrap_zero(self):
        """When last_done=True, last_value is not used in bootstrapping."""
        buf = _make_buffer(4)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        adv_done = buf.advantages.copy()

        buf2 = _make_buffer(4)
        _fill_buffer(buf2)
        # Copy the same data
        buf2.tokens[:] = buf.tokens
        buf2.rewards[:] = buf.rewards
        buf2.values[:] = buf.values
        buf2.dones[:] = buf.dones
        buf2._ptr = buf.n_steps
        buf2._full = True
        buf2.compute_returns_and_advantages(last_value=999.0, last_done=True)
        adv_nodone = buf2.advantages.copy()

        # When last_done=True, last_value=999 should not change anything
        np.testing.assert_allclose(adv_done, adv_nodone, rtol=1e-5)

    def test_returns_equals_advantages_plus_values(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        np.testing.assert_allclose(
            buf.returns, buf.advantages + buf.values, rtol=1e-5
        )

    def test_raises_if_not_full(self):
        buf = _make_buffer(16)
        tokens = np.zeros((N_ENTITIES, TOKEN_DIM), dtype=np.float32)
        hx = LSTMHiddenState.zeros(LSTM_LAYERS, LSTM_HIDDEN, 1)
        buf.add(tokens, hx, np.zeros(ACTION_DIM), 0.0, 0.0, False, 0.0)
        with self.assertRaises(RuntimeError):
            buf.compute_returns_and_advantages(0.0, True)


class TestRecurrentRolloutBufferSequences(unittest.TestCase):

    def test_get_sequences_returns_correct_number(self):
        buf = _make_buffer(32)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_sequences(seq_len=8, device=torch.device("cpu"))
        self.assertEqual(len(batches), 4)  # 32 / 8 = 4

    def test_get_sequences_batch_keys(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_sequences(seq_len=8, device=torch.device("cpu"))
        required_keys = {
            "tokens", "pad_masks", "hx_h", "hx_c",
            "actions", "log_probs", "advantages", "returns", "values",
        }
        for batch in batches:
            self.assertEqual(set(batch.keys()), required_keys)

    def test_get_sequences_tensor_shapes(self):
        N_STEPS, SEQ_LEN = 16, 8
        buf = _make_buffer(N_STEPS)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_sequences(seq_len=SEQ_LEN, device=torch.device("cpu"))
        for batch in batches:
            self.assertEqual(
                batch["tokens"].shape, (1, SEQ_LEN, N_ENTITIES, TOKEN_DIM)
            )
            self.assertEqual(
                batch["hx_h"].shape, (LSTM_LAYERS, 1, LSTM_HIDDEN)
            )
            self.assertEqual(
                batch["actions"].shape, (1, SEQ_LEN, ACTION_DIM)
            )
            self.assertEqual(batch["advantages"].shape, (1, SEQ_LEN))

    def test_get_sequences_invalid_seq_len_raises(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        with self.assertRaises(ValueError):
            buf.get_sequences(seq_len=7, device=torch.device("cpu"))

    def test_get_sequences_advantages_normalized(self):
        buf = _make_buffer(32)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_sequences(seq_len=8, device=torch.device("cpu"), normalize_advantages=True)
        all_adv = torch.cat([b["advantages"].flatten() for b in batches])
        # Mean ≈ 0, std ≈ 1
        #: Advantage normalisation is approximate; these tolerances accept
        #: floating-point and sample-size variation from combining 4 batches.
        _MEAN_TOLERANCE_PLACES = 4   # |mean| < 0.0001
        _STD_TOLERANCE_DELTA = 0.05  # |std - 1.0| < 0.05 across 32 samples
        self.assertAlmostEqual(all_adv.mean().item(), 0.0, places=_MEAN_TOLERANCE_PLACES)
        self.assertAlmostEqual(all_adv.std().item(), 1.0, delta=_STD_TOLERANCE_DELTA)

    def test_get_sequences_no_normalization(self):
        buf = _make_buffer(16)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(0.0, True)
        expected = buf.advantages.copy()
        batches = buf.get_sequences(seq_len=8, device=torch.device("cpu"), normalize_advantages=False)
        retrieved = np.concatenate([b["advantages"].squeeze(0).numpy() for b in batches])
        np.testing.assert_allclose(retrieved, expected, rtol=1e-5)


class TestRecurrentRolloutBufferMemoryOverhead(unittest.TestCase):
    """Rollout buffer memory overhead < 20 % vs. a non-recurrent baseline.

    The 20 % bound holds when the LSTM hidden dimension is small relative to
    the entity-token observation.  Here we use a realistic small-LSTM config:
    max_entities=32, lstm_hidden_size=32, lstm_num_layers=1.

    Non-recurrent total (n_steps=512):
      tokens:    512 × 32 × 16 × 4 B ≈ 1 048 576 B
      pad_masks: 512 × 32 × 1  B =    16 384  B
      actions:   512 ×  3 × 4  B =     6 144  B
      6 float arrays (log_probs, rewards, dones, values, adv, ret):
                 512 × 6  × 4  B =    12 288  B
      ─────────────────────────────────────────────────
      total ≈ 1 083 392 B

    LSTM overhead (hx_h + hx_c): 512 × 1 × 32 × 4 × 2 = 131 072 B  (~12%)
    """

    def test_memory_overhead_under_20_percent(self):
        n_steps = 512
        max_entities = 32
        lstm_hidden = 32
        lstm_layers = 1

        buf = RecurrentRolloutBuffer(
            n_steps=n_steps,
            max_entities=max_entities,
            token_dim=TOKEN_DIM,
            action_dim=ACTION_DIM,
            lstm_hidden_size=lstm_hidden,
            lstm_num_layers=lstm_layers,
        )

        # Non-recurrent baseline: all arrays except hx_h and hx_c
        baseline_bytes = (
            n_steps * max_entities * TOKEN_DIM * 4   # tokens (float32)
            + n_steps * max_entities                  # pad_masks (bool = 1 B)
            + n_steps * ACTION_DIM * 4                # actions
            + n_steps * 4 * 6                         # log_probs, rewards, dones, values, adv, ret
        )

        # LSTM-only overhead
        recurrent_overhead_bytes = (
            n_steps * lstm_layers * lstm_hidden * 4 * 2  # hx_h + hx_c
        )
        overhead_fraction = recurrent_overhead_bytes / max(baseline_bytes, 1)
        self.assertLess(
            overhead_fraction, 0.20,
            f"LSTM hidden state overhead {overhead_fraction:.2%} exceeds 20%",
        )


class TestRecurrentRolloutBufferEpisodeBoundary(unittest.TestCase):
    """Hidden states at episode starts are correctly zeroed."""

    def test_hidden_state_zeroed_after_done(self):
        """The hidden state stored at t+1 should be zero when done at t."""
        buf = _make_buffer(8)
        policy = _make_policy()
        hx = policy.initial_state(1)
        tokens_np = np.random.randn(N_ENTITIES, TOKEN_DIM).astype(np.float32)
        tokens_t = torch.tensor(tokens_np).unsqueeze(0)

        for i in range(8):
            # Run policy to get next state
            _, _, hx_new = policy.act(tokens_t, hx)
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            done = (i == 3)  # episode ends at step 3
            # Store hx at the START of this step (before forward pass)
            buf.add(tokens_np, hx, action, 0.0, 0.0, done, 0.0)
            if done:
                # Reset hidden state for the next episode
                hx = policy.initial_state(1)
            else:
                hx = hx_new

        # Hidden state stored at step 4 (first step of new episode) should be zero
        # because we reset hx after done at step 3
        np.testing.assert_array_equal(
            buf.hx_h[4], np.zeros((LSTM_LAYERS, LSTM_HIDDEN)),
            err_msg="hx_h at step after episode reset should be zero",
        )
        np.testing.assert_array_equal(
            buf.hx_c[4], np.zeros((LSTM_LAYERS, LSTM_HIDDEN)),
            err_msg="hx_c at step after episode reset should be zero",
        )


# ---------------------------------------------------------------------------
# End-to-end: policy rollout + buffer + evaluate_actions
# ---------------------------------------------------------------------------


class TestRecurrentPolicyEndToEnd(unittest.TestCase):

    def test_full_rollout_and_evaluate(self):
        """Simulate a short PPO rollout: collect → GAE → evaluate_actions."""
        torch.manual_seed(42)
        np.random.seed(42)

        N_STEPS = 16
        SEQ_LEN = 8
        policy = _make_policy()
        buf = _make_buffer(N_STEPS)

        hx = policy.initial_state(1)
        tokens_np = np.random.randn(N_ENTITIES, TOKEN_DIM).astype(np.float32)

        for i in range(N_STEPS):
            tokens_t = torch.tensor(tokens_np).unsqueeze(0)
            actions, log_probs, hx_new = policy.act(tokens_t, hx)
            values, _ = policy.get_value(tokens_t, hx)
            done = (i + 1) % 8 == 0
            buf.add(
                tokens_np,
                hx,
                actions.squeeze(0).numpy(),
                log_probs.squeeze(0).item(),
                float(np.random.randn()),
                done,
                values.item(),
            )
            if done:
                hx = policy.initial_state(1)
            else:
                hx = hx_new

        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        batches = buf.get_sequences(seq_len=SEQ_LEN, device=torch.device("cpu"))
        self.assertEqual(len(batches), N_STEPS // SEQ_LEN)

        for batch in batches:
            hx_batch = LSTMHiddenState(
                h=batch["hx_h"],
                c=batch["hx_c"],
            )
            log_probs_new, entropy, values_new = policy.evaluate_actions(
                batch["tokens"],
                hx_batch,
                batch["actions"],
                pad_mask_seq=batch["pad_masks"],
            )
            self.assertEqual(log_probs_new.shape, (1, SEQ_LEN))
            self.assertFalse(torch.isnan(log_probs_new).any())
            self.assertFalse(torch.isnan(entropy).any())
            self.assertFalse(torch.isnan(values_new).any())


if __name__ == "__main__":
    unittest.main()
