# SPDX-License-Identifier: MIT
# tests/test_wfm1.py
"""Tests for WFM-1 — Wargames Foundation Model 1 (E12.1).

Coverage
--------
* ScenarioCard — to_tensor() shape, one-hot layout, clamp
* EchelonEncoder — forward shapes, echelon embedding usage
* CrossEchelonTransformer — single and multi-echelon forward
* WFM1Policy — act(), get_value(), evaluate_actions(), evaluate_actions gradient
* WFM1Policy — multi-echelon input path
* WFM1Policy — adapter_parameters() / base_parameters() partition
* WFM1Policy — freeze_base() / unfreeze_base()
* WFM1Policy — save_checkpoint() / load_checkpoint()
* WFM1Policy — finetune_loss() supervised loss
* WFM1Policy — no-NaN outputs with random inputs
* WFM1Policy — shared vs. independent echelon encoder parameter counts
* WFM1RolloutBuffer — add(), is_full, GAE computation, get_batches()
* WFM1MultiTaskTrainer — dry-run (synthetic env, 1 update)
* WFM1BenchmarkScenario — to_scenario_card()
* WFM1BenchmarkConfig — default field values
* WFM1BenchmarkSummary — mean statistics, acceptance criteria
* WFM1Benchmark — dry-run (synthetic env, 1 scenario, 2 episodes)
* HELD_OUT_SCENARIOS — exactly 20 entries, unique names
"""

from __future__ import annotations

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
from models.wfm1 import (
    ECHELON_BATTALION,
    ECHELON_BRIGADE,
    ECHELON_DIVISION,
    ECHELON_CORPS,
    WEATHER_CLEAR,
    WEATHER_RAIN,
    WEATHER_FOG,
    WEATHER_SNOW,
    TERRAIN_PROCEDURAL,
    TERRAIN_GIS_WATERLOO,
    _SCENARIO_CARD_RAW_DIM,
    _N_ECHELONS,
    ScenarioCard,
    EchelonEncoder,
    CrossEchelonTransformer,
    WFM1Policy,
)
from training.wfm1_multitask import (
    EchelonTaskConfig,
    WFM1TrainConfig,
    WFM1RolloutBuffer,
    WFM1MultiTaskTrainer,
)
from training.wfm1_benchmark import (
    HELD_OUT_SCENARIOS,
    WFM1BenchmarkScenario,
    WFM1BenchmarkConfig,
    WFM1BenchmarkResult,
    WFM1BenchmarkSummary,
    WFM1Benchmark,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

BATCH = 4
N_ENTITIES = 8
D_MODEL = 32
N_HEADS = 4
ACTION_DIM = 3


def _make_tokens(batch: int = BATCH, n: int = N_ENTITIES) -> torch.Tensor:
    return torch.randn(batch, n, ENTITY_TOKEN_DIM)


def _make_policy(**kw) -> WFM1Policy:
    defaults = dict(
        token_dim=ENTITY_TOKEN_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_echelon_layers=1,
        n_cross_layers=1,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
        card_hidden_size=16,
    )
    defaults.update(kw)
    return WFM1Policy(**defaults)


# ---------------------------------------------------------------------------
# ScenarioCard
# ---------------------------------------------------------------------------


class TestScenarioCard(unittest.TestCase):

    def test_to_tensor_shape(self):
        card = ScenarioCard()
        vec = card.to_tensor()
        self.assertEqual(vec.shape, (12,))

    def test_echelon_onehot_battalion(self):
        card = ScenarioCard(echelon_level=ECHELON_BATTALION)
        vec = card.to_tensor()
        # Echelon one-hot at indices 1:5
        self.assertEqual(vec[1].item(), 1.0)
        self.assertEqual(vec[2].item(), 0.0)
        self.assertEqual(vec[3].item(), 0.0)
        self.assertEqual(vec[4].item(), 0.0)

    def test_echelon_onehot_corps(self):
        card = ScenarioCard(echelon_level=ECHELON_CORPS)
        vec = card.to_tensor()
        self.assertEqual(vec[4].item(), 1.0)

    def test_weather_onehot_fog(self):
        card = ScenarioCard(weather_code=WEATHER_FOG)
        vec = card.to_tensor()
        self.assertEqual(vec[7].item(), 1.0)  # index 5 + 2

    def test_unit_counts_normalised(self):
        card = ScenarioCard(n_blue_units=32.0, n_red_units=16.0, max_units=64.0)
        vec = card.to_tensor()
        self.assertAlmostEqual(vec[9].item(), 0.5, places=5)
        self.assertAlmostEqual(vec[10].item(), 0.25, places=5)

    def test_terrain_normalised(self):
        card = ScenarioCard(terrain_type=TERRAIN_GIS_WATERLOO)
        vec = card.to_tensor()
        self.assertAlmostEqual(vec[11].item(), 0.25, places=5)

    def test_echelon_clamp_out_of_range(self):
        # echelon = 99 should be clamped to 3
        card = ScenarioCard(echelon_level=99)
        vec = card.to_tensor()
        self.assertEqual(vec[4].item(), 1.0)

    def test_to_tensor_no_nan(self):
        card = ScenarioCard(weather_code=WEATHER_SNOW, terrain_type=TERRAIN_GIS_WATERLOO)
        vec = card.to_tensor()
        self.assertFalse(torch.isnan(vec).any())

    def test_device_argument(self):
        card = ScenarioCard()
        vec = card.to_tensor(device=torch.device("cpu"))
        self.assertEqual(vec.device.type, "cpu")


# ---------------------------------------------------------------------------
# EchelonEncoder
# ---------------------------------------------------------------------------


class TestEchelonEncoder(unittest.TestCase):

    def _make_encoder(self, **kw) -> EchelonEncoder:
        defaults = dict(token_dim=ENTITY_TOKEN_DIM, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1)
        defaults.update(kw)
        return EchelonEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_encoder()
        tokens = _make_tokens()
        out = enc(tokens, echelon=ECHELON_BATTALION)
        self.assertEqual(out.shape, (BATCH, D_MODEL))

    def test_output_dim_property(self):
        enc = self._make_encoder(d_model=64)
        self.assertEqual(enc.output_dim, 64)

    def test_different_echelons_different_output(self):
        enc = self._make_encoder(use_echelon_embedding=True)
        tokens = _make_tokens(batch=1)
        out_bn = enc(tokens, echelon=ECHELON_BATTALION)
        out_corps = enc(tokens, echelon=ECHELON_CORPS)
        # Different echelon embeddings should produce different outputs
        self.assertFalse(torch.allclose(out_bn, out_corps))

    def test_same_echelon_reproducible(self):
        enc = self._make_encoder()
        enc.eval()
        tokens = _make_tokens(batch=2)
        out1 = enc(tokens, echelon=ECHELON_BRIGADE)
        out2 = enc(tokens, echelon=ECHELON_BRIGADE)
        self.assertTrue(torch.allclose(out1, out2))

    def test_padding_mask_applied(self):
        enc = self._make_encoder()
        enc.eval()
        tokens = _make_tokens(batch=2)
        pad_mask = torch.zeros(2, N_ENTITIES, dtype=torch.bool)
        pad_mask[0, 4:] = True  # pad second half for batch item 0
        out_masked = enc(tokens, echelon=ECHELON_BATTALION, pad_mask=pad_mask)
        out_unmasked = enc(tokens, echelon=ECHELON_BATTALION)
        # With padding the outputs differ
        self.assertFalse(torch.allclose(out_masked, out_unmasked))

    def test_no_nan(self):
        enc = self._make_encoder()
        tokens = _make_tokens()
        out = enc(tokens, echelon=ECHELON_DIVISION)
        self.assertFalse(torch.isnan(out).any())

    def test_no_echelon_embedding(self):
        enc = self._make_encoder(use_echelon_embedding=False)
        tokens = _make_tokens()
        out = enc(tokens, echelon=ECHELON_BATTALION)
        self.assertEqual(out.shape, (BATCH, D_MODEL))


# ---------------------------------------------------------------------------
# CrossEchelonTransformer
# ---------------------------------------------------------------------------


class TestCrossEchelonTransformer(unittest.TestCase):

    def _make_cross(self, **kw) -> CrossEchelonTransformer:
        defaults = dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=1)
        defaults.update(kw)
        return CrossEchelonTransformer(**defaults)

    def test_single_echelon_output_shape(self):
        cross = self._make_cross()
        encs = torch.randn(BATCH, 1, D_MODEL)
        ids = torch.tensor([ECHELON_BATTALION])
        out = cross(encs, ids)
        self.assertEqual(out.shape, (BATCH, D_MODEL))

    def test_multi_echelon_output_shape(self):
        cross = self._make_cross()
        encs = torch.randn(BATCH, 4, D_MODEL)
        ids = torch.tensor([0, 1, 2, 3])
        out = cross(encs, ids)
        self.assertEqual(out.shape, (BATCH, D_MODEL))

    def test_no_nan_multi_echelon(self):
        cross = self._make_cross()
        encs = torch.randn(2, 3, D_MODEL)
        ids = torch.tensor([0, 2, 3])
        out = cross(encs, ids)
        self.assertFalse(torch.isnan(out).any())


# ---------------------------------------------------------------------------
# WFM1Policy
# ---------------------------------------------------------------------------


class TestWFM1PolicyAct(unittest.TestCase):

    def test_act_shapes(self):
        policy = _make_policy()
        tokens = _make_tokens()
        actions, log_probs = policy.act(tokens)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))
        self.assertEqual(log_probs.shape, (BATCH,))

    def test_act_no_nan(self):
        policy = _make_policy()
        tokens = _make_tokens()
        actions, log_probs = policy.act(tokens)
        self.assertFalse(torch.isnan(actions).any())
        self.assertFalse(torch.isnan(log_probs).any())

    def test_act_deterministic_reproducible(self):
        policy = _make_policy()
        policy.eval()
        tokens = _make_tokens(batch=1)
        a1, _ = policy.act(tokens, deterministic=True)
        a2, _ = policy.act(tokens, deterministic=True)
        self.assertTrue(torch.allclose(a1, a2))

    def test_act_with_card(self):
        policy = _make_policy()
        tokens = _make_tokens()
        card = ScenarioCard(echelon_level=ECHELON_BATTALION, weather_code=WEATHER_RAIN)
        actions, log_probs = policy.act(tokens, card=card)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))

    def test_act_with_card_vec(self):
        policy = _make_policy()
        tokens = _make_tokens()
        card_vec = torch.zeros(_SCENARIO_CARD_RAW_DIM)
        actions, log_probs = policy.act(tokens, card_vec=card_vec)
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))

    def test_act_multi_echelon(self):
        policy = _make_policy()
        tokens_per_echelon = {
            ECHELON_BATTALION: _make_tokens(),
            ECHELON_BRIGADE: _make_tokens(n=16),
        }
        actions, log_probs = policy.act(
            tokens=None,
            tokens_per_echelon=tokens_per_echelon,
        )
        self.assertEqual(actions.shape, (BATCH, ACTION_DIM))

    def test_act_with_padding_mask(self):
        policy = _make_policy()
        tokens = _make_tokens()
        pad_mask = torch.zeros(BATCH, N_ENTITIES, dtype=torch.bool)
        pad_mask[:, 6:] = True
        actions, _ = policy.act(tokens, pad_mask=pad_mask)
        self.assertFalse(torch.isnan(actions).any())


class TestWFM1PolicyValue(unittest.TestCase):

    def test_get_value_shape(self):
        policy = _make_policy()
        tokens = _make_tokens()
        values = policy.get_value(tokens)
        self.assertEqual(values.shape, (BATCH,))

    def test_get_value_no_nan(self):
        policy = _make_policy()
        tokens = _make_tokens()
        values = policy.get_value(tokens)
        self.assertFalse(torch.isnan(values).any())


class TestWFM1PolicyEvaluateActions(unittest.TestCase):

    def test_evaluate_actions_shapes(self):
        policy = _make_policy()
        tokens = _make_tokens()
        actions = torch.randn(BATCH, ACTION_DIM)
        log_probs, entropy, values = policy.evaluate_actions(tokens, actions)
        self.assertEqual(log_probs.shape, (BATCH,))
        self.assertEqual(entropy.shape, (BATCH,))
        self.assertEqual(values.shape, (BATCH,))

    def test_evaluate_actions_gradient_flows(self):
        policy = _make_policy()
        tokens = _make_tokens()
        actions = torch.randn(BATCH, ACTION_DIM)
        log_probs, entropy, values = policy.evaluate_actions(tokens, actions)
        loss = -log_probs.mean() - 0.01 * entropy.mean() + values.mean()
        loss.backward()
        # Check that at least some parameters received gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad)


class TestWFM1PolicyAdapterAPI(unittest.TestCase):

    def test_adapter_parameters_non_empty(self):
        policy = _make_policy()
        adapter_params = policy.adapter_parameters()
        self.assertGreater(len(adapter_params), 0)

    def test_base_parameters_non_empty(self):
        policy = _make_policy()
        base_params = policy.base_parameters()
        self.assertGreater(len(base_params), 0)

    def test_adapter_and_base_partition_all_parameters(self):
        policy = _make_policy()
        adapter_ids = {id(p) for p in policy.adapter_parameters()}
        base_ids = {id(p) for p in policy.base_parameters()}
        all_ids = {id(p) for p in policy.parameters()}
        # Partition should cover all parameters with no overlap
        self.assertEqual(adapter_ids | base_ids, all_ids)
        self.assertEqual(len(adapter_ids & base_ids), 0)

    def test_freeze_base_sets_requires_grad_false(self):
        policy = _make_policy()
        policy.freeze_base()
        base_params = policy.base_parameters()
        for p in base_params:
            self.assertFalse(p.requires_grad)

    def test_unfreeze_base_restores_requires_grad(self):
        policy = _make_policy()
        policy.freeze_base()
        policy.unfreeze_base()
        for p in policy.parameters():
            self.assertTrue(p.requires_grad)

    def test_adapter_still_trainable_after_freeze(self):
        policy = _make_policy()
        policy.freeze_base()
        adapter_params = policy.adapter_parameters()
        for p in adapter_params:
            self.assertTrue(p.requires_grad)


class TestWFM1PolicySharedEncoder(unittest.TestCase):

    def test_shared_encoder_fewer_params_than_separate(self):
        shared = _make_policy(share_echelon_encoders=True)
        separate = _make_policy(share_echelon_encoders=False)
        n_shared = sum(p.numel() for p in shared.parameters())
        n_sep = sum(p.numel() for p in separate.parameters())
        # Shared should have fewer (or equal) parameters
        self.assertLessEqual(n_shared, n_sep)

    def test_independent_encoders_different_parameters(self):
        policy = _make_policy(share_echelon_encoders=False)
        # All four encoders should be distinct modules
        encoders = list(policy.echelon_encoders)
        for i in range(len(encoders)):
            for j in range(i + 1, len(encoders)):
                self.assertIsNot(encoders[i], encoders[j])


class TestWFM1PolicyCheckpoint(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        policy = _make_policy()
        tokens = _make_tokens(batch=1)

        with torch.no_grad():
            actions_before, _ = policy.act(tokens, deterministic=True)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "wfm1_test.pt"
            policy.save_checkpoint(path)
            loaded = WFM1Policy.load_checkpoint(path)

        with torch.no_grad():
            actions_after, _ = loaded.act(tokens, deterministic=True)

        self.assertTrue(torch.allclose(actions_before, actions_after, atol=1e-5))

    def test_checkpoint_file_created(self):
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "wfm1.pt"
            returned_path = policy.save_checkpoint(path)
            self.assertTrue(returned_path.exists())


class TestWFM1PolicyFinetuneLoss(unittest.TestCase):

    def test_finetune_loss_scalar(self):
        policy = _make_policy()
        batch = {
            "tokens": _make_tokens(),
            "actions": torch.randn(BATCH, ACTION_DIM),
        }
        loss = policy.finetune_loss(batch)
        self.assertEqual(loss.shape, ())

    def test_finetune_loss_non_negative(self):
        policy = _make_policy()
        batch = {
            "tokens": _make_tokens(),
            "actions": torch.randn(BATCH, ACTION_DIM),
        }
        loss = policy.finetune_loss(batch)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_finetune_loss_with_card_vec(self):
        policy = _make_policy()
        batch = {
            "tokens": _make_tokens(),
            "actions": torch.randn(BATCH, ACTION_DIM),
            "card_vec": torch.zeros(_SCENARIO_CARD_RAW_DIM),
        }
        loss = policy.finetune_loss(batch)
        self.assertFalse(torch.isnan(loss))

    def test_finetune_loss_zero_for_perfect_prediction(self):
        """Loss should be near-zero when actions match actor mean exactly."""
        policy = _make_policy()
        tokens = _make_tokens()
        with torch.no_grad():
            # Get the true actor mean
            from models.wfm1 import ECHELON_BATTALION
            fused = policy._encode_single(tokens)
            target_actions = policy.actor_head(fused).detach()

        batch = {
            "tokens": tokens,
            "actions": target_actions,
        }
        loss = policy.finetune_loss(batch)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


# ---------------------------------------------------------------------------
# WFM1RolloutBuffer
# ---------------------------------------------------------------------------


class TestWFM1RolloutBuffer(unittest.TestCase):

    def _make_buf(self, n_steps=16, max_entities=8) -> WFM1RolloutBuffer:
        return WFM1RolloutBuffer(
            n_steps=n_steps,
            max_entities=max_entities,
            token_dim=ENTITY_TOKEN_DIM,
            action_dim=ACTION_DIM,
        )

    def _fill_buf(self, buf: WFM1RolloutBuffer) -> None:
        tokens = np.zeros((buf.max_entities, ENTITY_TOKEN_DIM), dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(buf.n_steps):
            buf.add(tokens=tokens, action=action, log_prob=0.0,
                    reward=1.0, done=False, value=0.5)

    def test_not_full_initially(self):
        buf = self._make_buf()
        self.assertFalse(buf.is_full)

    def test_is_full_after_n_steps(self):
        buf = self._make_buf(n_steps=8)
        self._fill_buf(buf)
        self.assertTrue(buf.is_full)

    def test_add_raises_when_full(self):
        buf = self._make_buf(n_steps=4)
        self._fill_buf(buf)
        with self.assertRaises(RuntimeError):
            buf.add(
                tokens=np.zeros((4, ENTITY_TOKEN_DIM)),
                action=np.zeros(ACTION_DIM),
                log_prob=0.0, reward=0.0, done=False, value=0.0,
            )

    def test_reset_clears_buffer(self):
        buf = self._make_buf(n_steps=4)
        self._fill_buf(buf)
        buf.reset()
        self.assertFalse(buf.is_full)

    def test_gae_computation(self):
        buf = self._make_buf(n_steps=8)
        self._fill_buf(buf)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        # All rewards = 1; advantages should be positive
        self.assertTrue(np.all(buf.advantages >= 0))

    def test_get_batches_shapes(self):
        buf = self._make_buf(n_steps=16, max_entities=8)
        self._fill_buf(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_batches(batch_size=4, device=torch.device("cpu"))
        self.assertGreater(len(batches), 0)
        first = batches[0]
        self.assertIn("tokens", first)
        self.assertIn("actions", first)
        self.assertIn("advantages", first)

    def test_get_batches_token_shape(self):
        n = 16
        buf = self._make_buf(n_steps=n, max_entities=8)
        self._fill_buf(buf)
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_batches(batch_size=n, device=torch.device("cpu"))
        tokens = batches[0]["tokens"]
        self.assertEqual(tokens.shape, (n, 8, ENTITY_TOKEN_DIM))

    def test_pad_mask_stored_correctly(self):
        buf = self._make_buf(n_steps=4, max_entities=8)
        tokens = np.zeros((6, ENTITY_TOKEN_DIM), dtype=np.float32)  # 6 < 8
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(4):
            buf.add(tokens=tokens, action=action, log_prob=0.0,
                    reward=1.0, done=False, value=0.5)
        # Positions 6:8 should be True (padded)
        self.assertTrue(np.all(buf.pad_masks[:, 6:]))

    def test_advantage_normalisation(self):
        buf = self._make_buf(n_steps=16)
        rng = np.random.default_rng(0)
        tokens = np.zeros((8, ENTITY_TOKEN_DIM), dtype=np.float32)
        for _ in range(16):
            buf.add(
                tokens=tokens,
                action=np.zeros(ACTION_DIM, dtype=np.float32),
                log_prob=0.0,
                reward=float(rng.standard_normal()),
                done=False,
                value=float(rng.standard_normal()),
            )
        buf.compute_returns_and_advantages(0.0, True)
        batches = buf.get_batches(
            batch_size=16,
            device=torch.device("cpu"),
            normalize_advantages=True,
        )
        adv = batches[0]["advantages"].numpy()
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(adv.std()), 1.0, places=3)


# ---------------------------------------------------------------------------
# WFM1MultiTaskTrainer — dry-run
# ---------------------------------------------------------------------------


class TestWFM1MultiTaskTrainerDryRun(unittest.TestCase):

    def _make_cfg(self) -> WFM1TrainConfig:
        return WFM1TrainConfig(
            total_steps=64,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_echelon_layers=1,
            n_cross_layers=1,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
            card_hidden_size=16,
            echelon_tasks=[
                EchelonTaskConfig(
                    echelon=ECHELON_BATTALION,
                    env_id="battalion",
                    max_entities=8,
                    action_dim=ACTION_DIM,
                )
            ],
            wandb_project=None,
            checkpoint_interval=9999,  # don't checkpoint during test
            device="cpu",
        )

    def test_train_returns_result(self):
        cfg = self._make_cfg()
        trainer = WFM1MultiTaskTrainer(cfg)
        result = trainer.train()
        self.assertIsNotNone(result)
        self.assertGreater(result.total_steps, 0)

    def test_train_result_has_reward_dict(self):
        cfg = self._make_cfg()
        trainer = WFM1MultiTaskTrainer(cfg)
        result = trainer.train()
        self.assertIn(str(ECHELON_BATTALION), result.mean_reward_per_echelon)

    def test_train_multi_echelon_dry_run(self):
        cfg = WFM1TrainConfig(
            total_steps=64,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_echelon_layers=1,
            n_cross_layers=1,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
            card_hidden_size=16,
            echelon_tasks=[
                EchelonTaskConfig(echelon=ECHELON_BATTALION, env_id="battalion",
                                  max_entities=8, action_dim=ACTION_DIM),
                EchelonTaskConfig(echelon=ECHELON_BRIGADE, env_id="brigade",
                                  max_entities=12, action_dim=ACTION_DIM),
            ],
            wandb_project=None,
            checkpoint_interval=9999,
            device="cpu",
        )
        trainer = WFM1MultiTaskTrainer(cfg)
        result = trainer.train()
        self.assertGreater(result.total_steps, 0)

    def test_existing_policy_accepted(self):
        policy = _make_policy()
        cfg = self._make_cfg()
        trainer = WFM1MultiTaskTrainer(cfg, policy=policy)
        result = trainer.train()
        self.assertGreater(result.total_steps, 0)


# ---------------------------------------------------------------------------
# WFM1BenchmarkScenario
# ---------------------------------------------------------------------------


class TestWFM1BenchmarkScenario(unittest.TestCase):

    def test_from_dict(self):
        d = HELD_OUT_SCENARIOS[0]
        s = WFM1BenchmarkScenario.from_dict(d)
        self.assertEqual(s.name, d["name"])
        self.assertEqual(s.echelon, d["echelon"])

    def test_to_scenario_card(self):
        d = HELD_OUT_SCENARIOS[0]
        s = WFM1BenchmarkScenario.from_dict(d)
        card = s.to_scenario_card()
        self.assertIsInstance(card, ScenarioCard)
        self.assertEqual(card.echelon_level, s.echelon)
        self.assertEqual(card.weather_code, s.weather_code)

    def test_to_scenario_card_vec_shape(self):
        s = WFM1BenchmarkScenario.from_dict(HELD_OUT_SCENARIOS[3])
        vec = s.to_scenario_card().to_tensor()
        self.assertEqual(vec.shape, (_SCENARIO_CARD_RAW_DIM,))


# ---------------------------------------------------------------------------
# HELD_OUT_SCENARIOS
# ---------------------------------------------------------------------------


class TestHeldOutScenarios(unittest.TestCase):

    def test_exactly_20_scenarios(self):
        self.assertEqual(len(HELD_OUT_SCENARIOS), 20)

    def test_unique_names(self):
        names = [s["name"] for s in HELD_OUT_SCENARIOS]
        self.assertEqual(len(names), len(set(names)))

    def test_unique_seeds(self):
        seeds = [s["seed"] for s in HELD_OUT_SCENARIOS]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_all_have_required_keys(self):
        required = {"name", "echelon", "terrain", "weather", "n_blue", "n_red", "seed"}
        for s in HELD_OUT_SCENARIOS:
            self.assertEqual(required, required & s.keys())

    def test_echelon_values_valid(self):
        for s in HELD_OUT_SCENARIOS:
            self.assertIn(s["echelon"], [0, 1, 2, 3])

    def test_weather_values_valid(self):
        for s in HELD_OUT_SCENARIOS:
            self.assertIn(s["weather"], [WEATHER_CLEAR, WEATHER_RAIN, WEATHER_FOG, WEATHER_SNOW])


# ---------------------------------------------------------------------------
# WFM1BenchmarkSummary
# ---------------------------------------------------------------------------


class TestWFM1BenchmarkSummary(unittest.TestCase):

    def _make_summary(
        self,
        zs_wr: float = 0.6,
        ft_wr: float = 0.7,
        sp_wr: float = 0.8,
    ) -> WFM1BenchmarkSummary:
        cfg = WFM1BenchmarkConfig(n_eval_episodes=5, n_scenarios=2)
        results = [
            WFM1BenchmarkResult(
                scenario_name="s1",
                condition="zero_shot",
                win_rate=zs_wr,
                mean_steps=100.0,
                std_steps=10.0,
                n_episodes=5,
            ),
            WFM1BenchmarkResult(
                scenario_name="s1",
                condition="finetuned",
                win_rate=ft_wr,
                mean_steps=90.0,
                std_steps=8.0,
                n_episodes=5,
                finetune_steps_used=10000,
            ),
            WFM1BenchmarkResult(
                scenario_name="s1",
                condition="specialist",
                win_rate=sp_wr,
                mean_steps=85.0,
                std_steps=7.0,
                n_episodes=5,
            ),
        ]
        return WFM1BenchmarkSummary(results=results, config=cfg)

    def test_mean_zero_shot_win_rate(self):
        s = self._make_summary(zs_wr=0.6)
        self.assertAlmostEqual(s.mean_zero_shot_win_rate, 0.6)

    def test_finetune_recovery(self):
        s = self._make_summary(ft_wr=0.8, sp_wr=1.0)
        self.assertAlmostEqual(s.finetune_recovery, 0.8)

    def test_meets_zero_shot_criterion_pass(self):
        s = self._make_summary(zs_wr=0.6)  # ≥ 0.55
        self.assertTrue(s.meets_zero_shot_criterion)

    def test_meets_zero_shot_criterion_fail(self):
        s = self._make_summary(zs_wr=0.3)  # < 0.55
        self.assertFalse(s.meets_zero_shot_criterion)

    def test_meets_finetune_criterion_pass(self):
        s = self._make_summary(ft_wr=0.8, sp_wr=1.0)  # 80% recovery
        self.assertTrue(s.meets_finetune_criterion)

    def test_meets_finetune_criterion_fail(self):
        s = self._make_summary(ft_wr=0.5, sp_wr=1.0)  # 50% < 80%
        self.assertFalse(s.meets_finetune_criterion)

    def test_all_criteria_met(self):
        s = self._make_summary(zs_wr=0.6, ft_wr=0.8, sp_wr=1.0)
        self.assertTrue(s.all_criteria_met)

    def test_str_contains_win_rates(self):
        s = self._make_summary()
        text = str(s)
        self.assertIn("zero-shot", text.lower())

    def test_write_markdown(self):
        s = self._make_summary()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bench.md"
            returned = s.write_markdown(p)
            self.assertTrue(returned.exists())
            content = returned.read_text()
            self.assertIn("WFM-1 Benchmark", content)


# ---------------------------------------------------------------------------
# WFM1Benchmark — dry-run
# ---------------------------------------------------------------------------


class TestWFM1BenchmarkDryRun(unittest.TestCase):

    def _make_cfg(self) -> WFM1BenchmarkConfig:
        return WFM1BenchmarkConfig(
            n_eval_episodes=2,
            n_scenarios=2,
            finetune_steps=5,
            max_steps_per_episode=10,
            specialist_train_steps=0,
        )

    def test_run_with_none_policy(self):
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=None)
        self.assertIsNotNone(summary)
        # Should have 3 conditions × 2 scenarios = 6 results
        self.assertEqual(len(summary.results), 6)

    def test_run_with_wfm1_policy(self):
        policy = _make_policy()
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=policy)
        self.assertIsNotNone(summary)
        zs = [r for r in summary.results if r.condition == "zero_shot"]
        self.assertEqual(len(zs), 2)

    def test_run_conditions_present(self):
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=None)
        conditions = {r.condition for r in summary.results}
        self.assertEqual(conditions, {"zero_shot", "finetuned", "specialist"})

    def test_run_win_rates_in_range(self):
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=None)
        for r in summary.results:
            self.assertGreaterEqual(r.win_rate, 0.0)
            self.assertLessEqual(r.win_rate, 1.0)

    def test_run_finetune_steps_recorded(self):
        policy = _make_policy()
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=policy)
        ft_results = [r for r in summary.results if r.condition == "finetuned"]
        for r in ft_results:
            self.assertEqual(r.finetune_steps_used, cfg.finetune_steps)

    def test_summary_str(self):
        cfg = self._make_cfg()
        bench = WFM1Benchmark(cfg)
        summary = bench.run(wfm1_policy=None)
        text = str(summary)
        self.assertIn("WFM-1", text)


if __name__ == "__main__":
    unittest.main()
