# tests/test_python_api.py
"""Comprehensive tests for the whole-project Python API.

Validates that all public symbols are importable from the top-level package
namespaces (``envs``, ``models``, ``training``, ``analysis``) and that
the primary training workflow can be exercised programmatically without
touching Hydra or YAML configs.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# envs package — public API surface
# ---------------------------------------------------------------------------


class TestEnvsPackageAPI(unittest.TestCase):
    """All public names must be importable directly from ``envs``."""

    def test_environments_importable(self) -> None:
        from envs import (  # noqa: F401
            BattalionEnv,
            BrigadeEnv,
            DivisionEnv,
            CorpsEnv,
            CavalryCorpsEnv,
            ArtilleryCorpsEnv,
            MultiBattalionEnv,
        )

    def test_reward_utilities_importable(self) -> None:
        from envs import RewardWeights, RewardComponents, compute_reward  # noqa: F401

    def test_battalion_constants_importable(self) -> None:
        from envs import (  # noqa: F401
            DESTROYED_THRESHOLD,
            MAP_WIDTH,
            MAP_HEIGHT,
            MAX_STEPS,
        )

    def test_configuration_types_importable(self) -> None:
        from envs import (  # noqa: F401
            LogisticsConfig,
            LogisticsState,
            SupplyWagon,
            MoraleConfig,
            Formation,
            WeatherConfig,
            WeatherState,
        )

    def test_corps_constants_importable(self) -> None:
        from envs import (  # noqa: F401
            CORPS_OBS_DIM,
            N_CORPS_SECTORS,
            N_OBJECTIVES,
            CORPS_MAP_WIDTH,
            CORPS_MAP_HEIGHT,
            N_ROAD_FEATURES,
        )

    def test_simulation_primitives_importable(self) -> None:
        from envs import SimEngine, EpisodeResult  # noqa: F401

    def test_hrl_options_importable(self) -> None:
        from envs import MacroAction, Option, make_default_options, SMDPWrapper  # noqa: F401

    def test_reward_weights_instantiate(self) -> None:
        from envs import RewardWeights
        rw = RewardWeights(win_bonus=20.0, loss_penalty=-20.0)
        self.assertEqual(rw.win_bonus, 20.0)
        self.assertEqual(rw.loss_penalty, -20.0)

    def test_battalion_env_uses_reward_weights(self) -> None:
        from envs import BattalionEnv, RewardWeights
        rw = RewardWeights(win_bonus=15.0)
        env = BattalionEnv(reward_weights=rw, randomize_terrain=False)
        try:
            obs, _ = env.reset(seed=0)
            self.assertIsNotNone(obs)
        finally:
            env.close()

    def test_formation_enum_values(self) -> None:
        from envs import Formation
        # Enum should have at least a line and column formation
        names = {f.name for f in Formation}
        self.assertGreater(len(names), 0)

    def test_all_list_complete(self) -> None:
        import envs
        for name in envs.__all__:
            self.assertTrue(
                hasattr(envs, name),
                f"envs.__all__ lists '{name}' but it is not an attribute of envs",
            )


# ---------------------------------------------------------------------------
# models package — public API surface
# ---------------------------------------------------------------------------


class TestModelsPackageAPI(unittest.TestCase):
    """All public names must be importable directly from ``models``."""

    def test_mlp_policy_importable(self) -> None:
        from models import BattalionMlpPolicy  # noqa: F401

    def test_mappo_importable(self) -> None:
        from models import MAPPOActor, MAPPOCritic, MAPPOPolicy  # noqa: F401

    def test_entity_encoder_importable(self) -> None:
        from models import (  # noqa: F401
            ENTITY_TOKEN_DIM,
            EntityEncoder,
            EntityActorCriticPolicy,
            SpatialPositionalEncoding,
        )

    def test_recurrent_policy_importable(self) -> None:
        from models import (  # noqa: F401
            LSTMHiddenState,
            RecurrentEntityEncoder,
            RecurrentActorCriticPolicy,
            RecurrentRolloutBuffer,
        )

    def test_wfm1_importable(self) -> None:
        from models import (  # noqa: F401
            WFM1Policy,
            ScenarioCard,
            EchelonEncoder,
            CrossEchelonTransformer,
        )

    def test_wfm1_constants_importable(self) -> None:
        from models import (  # noqa: F401
            ECHELON_BATTALION,
            ECHELON_BRIGADE,
            ECHELON_DIVISION,
            ECHELON_CORPS,
            WEATHER_CLEAR,
            TERRAIN_PROCEDURAL,
        )

    def test_entity_token_dim_value(self) -> None:
        from models import ENTITY_TOKEN_DIM
        self.assertIsInstance(ENTITY_TOKEN_DIM, int)
        self.assertGreater(ENTITY_TOKEN_DIM, 0)

    def test_all_list_complete(self) -> None:
        import models
        for name in models.__all__:
            self.assertTrue(
                hasattr(models, name),
                f"models.__all__ lists '{name}' but it is not an attribute of models",
            )


# ---------------------------------------------------------------------------
# training package — public API surface
# ---------------------------------------------------------------------------


class TestTrainingPackageAPI(unittest.TestCase):
    """All public names must be importable directly from ``training``."""

    def test_train_function_importable(self) -> None:
        from training import train  # noqa: F401
        self.assertTrue(callable(train))

    def test_training_config_importable(self) -> None:
        from training import TrainingConfig  # noqa: F401

    def test_evaluation_importable(self) -> None:
        from training import (  # noqa: F401
            evaluate,
            evaluate_detailed,
            run_episodes_with_model,
            EvaluationResult,
        )

    def test_callbacks_importable(self) -> None:
        from training import (  # noqa: F401
            WandbCallback,
            RewardBreakdownCallback,
            EloEvalCallback,
            ManifestCheckpointCallback,
            ManifestEvalCallback,
        )

    def test_self_play_importable(self) -> None:
        from training import (  # noqa: F401
            OpponentPool,
            SelfPlayCallback,
            WinRateVsPoolCallback,
            evaluate_vs_pool,
            TeamOpponentPool,
            evaluate_team_vs_pool,
            nash_exploitability_proxy,
        )

    def test_curriculum_importable(self) -> None:
        from training import (  # noqa: F401
            CurriculumScheduler,
            CurriculumStage,
            load_v1_weights_into_mappo,
        )

    def test_policy_registry_importable(self) -> None:
        from training import PolicyRegistry, Echelon, PolicyEntry  # noqa: F401

    def test_artifacts_importable(self) -> None:
        from training import (  # noqa: F401
            CheckpointManifest,
            checkpoint_name_prefix,
            checkpoint_final_stem,
            checkpoint_best_filename,
            parse_step_from_checkpoint_name,
        )

    def test_elo_importable(self) -> None:
        from training import (  # noqa: F401
            EloRegistry,
            TeamEloRegistry,
            DEFAULT_RATING,
            BASELINE_RATINGS,
        )

    def test_benchmarks_importable(self) -> None:
        from training import (  # noqa: F401
            WFM1Benchmark,
            WFM1BenchmarkConfig,
            WFM1BenchmarkResult,
            WFM1BenchmarkSummary,
            TransferBenchmark,
            TransferEvalConfig,
            TransferResult,
            TransferSummary,
            HistoricalBenchmark,
            BenchmarkEntry,
            BenchmarkSummary,
        )

    def test_all_list_complete(self) -> None:
        import training
        for name in training.__all__:
            self.assertTrue(
                hasattr(training, name),
                f"training.__all__ lists '{name}' but it is not an attribute of training",
            )


# ---------------------------------------------------------------------------
# analysis package — public API surface
# ---------------------------------------------------------------------------


class TestAnalysisPackageAPI(unittest.TestCase):
    """All public names must be importable directly from ``analysis``."""

    def test_coa_importable(self) -> None:
        from analysis import (  # noqa: F401
            COAScore,
            CourseOfAction,
            COAGenerator,
            generate_coas,
            STRATEGY_LABELS,
        )

    def test_corps_coa_importable(self) -> None:
        from analysis import (  # noqa: F401
            CorpsCOAScore,
            CorpsCourseOfAction,
            COAExplanation,
            COAModification,
            CorpsCOAGenerator,
            generate_corps_coas,
            CORPS_STRATEGY_LABELS,
        )

    def test_saliency_importable(self) -> None:
        from analysis import (  # noqa: F401
            OBSERVATION_FEATURES,
            SaliencyAnalyzer,
            compute_gradient_saliency,
            compute_integrated_gradients,
            compute_shap_importance,
            plot_saliency_map,
            plot_feature_importance,
        )

    def test_observation_features_length(self) -> None:
        from analysis import OBSERVATION_FEATURES
        from envs import BattalionEnv

        # Derive expected observation length from BattalionEnv with optional
        # features disabled so that OBSERVATION_FEATURES stays in sync with
        # the core observation schema.
        env = BattalionEnv(
            enable_formations=False,
            enable_logistics=False,
            enable_weather=False,
        )
        try:
            obs_dim = env.observation_space.shape[0]
        finally:
            env.close()

        self.assertEqual(len(OBSERVATION_FEATURES), obs_dim)

    def test_all_list_complete(self) -> None:
        import analysis
        for name in analysis.__all__:
            self.assertTrue(
                hasattr(analysis, name),
                f"analysis.__all__ lists '{name}' but it is not an attribute of analysis",
            )


# ---------------------------------------------------------------------------
# TrainingConfig — dataclass behaviour
# ---------------------------------------------------------------------------


class TestTrainingConfig(unittest.TestCase):
    """Unit tests for the TrainingConfig dataclass."""

    def test_default_instantiation(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig()
        self.assertEqual(cfg.total_timesteps, 1_000_000)
        self.assertEqual(cfg.n_envs, 8)
        self.assertEqual(cfg.curriculum_level, 5)
        self.assertEqual(cfg.seed, 42)
        self.assertTrue(cfg.enable_wandb)
        self.assertFalse(cfg.enable_self_play)
        self.assertEqual(cfg.elo_opponents, [])

    def test_custom_values(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig(
            total_timesteps=500_000,
            n_envs=4,
            curriculum_level=3,
            seed=7,
        )
        self.assertEqual(cfg.total_timesteps, 500_000)
        self.assertEqual(cfg.n_envs, 4)
        self.assertEqual(cfg.curriculum_level, 3)
        self.assertEqual(cfg.seed, 7)

    def test_dataclass_replace(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig()
        cfg2 = dataclasses.replace(cfg, total_timesteps=200_000)
        self.assertEqual(cfg2.total_timesteps, 200_000)
        self.assertEqual(cfg2.n_envs, cfg.n_envs)  # unchanged

    def test_asdict(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig()
        d = dataclasses.asdict(cfg)
        self.assertIn("total_timesteps", d)
        self.assertIn("learning_rate", d)
        self.assertIn("reward_win_bonus", d)
        self.assertIn("enable_wandb", d)

    def test_reward_weight_fields(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig(
            reward_win_bonus=20.0,
            reward_loss_penalty=-20.0,
        )
        self.assertEqual(cfg.reward_win_bonus, 20.0)
        self.assertEqual(cfg.reward_loss_penalty, -20.0)

    def test_wandb_fields(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig(
            enable_wandb=False,
            wandb_project="my_project",
            wandb_entity="my_team",
        )
        self.assertFalse(cfg.enable_wandb)
        self.assertEqual(cfg.wandb_project, "my_project")
        self.assertEqual(cfg.wandb_entity, "my_team")

    def test_self_play_fields(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig(
            enable_self_play=True,
            self_play_pool_max_size=5,
            self_play_snapshot_freq=25_000,
        )
        self.assertTrue(cfg.enable_self_play)
        self.assertEqual(cfg.self_play_pool_max_size, 5)

    def test_elo_opponents_list(self) -> None:
        from training import TrainingConfig
        cfg = TrainingConfig(elo_opponents=["scripted_l1", "scripted_l3"])
        self.assertEqual(cfg.elo_opponents, ["scripted_l1", "scripted_l3"])


# ---------------------------------------------------------------------------
# train() function — validation and programmatic usage
# ---------------------------------------------------------------------------


class TestTrainFunction(unittest.TestCase):
    """Tests for the programmatic train() API."""

    def test_train_is_callable(self) -> None:
        from training import train
        self.assertTrue(callable(train))

    def test_train_unknown_kwarg_raises(self) -> None:
        from training import train
        with self.assertRaises(ValueError) as ctx:
            train(total_timesteps=1, not_a_real_field=True)
        self.assertIn("not_a_real_field", str(ctx.exception))

    def test_train_zero_timesteps_raises(self) -> None:
        from training import train
        with self.assertRaises(ValueError):
            train(total_timesteps=0)

    def test_train_zero_envs_raises(self) -> None:
        from training import train
        with self.assertRaises(ValueError):
            train(total_timesteps=1, n_envs=0)

    def test_train_invalid_checkpoint_freq_raises(self) -> None:
        from training import train
        with self.assertRaises(ValueError):
            train(total_timesteps=1, checkpoint_freq=0)

    def test_train_resume_missing_path_raises(self) -> None:
        from training import train
        with self.assertRaises(FileNotFoundError):
            train(total_timesteps=1, resume="/nonexistent/checkpoint.zip")

    def test_train_returns_ppo_model(self) -> None:
        """train() with minimal config completes and returns a PPO model."""
        from stable_baselines3 import PPO
        # training.__init__ exports train as a function; sys.modules gives the actual module
        _train_mod_ref = sys.modules["training.train"]
        from training import train, TrainingConfig

        cfg = TrainingConfig(
            total_timesteps=32,
            n_envs=1,
            n_steps=32,
            batch_size=16,
            checkpoint_freq=100,
            eval_freq=100,
            enable_wandb=False,
            write_manifest=False,
            keep_legacy_aliases=False,
            prune_on_run_end=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
            )
            with patch.object(_train_mod_ref, "wandb"):
                model = train(cfg)
        self.assertIsInstance(model, PPO)

    def test_train_kwarg_overrides(self) -> None:
        """Keyword overrides to train() are applied to the config."""
        from stable_baselines3 import PPO
        # training.__init__ exports train as a function; sys.modules gives the actual module
        _train_mod_ref = sys.modules["training.train"]
        from training import train

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_train_mod_ref, "wandb"):
                model = train(
                    total_timesteps=32,
                    n_envs=1,
                    n_steps=32,
                    batch_size=16,
                    checkpoint_freq=100,
                    eval_freq=100,
                    enable_wandb=False,
                    write_manifest=False,
                    keep_legacy_aliases=False,
                    prune_on_run_end=False,
                    checkpoint_dir=tmpdir,
                    log_dir=tmpdir,
                )
        self.assertIsInstance(model, PPO)

    def test_train_config_then_kwarg_override(self) -> None:
        """Config + kwarg override: the kwarg wins."""
        from stable_baselines3 import PPO
        # training.__init__ exports train as a function; sys.modules gives the actual module
        _train_mod_ref = sys.modules["training.train"]
        from training import train, TrainingConfig

        cfg = TrainingConfig(
            total_timesteps=64,   # will be overridden
            n_envs=1,
            n_steps=32,
            batch_size=16,
            checkpoint_freq=100,
            eval_freq=100,
            enable_wandb=False,
            write_manifest=False,
            keep_legacy_aliases=False,
            prune_on_run_end=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = dataclasses.replace(cfg, checkpoint_dir=tmpdir, log_dir=tmpdir)
            with patch.object(_train_mod_ref, "wandb"):
                model = train(cfg, total_timesteps=32)
        self.assertIsInstance(model, PPO)

    def test_train_saves_checkpoint(self) -> None:
        """train() writes at least one .zip file to checkpoint_dir."""
        # training.__init__ exports train as a function; sys.modules gives the actual module
        _train_mod_ref = sys.modules["training.train"]
        from training import train

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_train_mod_ref, "wandb"):
                train(
                    total_timesteps=32,
                    n_envs=1,
                    n_steps=32,
                    batch_size=16,
                    checkpoint_freq=100,
                    eval_freq=100,
                    enable_wandb=False,
                    write_manifest=False,
                    keep_legacy_aliases=False,
                    prune_on_run_end=False,
                    checkpoint_dir=tmpdir,
                    log_dir=tmpdir,
                )
            zips = list(Path(tmpdir).rglob("*.zip"))
        self.assertGreater(len(zips), 0, "No .zip checkpoints were saved")


# ---------------------------------------------------------------------------
# EvaluationResult — structured evaluation API
# ---------------------------------------------------------------------------


class TestEvaluationResultAPI(unittest.TestCase):
    """EvaluationResult is importable and usable from training package."""

    def test_result_importable_from_training(self) -> None:
        from training import EvaluationResult  # noqa: F401

    def test_result_fields(self) -> None:
        from training import EvaluationResult
        r = EvaluationResult(
            wins=7, draws=2, losses=1, n_episodes=10,
            win_rate=0.7, draw_rate=0.2, loss_rate=0.1,
        )
        self.assertEqual(r.wins, 7)
        self.assertAlmostEqual(r.win_rate, 0.7)

    def test_evaluate_importable_from_training(self) -> None:
        from training import evaluate  # noqa: F401
        self.assertTrue(callable(evaluate))


# ---------------------------------------------------------------------------
# EloRegistry — accessible from training package
# ---------------------------------------------------------------------------


class TestEloAPI(unittest.TestCase):
    def test_elo_registry_importable(self) -> None:
        from training import EloRegistry, DEFAULT_RATING, BASELINE_RATINGS  # noqa: F401

    def test_baseline_ratings_contains_all_scripted_levels(self) -> None:
        from training import BASELINE_RATINGS
        for level in range(1, 6):
            self.assertIn(f"scripted_l{level}", BASELINE_RATINGS)

    def test_elo_update_roundtrip(self) -> None:
        from training import EloRegistry
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = EloRegistry(path=Path(tmpdir) / "elo.json")
            delta = reg.update("my_agent", "scripted_l3", outcome=0.6, n_games=10)
            self.assertIsInstance(delta, float)
            rating = reg.get_rating("my_agent")
            self.assertGreater(rating, 0.0)


# ---------------------------------------------------------------------------
# PolicyRegistry — accessible from training package
# ---------------------------------------------------------------------------


class TestPolicyRegistryAPI(unittest.TestCase):
    def test_policy_registry_importable(self) -> None:
        from training import PolicyRegistry, Echelon, PolicyEntry  # noqa: F401

    def test_echelon_values(self) -> None:
        from training import Echelon
        self.assertEqual(Echelon.BATTALION.value, "battalion")
        self.assertEqual(Echelon.BRIGADE.value, "brigade")
        self.assertEqual(Echelon.DIVISION.value, "division")

    def test_registry_register_and_list(self) -> None:
        from training import PolicyRegistry, Echelon
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = PolicyRegistry(Path(tmpdir) / "registry.json")
            entry = reg.register(
                echelon=Echelon.BATTALION,
                version="v1",
                path="checkpoints/model.pt",
            )
            entries = reg.list()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].version, "v1")


# ---------------------------------------------------------------------------
# CurriculumScheduler — accessible from training package
# ---------------------------------------------------------------------------


class TestCurriculumAPI(unittest.TestCase):
    def test_curriculum_importable(self) -> None:
        from training import CurriculumScheduler, CurriculumStage  # noqa: F401

    def test_scheduler_stages(self) -> None:
        from training import CurriculumScheduler, CurriculumStage
        scheduler = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)
        self.assertEqual(scheduler.stage, CurriculumStage.STAGE_1V1)

    def test_scheduler_promotion(self) -> None:
        from training import CurriculumScheduler
        scheduler = CurriculumScheduler(promote_threshold=0.5, win_rate_window=4)
        for _ in range(4):
            scheduler.record_episode(win=True)
        self.assertTrue(scheduler.should_promote())
        next_stage = scheduler.promote()
        self.assertIsNotNone(next_stage)


# ---------------------------------------------------------------------------
# CheckpointManifest — accessible from training package
# ---------------------------------------------------------------------------


class TestArtifactsAPI(unittest.TestCase):
    def test_manifest_importable(self) -> None:
        from training import (  # noqa: F401
            CheckpointManifest,
            checkpoint_name_prefix,
            checkpoint_final_stem,
            checkpoint_best_filename,
            parse_step_from_checkpoint_name,
        )

    def test_checkpoint_name_prefix(self) -> None:
        from training import checkpoint_name_prefix
        prefix = checkpoint_name_prefix(seed=1, curriculum_level=5, enable_v2=True)
        self.assertIn("1", prefix)
        self.assertIn("5", prefix)

    def test_manifest_register_and_read(self) -> None:
        from training import CheckpointManifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = CheckpointManifest(Path(tmpdir) / "manifest.jsonl")
            fake_path = Path(tmpdir) / "model_100_steps.zip"
            fake_path.write_text("fake", encoding="utf-8")
            registered = manifest.register(
                fake_path,
                artifact_type="periodic",
                seed=0,
                curriculum_level=5,
                run_id=None,
                config_hash="abc123",
                step=100,
            )
            self.assertTrue(registered)
            entry = manifest.latest_entry_for_path(fake_path)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["step"], 100)


# ---------------------------------------------------------------------------
# SimEngine — accessible from envs package
# ---------------------------------------------------------------------------


class TestSimEngineAPI(unittest.TestCase):
    def test_sim_engine_importable(self) -> None:
        from envs import SimEngine, EpisodeResult  # noqa: F401

    def test_episode_result_fields(self) -> None:
        from envs import EpisodeResult
        # EpisodeResult is a dataclass — verify it has expected fields
        field_names = {f.name for f in dataclasses.fields(EpisodeResult)}
        self.assertIn("winner", field_names)
        self.assertIn("steps", field_names)


if __name__ == "__main__":
    unittest.main()
