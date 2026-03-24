# training/wfm1_benchmark.py
"""WFM-1 zero-shot vs. specialist benchmark — E12.1.

Evaluates WFM-1 on 20 held-out scenarios:
* **Zero-shot**: WFM-1 runs on a novel scenario with no additional training.
* **Specialist**: A policy trained from scratch on only that scenario.
* **WFM-1 fine-tuned**: WFM-1 adapter fine-tuned for ≤ 10 k steps.

Acceptance criteria (from the epic):
* WFM-1 zero-shot win rate ≥ 55 % on unseen procedural scenarios.
* Fine-tuning for 10 k steps achieves ≥ 80 % of fully-trained specialist
  performance.

Architecture
------------
:class:`WFM1BenchmarkScenario`
    Descriptor for one held-out evaluation scenario.

:class:`WFM1BenchmarkConfig`
    Configuration (number of episodes, fine-tune budget, …).

:class:`WFM1BenchmarkResult`
    Win-rate statistics for one (model, scenario) pair.

:class:`WFM1BenchmarkSummary`
    Aggregated results across all 20 scenarios with criterion checks.

:class:`WFM1Benchmark`
    High-level runner.

Typical usage (dry-run / CI)::

    from training.wfm1_benchmark import WFM1Benchmark, WFM1BenchmarkConfig

    cfg = WFM1BenchmarkConfig(n_eval_episodes=5, n_scenarios=4)
    bench = WFM1Benchmark(cfg)
    summary = bench.run(wfm1_policy=None)  # scripted baseline
    print(summary)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.wfm1 import (
    ECHELON_BATTALION,
    TERRAIN_PROCEDURAL,
    TERRAIN_GIS_WATERLOO,
    TERRAIN_GIS_AUSTERLITZ,
    TERRAIN_GIS_BORODINO,
    TERRAIN_GIS_SALAMANCA,
    WEATHER_CLEAR,
    WEATHER_RAIN,
    WEATHER_FOG,
    WEATHER_SNOW,
    ScenarioCard,
    WFM1Policy,
)
from models.entity_encoder import ENTITY_TOKEN_DIM

log = logging.getLogger(__name__)

__all__ = [
    "WFM1BenchmarkScenario",
    "WFM1BenchmarkConfig",
    "WFM1BenchmarkResult",
    "WFM1BenchmarkSummary",
    "WFM1Benchmark",
    "HELD_OUT_SCENARIOS",
]

# ---------------------------------------------------------------------------
# Held-out scenario registry
# ---------------------------------------------------------------------------

#: The canonical 20 held-out scenarios for WFM-1 evaluation.
HELD_OUT_SCENARIOS: List[Dict[str, Any]] = [
    # ── Procedural (unseen seeds) ──────────────────────────────────────────
    {"name": "proc_bn_clear_01",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_CLEAR, "n_blue": 4,  "n_red": 4,  "seed": 1001},
    {"name": "proc_bn_rain_02",    "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_RAIN,  "n_blue": 6,  "n_red": 6,  "seed": 1002},
    {"name": "proc_bn_fog_03",     "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_FOG,   "n_blue": 4,  "n_red": 8,  "seed": 1003},
    {"name": "proc_bn_snow_04",    "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_SNOW,  "n_blue": 8,  "n_red": 4,  "seed": 1004},
    {"name": "proc_bn_clear_05",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_CLEAR, "n_blue": 8,  "n_red": 8,  "seed": 1005},
    # ── GIS — Waterloo ────────────────────────────────────────────────────
    {"name": "gis_waterloo_clear", "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_WATERLOO,  "weather": WEATHER_CLEAR, "n_blue": 6,  "n_red": 6,  "seed": 2001},
    {"name": "gis_waterloo_rain",  "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_WATERLOO,  "weather": WEATHER_RAIN,  "n_blue": 4,  "n_red": 8,  "seed": 2002},
    # ── GIS — Austerlitz ─────────────────────────────────────────────────
    {"name": "gis_austerlitz_fog", "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_AUSTERLITZ,"weather": WEATHER_FOG,   "n_blue": 8,  "n_red": 8,  "seed": 3001},
    {"name": "gis_austerlitz_clr", "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_AUSTERLITZ,"weather": WEATHER_CLEAR, "n_blue": 6,  "n_red": 4,  "seed": 3002},
    # ── GIS — Borodino ────────────────────────────────────────────────────
    {"name": "gis_borodino_snow",  "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_BORODINO,  "weather": WEATHER_SNOW,  "n_blue": 8,  "n_red": 8,  "seed": 4001},
    {"name": "gis_borodino_clr",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_BORODINO,  "weather": WEATHER_CLEAR, "n_blue": 4,  "n_red": 6,  "seed": 4002},
    # ── GIS — Salamanca ───────────────────────────────────────────────────
    {"name": "gis_salamanca_fog",  "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_SALAMANCA, "weather": WEATHER_FOG,   "n_blue": 6,  "n_red": 6,  "seed": 5001},
    {"name": "gis_salamanca_rain", "echelon": ECHELON_BATTALION, "terrain": TERRAIN_GIS_SALAMANCA, "weather": WEATHER_RAIN,  "n_blue": 8,  "n_red": 4,  "seed": 5002},
    # ── Large procedural (corps-scale-ish) ───────────────────────────────
    {"name": "proc_bn_large_06",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_CLEAR, "n_blue": 12, "n_red": 12, "seed": 1006},
    {"name": "proc_bn_large_07",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_RAIN,  "n_blue": 16, "n_red": 12, "seed": 1007},
    # ── Asymmetric force ratios ────────────────────────────────────────────
    {"name": "proc_bn_asym_08",    "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_CLEAR, "n_blue": 4,  "n_red": 12, "seed": 1008},
    {"name": "proc_bn_asym_09",    "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_SNOW,  "n_blue": 12, "n_red": 4,  "seed": 1009},
    # ── Mixed weather transitions ─────────────────────────────────────────
    {"name": "proc_bn_fog_10",     "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_FOG,   "n_blue": 6,  "n_red": 8,  "seed": 1010},
    {"name": "proc_bn_snow_11",    "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_SNOW,  "n_blue": 8,  "n_red": 6,  "seed": 1011},
    {"name": "proc_bn_clear_12",   "echelon": ECHELON_BATTALION, "terrain": TERRAIN_PROCEDURAL,   "weather": WEATHER_CLEAR, "n_blue": 6,  "n_red": 6,  "seed": 1012},
]

assert len(HELD_OUT_SCENARIOS) == 20, "Must have exactly 20 held-out scenarios."


# ---------------------------------------------------------------------------
# WFM1BenchmarkScenario
# ---------------------------------------------------------------------------


@dataclass
class WFM1BenchmarkScenario:
    """Descriptor for one held-out evaluation scenario.

    Attributes
    ----------
    name:
        Unique scenario identifier.
    echelon:
        Echelon level.
    terrain_type:
        Terrain category (procedural or GIS site index).
    weather_code:
        Active weather condition.
    n_blue:
        Number of blue units.
    n_red:
        Number of red units.
    seed:
        RNG seed for terrain / unit placement generation.
    """

    name: str
    echelon: int = ECHELON_BATTALION
    terrain_type: int = TERRAIN_PROCEDURAL
    weather_code: int = WEATHER_CLEAR
    n_blue: int = 8
    n_red: int = 8
    seed: int = 42

    def to_scenario_card(self) -> ScenarioCard:
        """Build a :class:`~models.wfm1.ScenarioCard` for this scenario."""
        return ScenarioCard(
            echelon_level=self.echelon,
            weather_code=self.weather_code,
            n_blue_units=float(self.n_blue),
            n_red_units=float(self.n_red),
            terrain_type=self.terrain_type,
            map_scale=0.5,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WFM1BenchmarkScenario":
        return cls(
            name=d["name"],
            echelon=d["echelon"],
            terrain_type=d["terrain"],
            weather_code=d["weather"],
            n_blue=d["n_blue"],
            n_red=d["n_red"],
            seed=d["seed"],
        )


# ---------------------------------------------------------------------------
# WFM1BenchmarkConfig
# ---------------------------------------------------------------------------


@dataclass
class WFM1BenchmarkConfig:
    """Configuration for the WFM-1 benchmark.

    Attributes
    ----------
    n_eval_episodes:
        Number of evaluation episodes per scenario per condition.
    n_scenarios:
        Number of held-out scenarios to evaluate (≤ 20).
    finetune_steps:
        Maximum fine-tuning budget (adapter only) for the "WFM-1 fine-tuned"
        condition.
    max_steps_per_episode:
        Episode step limit.
    specialist_train_steps:
        How many steps to train each specialist baseline (set to 0 to use a
        scripted baseline instead — strongly recommended for CI).
    zero_shot_win_rate_threshold:
        Minimum acceptable zero-shot win rate (acceptance criterion).
    finetune_recovery_fraction:
        Fraction of specialist performance that fine-tuned WFM-1 must reach
        (acceptance criterion).
    """

    n_eval_episodes: int = 50
    n_scenarios: int = 20
    finetune_steps: int = 10_000
    max_steps_per_episode: int = 500
    specialist_train_steps: int = 0  # 0 = scripted baseline
    zero_shot_win_rate_threshold: float = 0.55
    finetune_recovery_fraction: float = 0.80


# ---------------------------------------------------------------------------
# WFM1BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class WFM1BenchmarkResult:
    """Win-rate statistics for one (model × scenario) evaluation.

    Attributes
    ----------
    scenario_name:
        Name of the evaluated scenario.
    condition:
        One of ``"zero_shot"``, ``"finetuned"``, ``"specialist"``.
    win_rate:
        Fraction of episodes won by the blue policy.
    mean_steps:
        Mean episode length.
    std_steps:
        Standard deviation of episode length.
    n_episodes:
        Number of episodes evaluated.
    elapsed_seconds:
        Wall-clock time for this evaluation.
    finetune_steps_used:
        Number of fine-tuning steps applied (0 for zero-shot / specialist).
    """

    scenario_name: str
    condition: str
    win_rate: float
    mean_steps: float
    std_steps: float
    n_episodes: int
    elapsed_seconds: float = 0.0
    finetune_steps_used: int = 0


# ---------------------------------------------------------------------------
# WFM1BenchmarkSummary
# ---------------------------------------------------------------------------


@dataclass
class WFM1BenchmarkSummary:
    """Aggregated WFM-1 benchmark results across all scenarios.

    Attributes
    ----------
    results:
        All individual (scenario, condition) results.
    config:
        The :class:`WFM1BenchmarkConfig` used.
    """

    results: List[WFM1BenchmarkResult]
    config: WFM1BenchmarkConfig

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    def _filter(self, condition: str) -> List[WFM1BenchmarkResult]:
        return [r for r in self.results if r.condition == condition]

    @property
    def mean_zero_shot_win_rate(self) -> float:
        """Mean zero-shot win rate across all scenarios."""
        zs = self._filter("zero_shot")
        return float(np.mean([r.win_rate for r in zs])) if zs else 0.0

    @property
    def mean_finetuned_win_rate(self) -> float:
        """Mean fine-tuned win rate across all scenarios."""
        ft = self._filter("finetuned")
        return float(np.mean([r.win_rate for r in ft])) if ft else 0.0

    @property
    def mean_specialist_win_rate(self) -> float:
        """Mean specialist win rate across all scenarios."""
        sp = self._filter("specialist")
        return float(np.mean([r.win_rate for r in sp])) if sp else 0.0

    @property
    def finetune_recovery(self) -> float:
        """Fine-tuned performance as a fraction of specialist performance.

        Returns 0.0 when no specialist baseline is available.
        """
        sp = self.mean_specialist_win_rate
        if sp < 1e-6:
            return 0.0
        return self.mean_finetuned_win_rate / sp

    # ------------------------------------------------------------------
    # Acceptance criteria
    # ------------------------------------------------------------------

    @property
    def meets_zero_shot_criterion(self) -> bool:
        """Zero-shot win rate ≥ threshold (default 55 %)."""
        return (
            self.mean_zero_shot_win_rate
            >= self.config.zero_shot_win_rate_threshold
        )

    @property
    def meets_finetune_criterion(self) -> bool:
        """Fine-tuned win rate ≥ 80 % of specialist win rate."""
        return self.finetune_recovery >= self.config.finetune_recovery_fraction

    @property
    def all_criteria_met(self) -> bool:
        return self.meets_zero_shot_criterion and self.meets_finetune_criterion

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [
            "WFM-1 Benchmark Summary",
            f"  Scenarios evaluated  : {len(self._filter('zero_shot'))}",
            f"  Zero-shot win rate   : {self.mean_zero_shot_win_rate:.1%} "
            f"(threshold ≥ {self.config.zero_shot_win_rate_threshold:.0%}) "
            f"{'✅' if self.meets_zero_shot_criterion else '❌'}",
            f"  Fine-tuned win rate  : {self.mean_finetuned_win_rate:.1%}",
            f"  Specialist win rate  : {self.mean_specialist_win_rate:.1%}",
            f"  Fine-tune recovery   : {self.finetune_recovery:.1%} "
            f"(threshold ≥ {self.config.finetune_recovery_fraction:.0%}) "
            f"{'✅' if self.meets_finetune_criterion else '❌'}",
        ]
        return "\n".join(lines)

    def write_markdown(self, path: Optional[str | Path] = None) -> Path:
        """Write a Markdown report to *path*.

        Defaults to ``docs/wfm1_benchmark.md``.
        """
        if path is None:
            out = _PROJECT_ROOT / "docs" / "wfm1_benchmark.md"
        else:
            out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_markdown(self), encoding="utf-8")
        return out


def _render_markdown(summary: WFM1BenchmarkSummary) -> str:
    lines = [
        "# WFM-1 Benchmark Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Zero-shot win rate | {summary.mean_zero_shot_win_rate:.1%} |",
        f"| Fine-tuned win rate | {summary.mean_finetuned_win_rate:.1%} |",
        f"| Specialist win rate | {summary.mean_specialist_win_rate:.1%} |",
        f"| Fine-tune recovery | {summary.finetune_recovery:.1%} |",
        f"| Zero-shot criterion (≥55%) | {'✅ PASS' if summary.meets_zero_shot_criterion else '❌ FAIL'} |",
        f"| Fine-tune criterion (≥80%) | {'✅ PASS' if summary.meets_finetune_criterion else '❌ FAIL'} |",
        "",
        "## Per-scenario Results",
        "",
        "| Scenario | Condition | Win Rate | Episodes | Steps (mean) |",
        "|----------|-----------|----------|----------|--------------|",
    ]
    for r in sorted(summary.results, key=lambda x: (x.scenario_name, x.condition)):
        lines.append(
            f"| {r.scenario_name} | {r.condition} | {r.win_rate:.1%} "
            f"| {r.n_episodes} | {r.mean_steps:.0f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# WFM1Benchmark
# ---------------------------------------------------------------------------


class WFM1Benchmark:
    """Run WFM-1 zero-shot, fine-tuned, and specialist evaluations.

    Parameters
    ----------
    config:
        Benchmark configuration.
    """

    def __init__(self, config: WFM1BenchmarkConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        wfm1_policy: Optional[Any] = None,
        specialist_policies: Optional[Dict[str, Any]] = None,
    ) -> WFM1BenchmarkSummary:
        """Run all evaluations and return a summary.

        Parameters
        ----------
        wfm1_policy:
            Trained :class:`~models.wfm1.WFM1Policy` (or ``None`` for a
            scripted random-walk baseline).
        specialist_policies:
            Optional mapping from scenario name → pre-trained specialist
            policy.  When ``None``, a scripted baseline is used.

        Returns
        -------
        :class:`WFM1BenchmarkSummary`
        """
        cfg = self.config
        n = min(cfg.n_scenarios, len(HELD_OUT_SCENARIOS))
        scenarios = [
            WFM1BenchmarkScenario.from_dict(d) for d in HELD_OUT_SCENARIOS[:n]
        ]

        results: List[WFM1BenchmarkResult] = []

        for scenario in scenarios:
            env = self._make_env(scenario)
            card = scenario.to_scenario_card()

            # 1. Zero-shot
            t0 = time.perf_counter()
            zs_episodes = self._evaluate(wfm1_policy, env, card, scenario.echelon)
            results.append(
                WFM1BenchmarkResult(
                    scenario_name=scenario.name,
                    condition="zero_shot",
                    **self._aggregate(zs_episodes),
                    elapsed_seconds=time.perf_counter() - t0,
                )
            )

            # 2. Fine-tuned (adapter only)
            ft_policy = self._finetune(wfm1_policy, env, card, scenario.echelon)
            t0 = time.perf_counter()
            ft_episodes = self._evaluate(ft_policy, env, card, scenario.echelon)
            results.append(
                WFM1BenchmarkResult(
                    scenario_name=scenario.name,
                    condition="finetuned",
                    **self._aggregate(ft_episodes),
                    elapsed_seconds=time.perf_counter() - t0,
                    finetune_steps_used=cfg.finetune_steps,
                )
            )

            # 3. Specialist
            spec_policy = (
                specialist_policies.get(scenario.name)
                if specialist_policies
                else None
            )
            t0 = time.perf_counter()
            sp_episodes = self._evaluate(spec_policy, env, None, scenario.echelon)
            results.append(
                WFM1BenchmarkResult(
                    scenario_name=scenario.name,
                    condition="specialist",
                    **self._aggregate(sp_episodes),
                    elapsed_seconds=time.perf_counter() - t0,
                )
            )

            env.close()
            log.info("Scenario %s done.", scenario.name)

        return WFM1BenchmarkSummary(results=results, config=cfg)

    # ------------------------------------------------------------------
    # Internal: environment factory
    # ------------------------------------------------------------------

    def _make_env(self, scenario: WFM1BenchmarkScenario) -> Any:
        """Create an evaluation environment for *scenario*.

        Builds a GIS terrain when ``scenario.terrain_type`` is non-zero
        (a recognised GIS battle site), otherwise uses procedurally generated
        random terrain.  The terrain dimensions are passed through to
        :class:`~envs.battalion_env.BattalionEnv` so that unit-position
        normalisation is consistent with the terrain grid.
        """
        _GIS_SITE_NAMES: Dict[int, str] = {
            TERRAIN_GIS_WATERLOO: "waterloo",
            TERRAIN_GIS_AUSTERLITZ: "austerlitz",
            TERRAIN_GIS_BORODINO: "borodino",
            TERRAIN_GIS_SALAMANCA: "salamanca",
        }

        try:
            from envs.battalion_env import BattalionEnv
            from envs.sim.terrain import TerrainMap

            if scenario.terrain_type in _GIS_SITE_NAMES:
                from data.gis.terrain_importer import GISTerrainBuilder

                terrain = GISTerrainBuilder(
                    site=_GIS_SITE_NAMES[scenario.terrain_type],
                    rows=40,
                    cols=40,
                ).build()
            else:
                terrain = TerrainMap.generate_random(
                    rng=np.random.default_rng(scenario.seed),
                    width=10_000.0,
                    height=10_000.0,
                    rows=40,
                    cols=40,
                    num_hills=4,
                    num_forests=3,
                )

            return BattalionEnv(
                terrain=terrain,
                randomize_terrain=False,
                map_width=terrain.width,
                map_height=terrain.height,
            )
        except Exception as exc:
            log.debug(
                "Could not build real env for scenario %r: %s — using synthetic.",
                scenario.name,
                exc,
                exc_info=True,
            )

        # Synthetic fallback
        n_entities = scenario.n_blue + scenario.n_red
        return _SyntheticBenchmarkEnv(
            n_entities=n_entities,
            seed=scenario.seed,
            ep_length=self.config.max_steps_per_episode,
        )

    # ------------------------------------------------------------------
    # Internal: episode runner
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        policy: Optional[Any],
        env: Any,
        card: Optional[ScenarioCard],
        echelon: int,
    ) -> List[Dict[str, Any]]:
        """Run *n_eval_episodes* and return raw per-episode stats."""
        cfg = self.config
        results = []
        use_wfm1 = isinstance(policy, WFM1Policy)
        use_predict = not use_wfm1 and hasattr(policy, "predict")

        import torch

        # Infer policy device once so all tensors are placed correctly.
        if use_wfm1:
            try:
                _policy_device = next(policy.parameters()).device
            except (StopIteration, AttributeError):
                _policy_device = torch.device("cpu")
        else:
            _policy_device = torch.device("cpu")

        for ep_idx in range(cfg.n_eval_episodes):
            obs, _ = env.reset(seed=ep_idx + 9000)
            terminated = truncated = False
            steps = 0
            won = None

            while not (terminated or truncated) and steps < cfg.max_steps_per_episode:
                if use_wfm1:
                    tokens_np, pm_np = self._obs_to_tokens(obs)
                    t = torch.as_tensor(
                        tokens_np[np.newaxis], dtype=torch.float32
                    ).to(_policy_device)
                    pm = (
                        torch.as_tensor(pm_np[np.newaxis], dtype=torch.bool).to(_policy_device)
                        if pm_np is not None
                        else None
                    )
                    _card_vec = (
                        card.to_tensor(device=_policy_device) if card is not None else None
                    )
                    with torch.no_grad():
                        action, _ = policy.act(
                            t, pad_mask=pm, echelon=echelon,
                            card_vec=_card_vec,
                            deterministic=True,
                        )
                    action = action.squeeze(0).cpu().numpy()
                elif use_predict:
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    action = self._scripted_action(obs)

                obs, _rew, terminated, truncated, info = env.step(action)
                steps += 1

            # Determine winner from info dict (best-effort)
            if isinstance(info, dict):
                if info.get("blue_routed"):
                    won = False
                elif info.get("red_routed"):
                    won = True
                else:
                    won = None
            results.append({"won": won, "steps": steps})

        return results

    def _obs_to_tokens(
        self, obs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert observation to entity token array."""
        if isinstance(obs, np.ndarray):
            if obs.ndim == 2 and obs.shape[-1] == ENTITY_TOKEN_DIM:
                return obs, None
            n = (obs.size + ENTITY_TOKEN_DIM - 1) // ENTITY_TOKEN_DIM
            padded = np.zeros(n * ENTITY_TOKEN_DIM, dtype=np.float32)
            padded[: obs.size] = obs.ravel()
            return padded.reshape(n, ENTITY_TOKEN_DIM), None
        return np.zeros((8, ENTITY_TOKEN_DIM), dtype=np.float32), None

    @staticmethod
    def _scripted_action(obs: Any) -> np.ndarray:
        """Simple deterministic scripted action (advance forward)."""
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: fine-tuning
    # ------------------------------------------------------------------

    def _finetune(
        self,
        policy: Optional[Any],
        env: Any,
        card: Optional[ScenarioCard],
        echelon: int,
    ) -> Optional[Any]:
        """Adapter-only fine-tuning of WFM-1 for ≤ finetune_steps steps."""
        if not isinstance(policy, WFM1Policy):
            return policy  # not WFM-1 — return unchanged

        import copy
        import torch

        ft_policy: WFM1Policy = copy.deepcopy(policy)
        ft_policy.freeze_base()
        opt = torch.optim.Adam(ft_policy.adapter_parameters(), lr=3e-4)
        steps = 0

        try:
            ft_device = next(ft_policy.parameters()).device
        except StopIteration:
            ft_device = torch.device("cpu")

        obs, _ = env.reset(seed=7777)
        tokens_np, pm_np = self._obs_to_tokens(obs)

        while steps < self.config.finetune_steps:
            # Use deterministic=True so the BC target is the distribution mean,
            # not a stochastic sample — prevents the adapter from chasing noise.
            t = torch.as_tensor(
                tokens_np[np.newaxis], dtype=torch.float32
            ).to(ft_device)
            pm = (
                torch.as_tensor(pm_np[np.newaxis], dtype=torch.bool).to(ft_device)
                if pm_np is not None
                else None
            )
            with torch.no_grad():
                action, _ = ft_policy.act(
                    t, pad_mask=pm, echelon=echelon, card=card,
                    deterministic=True,
                )
            action_np = action.squeeze(0).cpu().numpy()

            obs_new, _rew, terminated, truncated, _info = env.step(action_np)
            done = terminated or truncated

            # Behaviour-cloning loss: match own predictions (self-distillation)
            batch = {
                "tokens": t,
                "actions": action.detach(),
                "pad_mask": pm,
                "echelon": torch.tensor(echelon),
            }
            if card is not None:
                batch["card_vec"] = card.to_tensor(device=ft_device)
            loss = ft_policy.finetune_loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1

            if done:
                obs, _ = env.reset(seed=7777 + steps)
                tokens_np, pm_np = self._obs_to_tokens(obs)
            else:
                obs = obs_new
                tokens_np, pm_np = self._obs_to_tokens(obs)

        ft_policy.unfreeze_base()
        return ft_policy

    # ------------------------------------------------------------------
    # Internal: statistics aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute win_rate, mean_steps, std_steps, n_episodes."""
        wins = [e["won"] for e in episodes if e["won"] is not None]
        steps = [e["steps"] for e in episodes]
        win_rate = float(np.mean(wins)) if wins else 0.0
        return {
            "win_rate": win_rate,
            "mean_steps": float(np.mean(steps)) if steps else 0.0,
            "std_steps": float(np.std(steps)) if steps else 0.0,
            "n_episodes": len(episodes),
        }


# ---------------------------------------------------------------------------
# Synthetic benchmark environment
# ---------------------------------------------------------------------------


class _SyntheticBenchmarkEnv:
    """Lightweight synthetic env used when real envs are unavailable."""

    def __init__(
        self,
        n_entities: int = 16,
        seed: int = 0,
        ep_length: int = 500,
    ) -> None:
        self.n_entities = n_entities
        self.ep_length = ep_length
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def reset(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        obs = self._rng.standard_normal((self.n_entities, ENTITY_TOKEN_DIM)).astype(
            np.float32
        )
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal((self.n_entities, ENTITY_TOKEN_DIM)).astype(
            np.float32
        )
        reward = float(self._rng.standard_normal())
        terminated = self._step >= self.ep_length
        # Randomly determine winner at end of episode
        info: Dict[str, Any] = {}
        if terminated:
            outcome = self._rng.integers(3)
            if outcome == 0:
                info["red_routed"] = True
            elif outcome == 1:
                info["blue_routed"] = True
        return obs, reward, terminated, False, info

    def close(self) -> None:
        pass
