# analysis/saliency.py
"""Strategy Explainability — saliency maps and feature importance (Epic E5.3).

Provides three complementary techniques for interpreting trained policies:

1. **Gradient saliency** — back-propagates the action magnitude through the
   policy network and uses the absolute input gradient as a proxy for feature
   importance.  Works with any differentiable PyTorch model, including
   :class:`~models.mlp_policy.BattalionMlpPolicy` (SB3 ActorCriticPolicy).

2. **Integrated gradients** — accumulates gradients along a straight-line
   path from a baseline (zero observation) to the real observation.  Produces
   attribution scores that sum to the model output difference, giving a
   theoretically grounded importance estimate.

3. **SHAP kernel / linear approximation** — model-agnostic, treats the policy
   as a black box and uses Shapley values to rank observation dimensions.
   Falls back to a lightweight permutation-based approximation when the
   ``shap`` package is not installed.

Typical usage::

    from analysis.saliency import SaliencyAnalyzer
    from envs.battalion_env import BattalionEnv
    from stable_baselines3 import PPO
    from models.mlp_policy import BattalionMlpPolicy

    env = BattalionEnv()
    model = PPO(BattalionMlpPolicy, env)
    obs, _ = env.reset(seed=0)

    analyzer = SaliencyAnalyzer(model)
    saliency   = analyzer.gradient_saliency(obs)
    ig_scores  = analyzer.integrated_gradients(obs)
    shap_vals  = analyzer.shap_importance(obs)

    fig = analyzer.plot_saliency(saliency, title="Gradient saliency")
    fig.savefig("saliency.png")

All public functions also accept a **batch** of observations (shape
``(N, obs_dim)``), returning a result of the same batch shape so they can be
used on entire episode trajectories.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "OBSERVATION_FEATURES",
    "SaliencyAnalyzer",
    "compute_gradient_saliency",
    "compute_integrated_gradients",
    "compute_shap_importance",
    "plot_saliency_map",
    "plot_feature_importance",
]

# ---------------------------------------------------------------------------
# Observation feature names (12-dimensional BattalionEnv observation)
# ---------------------------------------------------------------------------

#: Human-readable names for each of the 12 BattalionEnv observation dimensions.
#: Index order matches :meth:`~envs.battalion_env.BattalionEnv._get_obs`.
OBSERVATION_FEATURES: Tuple[str, ...] = (
    "blue_x",        # [0]  Blue battalion x-position (normalised 0–1)
    "blue_y",        # [1]  Blue battalion y-position (normalised 0–1)
    "blue_cos_θ",    # [2]  cos(Blue heading angle)
    "blue_sin_θ",    # [3]  sin(Blue heading angle)
    "blue_strength", # [4]  Blue strength / effective fighting power (0–1)
    "blue_morale",   # [5]  Blue morale (0–1)
    "dist_norm",     # [6]  Normalised Blue-to-Red distance (0–1)
    "cos_bearing",   # [7]  cos(bearing to Red)
    "sin_bearing",   # [8]  sin(bearing to Red)
    "red_strength",  # [9]  Red strength (0–1)
    "red_morale",    # [10] Red morale (0–1)
    "step_norm",     # [11] Normalised episode progress (0–1)
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_mlp_network(policy: Any) -> nn.Module:
    """Return the forward-callable network from a policy object.

    Supports:
    * SB3 ``ActorCriticPolicy`` (``BattalionMlpPolicy``) — returns the
      ``mlp_extractor`` + action-mean head wrapped in a tiny sequential.
    * SB3 ``PPO`` model — delegates to its ``policy`` attribute.
    * Plain ``nn.Module`` — returned as-is.

    The returned module accepts a float32 tensor of shape ``(batch, obs_dim)``
    and produces a scalar (or vector) output suitable for back-propagation.
    """
    # Unwrap SB3 PPO model
    if hasattr(policy, "policy") and isinstance(policy.policy, nn.Module):
        policy = policy.policy

    # SB3 ActorCriticPolicy
    if hasattr(policy, "mlp_extractor") and hasattr(policy, "action_net"):
        class _SB3MeanExtractor(nn.Module):
            def __init__(self, mlp_extractor: nn.Module, action_net: nn.Module) -> None:
                super().__init__()
                self._mlp = mlp_extractor
                self._action = action_net

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                # mlp_extractor returns (latent_pi, latent_vf)
                latent_pi, _ = self._mlp(x)
                return self._action(latent_pi)

        return _SB3MeanExtractor(policy.mlp_extractor, policy.action_net)

    # MAPPOPolicy / plain nn.Module
    if isinstance(policy, nn.Module):
        return policy

    raise TypeError(
        f"Unsupported policy type: {type(policy)}.  "
        "Pass an SB3 PPO model, ActorCriticPolicy, or plain nn.Module."
    )


def _to_tensor(obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert observation to a float32 torch tensor with batch dimension."""
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs.astype(np.float32))
    else:
        obs = obs.float()
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    return obs


# ---------------------------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------------------------

def compute_gradient_saliency(
    policy: Any,
    obs: Union[np.ndarray, torch.Tensor],
    *,
    reduce: str = "mean_abs",
) -> np.ndarray:
    """Compute gradient-based saliency scores for each observation dimension.

    Back-propagates through the policy network and uses the absolute gradient
    of the summed action output with respect to each input dimension as an
    importance proxy.

    Parameters
    ----------
    policy:
        Trained policy.  Accepts SB3 ``PPO``, ``ActorCriticPolicy``, or a
        plain ``nn.Module``.
    obs:
        Observation array of shape ``(obs_dim,)`` or ``(N, obs_dim)``.
    reduce:
        How to aggregate across the action dimension and batch:
        - ``"mean_abs"`` (default) — mean of absolute gradients.
        - ``"max_abs"`` — max of absolute gradients.
        - ``"sum_abs"`` — sum of absolute gradients.

    Returns
    -------
    np.ndarray
        Saliency scores of shape ``(obs_dim,)`` (after batch aggregation),
        or ``(N, obs_dim)`` if ``reduce="none"``.
    """
    net = _extract_mlp_network(policy)
    net.eval()

    x = _to_tensor(obs)
    x = x.requires_grad_(True)

    output = net(x)
    # Scalar target: sum of absolute action means over all actions and batch
    target = output.sum()
    target.backward()

    grad = x.grad.detach().numpy()  # (N, obs_dim)

    if reduce == "none":
        return np.abs(grad)
    elif reduce == "mean_abs":
        return np.abs(grad).mean(axis=0)
    elif reduce == "max_abs":
        return np.abs(grad).max(axis=0)
    elif reduce == "sum_abs":
        return np.abs(grad).sum(axis=0)
    else:
        raise ValueError(f"Unknown reduce mode: {reduce!r}")


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------

def compute_integrated_gradients(
    policy: Any,
    obs: Union[np.ndarray, torch.Tensor],
    *,
    baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Compute integrated gradient attributions.

    Follows the method of Sundararajan et al. (2017): accumulate gradients
    along the straight-line path from ``baseline`` to ``obs``, then multiply
    element-wise by ``(obs - baseline)``.  The result satisfies the
    *completeness* axiom: attributions sum to ``f(obs) - f(baseline)``.

    Parameters
    ----------
    policy:
        Trained policy (same supported types as :func:`compute_gradient_saliency`).
    obs:
        Single observation of shape ``(obs_dim,)`` or batch ``(N, obs_dim)``.
    baseline:
        Reference point.  Defaults to the all-zeros observation.
    n_steps:
        Number of interpolation steps along the path (higher → more accurate).

    Returns
    -------
    np.ndarray
        Attribution scores of shape ``(obs_dim,)`` (mean over batch if N>1).
    """
    net = _extract_mlp_network(policy)
    net.eval()

    x = _to_tensor(obs)  # (N, obs_dim)
    N, obs_dim = x.shape

    if baseline is None:
        base = torch.zeros_like(x)
    else:
        base = _to_tensor(baseline)
        if base.shape != x.shape:
            base = base.expand_as(x)

    # Accumulate gradients along the interpolation path
    accumulated_grads = torch.zeros_like(x)

    for alpha in np.linspace(0.0, 1.0, n_steps):
        x_interp = (base + alpha * (x - base)).detach().requires_grad_(True)
        output = net(x_interp).sum()
        output.backward()
        accumulated_grads += x_interp.grad.detach()

    # IG formula: (obs - baseline) * mean_gradient
    ig = ((x - base) * accumulated_grads / n_steps).detach().numpy()  # (N, obs_dim)
    return ig.mean(axis=0)  # (obs_dim,)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_importance(
    policy: Any,
    obs: Union[np.ndarray, torch.Tensor],
    *,
    background: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_samples: int = 100,
) -> np.ndarray:
    """Compute SHAP-based feature importance scores.

    Attempts to use the ``shap`` library (GradientExplainer for differentiable
    models).  When ``shap`` is not installed or the model is not compatible,
    falls back to a lightweight **permutation importance** approximation that
    estimates each feature's marginal contribution by masking it with the
    background mean.

    Parameters
    ----------
    policy:
        Trained policy.
    obs:
        Observations of shape ``(obs_dim,)`` or ``(N, obs_dim)``.
    background:
        Background dataset for SHAP / permutation baseline.  Defaults to
        the all-zeros observation (single sample).
    n_samples:
        Number of samples used in the permutation fallback.

    Returns
    -------
    np.ndarray
        Absolute SHAP values / permutation importances of shape ``(obs_dim,)``.
    """
    net = _extract_mlp_network(policy)
    net.eval()

    x_np = _to_tensor(obs).detach().numpy()  # (N, obs_dim)

    if background is None:
        bg_np = np.zeros((1, x_np.shape[1]), dtype=np.float32)
    else:
        bg_np = _to_tensor(background).detach().numpy()

    # ── Try shap library ──────────────────────────────────────────────────
    try:
        import shap  # type: ignore

        bg_tensor = torch.from_numpy(bg_np)
        explainer = shap.GradientExplainer(net, bg_tensor)
        shap_values = explainer.shap_values(torch.from_numpy(x_np))
        # shap_values may be a list (one per output) or an array
        if isinstance(shap_values, list):
            shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_arr = np.abs(np.array(shap_values))
        return shap_arr.mean(axis=0)  # (obs_dim,)

    except Exception:  # noqa: BLE001
        pass  # fall through to permutation fallback

    # ── Permutation-based fallback ────────────────────────────────────────
    return _permutation_importance(net, x_np, bg_np, n_samples=n_samples)


def _permutation_importance(
    net: nn.Module,
    x_np: np.ndarray,
    bg_np: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    """Lightweight permutation importance fallback.

    For each feature, replaces it with the background mean and measures the
    absolute change in the summed output.  Higher change → more important.
    """
    obs_dim = x_np.shape[1]
    bg_mean = bg_np.mean(axis=0)  # (obs_dim,)

    def _forward(arr: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.from_numpy(arr.astype(np.float32))
            return float(net(t).abs().sum().item())

    base_score = _forward(x_np)
    importances = np.zeros(obs_dim, dtype=np.float32)

    for feat_idx in range(obs_dim):
        x_masked = x_np.copy()
        x_masked[:, feat_idx] = bg_mean[feat_idx]
        masked_score = _forward(x_masked)
        importances[feat_idx] = abs(base_score - masked_score)

    return importances


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_saliency_map(
    saliency: np.ndarray,
    *,
    feature_names: Optional[Tuple[str, ...]] = None,
    title: str = "Gradient Saliency",
    normalise: bool = True,
    figsize: Tuple[float, float] = (9.0, 4.0),
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a horizontal bar chart of saliency scores.

    Parameters
    ----------
    saliency:
        1-D array of shape ``(obs_dim,)`` with non-negative saliency scores.
    feature_names:
        Feature labels for each dimension.  Defaults to
        :data:`OBSERVATION_FEATURES`.
    title:
        Plot title string.
    normalise:
        When ``True`` (default), divide by the maximum value so all bars lie
        in [0, 1].
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = OBSERVATION_FEATURES

    scores = np.array(saliency, dtype=np.float64)
    if normalise and scores.max() > 0:
        scores = scores / scores.max()

    n = len(scores)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y, scores, color="steelblue", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names[:n], fontsize=9)
    ax.set_xlabel("Normalised saliency" if normalise else "Saliency")
    ax.set_title(title)
    ax.invert_yaxis()  # highest at top

    # Annotate values
    for bar, val in zip(bars, scores):
        ax.text(
            min(val + 0.01, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color="black",
        )

    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    *,
    feature_names: Optional[Tuple[str, ...]] = None,
    title: str = "Feature Importance",
    top_k: int = 12,
    figsize: Tuple[float, float] = (9.0, 4.5),
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a horizontal bar chart of feature importances, sorted by value.

    Parameters
    ----------
    importances:
        1-D importance array of shape ``(obs_dim,)``.
    feature_names:
        Feature labels.  Defaults to :data:`OBSERVATION_FEATURES`.
    title:
        Plot title.
    top_k:
        Show at most this many features (sorted descending).
    figsize:
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = OBSERVATION_FEATURES

    scores = np.array(importances, dtype=np.float64)
    n = min(top_k, len(scores))

    # Sort descending
    order = np.argsort(scores)[::-1][:n]
    sorted_scores = scores[order]
    sorted_labels = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    bars = ax.barh(y, sorted_scores, color="darkorange", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel("Importance score")
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, val in zip(bars, sorted_scores):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=7,
            color="black",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SaliencyAnalyzer — convenience facade
# ---------------------------------------------------------------------------

class SaliencyAnalyzer:
    """Convenience wrapper that bundles all explainability methods.

    Parameters
    ----------
    policy:
        Trained policy.  Accepts an SB3 ``PPO`` model, an
        ``ActorCriticPolicy``, a :class:`~models.mappo_policy.MAPPOPolicy`,
        or any plain ``nn.Module``.
    feature_names:
        Override the default :data:`OBSERVATION_FEATURES` labels.

    Examples
    --------
    ::

        analyzer = SaliencyAnalyzer(ppo_model)
        obs, _ = env.reset(seed=0)

        sal  = analyzer.gradient_saliency(obs)
        ig   = analyzer.integrated_gradients(obs)
        shap = analyzer.shap_importance(obs)

        print(analyzer.top_features(sal, k=3))
        fig = analyzer.plot_saliency(sal)
    """

    def __init__(
        self,
        policy: Any,
        feature_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._policy = policy
        self.feature_names: Tuple[str, ...] = (
            feature_names if feature_names is not None else OBSERVATION_FEATURES
        )

    # ── Core methods ──────────────────────────────────────────────────────

    def gradient_saliency(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        *,
        reduce: str = "mean_abs",
    ) -> np.ndarray:
        """Return gradient saliency scores.  See :func:`compute_gradient_saliency`."""
        return compute_gradient_saliency(self._policy, obs, reduce=reduce)

    def integrated_gradients(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        *,
        baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_steps: int = 50,
    ) -> np.ndarray:
        """Return integrated gradient attributions.  See :func:`compute_integrated_gradients`."""
        return compute_integrated_gradients(
            self._policy, obs, baseline=baseline, n_steps=n_steps
        )

    def shap_importance(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        *,
        background: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_samples: int = 100,
    ) -> np.ndarray:
        """Return SHAP feature importance.  See :func:`compute_shap_importance`."""
        return compute_shap_importance(
            self._policy, obs, background=background, n_samples=n_samples
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    def top_features(
        self,
        scores: np.ndarray,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return the top-*k* feature names and their scores, sorted descending.

        Parameters
        ----------
        scores:
            1-D importance / saliency array.
        k:
            Number of top features to return.

        Returns
        -------
        list of (feature_name, score) tuples
        """
        order = np.argsort(scores)[::-1][:k]
        return [(self.feature_names[i], float(scores[i])) for i in order]

    def summary(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        *,
        n_steps: int = 20,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Compute all three importance metrics and return them as a dict.

        Keys: ``"gradient_saliency"``, ``"integrated_gradients"``,
        ``"shap_importance"``.
        """
        return {
            "gradient_saliency": self.gradient_saliency(obs),
            "integrated_gradients": self.integrated_gradients(obs, n_steps=n_steps),
            "shap_importance": self.shap_importance(obs, n_samples=n_samples),
        }

    # ── Plot helpers ──────────────────────────────────────────────────────

    def plot_saliency(
        self,
        saliency: Optional[np.ndarray] = None,
        obs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        title: str = "Gradient Saliency",
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
        """Plot saliency map.

        Either pass pre-computed ``saliency`` scores, or pass ``obs`` directly
        to compute them automatically.
        """
        if saliency is None:
            if obs is None:
                raise ValueError("Either saliency or obs must be provided.")
            saliency = self.gradient_saliency(obs)
        return plot_saliency_map(
            saliency,
            feature_names=self.feature_names,
            title=title,
            **kwargs,
        )

    def plot_importance(
        self,
        importances: Optional[np.ndarray] = None,
        obs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        title: str = "Feature Importance (SHAP)",
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
        """Plot feature importance bar chart.

        Either pass pre-computed ``importances`` scores, or pass ``obs``
        directly to compute SHAP importance automatically.
        """
        if importances is None:
            if obs is None:
                raise ValueError("Either importances or obs must be provided.")
            importances = self.shap_importance(obs)
        return plot_feature_importance(
            importances,
            feature_names=self.feature_names,
            title=title,
            **kwargs,
        )
