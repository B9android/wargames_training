# SPDX-License-Identifier: MIT
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

:func:`compute_gradient_saliency` accepts a batch of observations (shape
``(N, obs_dim)``).  With the default reduction it returns a single importance
vector of shape ``(obs_dim,)``; pass ``reduce="none"`` to get per-sample
saliency of shape ``(N, obs_dim)``.  :func:`compute_integrated_gradients` and
:func:`compute_shap_importance` always aggregate over the batch and return a
single vector of shape ``(obs_dim,)``.
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
# Observation feature names (17-dimensional BattalionEnv base observation)
# ---------------------------------------------------------------------------

#: Human-readable names for each of the 17 BattalionEnv base observation
#: dimensions (formations, logistics, and weather all disabled).
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
    "blue_elev",     # [12] Blue battalion terrain elevation (normalised 0–1)
    "blue_cover",    # [13] Blue battalion terrain cover (0–1)
    "red_elev",      # [14] Red battalion terrain elevation (normalised 0–1)
    "red_cover",     # [15] Red battalion terrain cover (0–1)
    "los",           # [16] Line-of-sight flag (0 blocked, 1 clear)
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
            def __init__(self, policy_module: Any) -> None:
                super().__init__()
                # Keep a reference to the full policy so we can mirror its
                # actual forward path, including feature extraction.
                self._policy = policy_module

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                # Mirror SB3 ActorCriticPolicy forward:
                #   features = extract_features(obs)
                #   latent_pi, latent_vf = mlp_extractor(features)
                #   action_mean = action_net(latent_pi)
                if hasattr(self._policy, "extract_features"):
                    features = self._policy.extract_features(x)
                elif hasattr(self._policy, "features_extractor"):
                    # Fallback in case only the features_extractor module is exposed.
                    features = self._policy.features_extractor(x)
                else:
                    # If no explicit feature extractor is present, assume x is
                    # already in the correct feature space.
                    features = x

                latent_pi, _ = self._policy.mlp_extractor(features)
                return self._policy.action_net(latent_pi)

        return _SB3MeanExtractor(policy)

    # MAPPOPolicy / plain nn.Module
    if isinstance(policy, nn.Module):
        return policy

    raise TypeError(
        f"Unsupported policy type: {type(policy)}.  "
        "Pass an SB3 PPO model, ActorCriticPolicy, or plain nn.Module."
    )


def _to_tensor(
    obs: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert observation to a float32 torch tensor with batch dimension.

    Parameters
    ----------
    obs:
        Observation as a NumPy array or torch tensor.  A missing batch
        dimension (1-D input) will be added automatically.
    device:
        Optional device to move the resulting tensor to.  If ``None``,
        NumPy inputs are converted on CPU and existing tensors keep their
        current device.
    """
    if isinstance(obs, np.ndarray):
        tensor = torch.from_numpy(obs.astype(np.float32))
        if device is not None:
            tensor = tensor.to(device)
    else:
        tensor = obs.float()
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


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

    device = next(net.parameters(), torch.tensor(0)).device
    x = _to_tensor(obs, device=device)
    x = x.requires_grad_(True)

    output = net(x)
    # Scalar target: sum of absolute action means over all actions and batch
    target = output.sum()
    target.backward()

    grad = x.grad.detach().cpu().numpy()  # (N, obs_dim)

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
        Must be >= 1.

    Returns
    -------
    np.ndarray
        Attribution scores of shape ``(obs_dim,)`` (mean over batch if N>1).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    net = _extract_mlp_network(policy)
    net.eval()

    device = next(net.parameters(), torch.tensor(0)).device
    x = _to_tensor(obs, device=device)  # (N, obs_dim)
    N, obs_dim = x.shape

    if baseline is None:
        base = torch.zeros_like(x)
    else:
        base = _to_tensor(baseline, device=device)
        if base.shape != x.shape:
            base = base.expand_as(x)

    # Accumulate gradients along the interpolation path.
    # Use torch.autograd.grad so that model parameter .grad buffers are never
    # touched — this is safe to call during/after training without side effects.
    accumulated_grads = torch.zeros_like(x)

    for alpha in np.linspace(0.0, 1.0, n_steps):
        x_interp = (base + alpha * (x - base)).detach().requires_grad_(True)
        output = net(x_interp).sum()
        (grad,) = torch.autograd.grad(output, x_interp)
        accumulated_grads = accumulated_grads + grad.detach()

    # IG formula: (obs - baseline) * mean_gradient
    ig = ((x - base) * accumulated_grads / n_steps).detach().cpu().numpy()  # (N, obs_dim)
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

    device = next(net.parameters(), torch.tensor(0)).device
    x_np = _to_tensor(obs, device=device).detach().cpu().numpy()  # (N, obs_dim)

    if background is None:
        bg_np = np.zeros((1, x_np.shape[1]), dtype=np.float32)
    else:
        bg_np = _to_tensor(background, device=device).detach().cpu().numpy()

    # ── Try shap library ──────────────────────────────────────────────────
    shap = None
    try:
        import shap as _shap  # type: ignore
        shap = _shap
    except ImportError:
        pass  # fall through to permutation fallback without a warning

    if shap is not None:
        try:
            bg_tensor = _to_tensor(bg_np, device=device)
            explainer = shap.GradientExplainer(net, bg_tensor)
            shap_values = explainer.shap_values(_to_tensor(x_np, device=device))
            # shap_values may be a list (one per output) or an array
            if isinstance(shap_values, list):
                shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_arr = np.abs(np.array(shap_values))
            return shap_arr.mean(axis=0)  # (obs_dim,)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Falling back to permutation importance because SHAP computation "
                f"failed with: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Permutation-based fallback ────────────────────────────────────────
    return _permutation_importance(net, x_np, bg_np, n_samples=n_samples)


def _permutation_importance(
    net: nn.Module,
    x_np: np.ndarray,
    bg_np: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    """Lightweight permutation importance fallback.

    For each feature, replaces it with values drawn from the background
    distribution (up to ``n_samples`` background rows, sampled with
    replacement) and measures the absolute change in the summed output.
    Higher change → more important feature.

    Parameters
    ----------
    net:
        Forward-callable network.
    x_np:
        Observations of shape ``(N, obs_dim)``.
    bg_np:
        Background dataset of shape ``(M, obs_dim)`` used as the replacement
        distribution.  When ``M < n_samples``, all background rows are used.
    n_samples:
        Maximum number of background samples to use for the permutation.
        Higher values give a more stable estimate at the cost of extra
        forward passes.
    """
    obs_dim = x_np.shape[1]
    rng = np.random.default_rng()

    # Sub-sample background rows (with replacement) if needed
    n_bg = bg_np.shape[0]
    if n_bg >= n_samples:
        bg_idx = rng.choice(n_bg, size=n_samples, replace=False)
    else:
        bg_idx = rng.choice(n_bg, size=n_samples, replace=True)
    bg_samples = bg_np[bg_idx]  # (n_samples, obs_dim)

    def _forward(arr: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.from_numpy(arr.astype(np.float32))
            return float(net(t).abs().sum().item())

    base_score = _forward(x_np)
    importances = np.zeros(obs_dim, dtype=np.float32)

    for feat_idx in range(obs_dim):
        # Replace the feature with the mean of the sampled background values
        replacement = float(bg_samples[:, feat_idx].mean())
        x_masked = x_np.copy()
        x_masked[:, feat_idx] = replacement
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
