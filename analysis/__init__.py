"""Wargames Training — analysis public API.

Stable interfaces for course-of-action generation and policy saliency analysis.
Import from this module to remain insulated from internal restructuring.

COA generation
--------------
:class:`COAGenerator`
    Generate and evaluate courses of action for battalion-level scenarios.
    Returns a ranked list of :class:`CourseOfAction` instances with
    win-rate, casualty, and time scores.

:class:`CorpsCOAGenerator`
    Generate and evaluate courses of action for corps-level scenarios.

:class:`COAScore`
    Per-strategy evaluation scores (win_rate, mean_casualties, mean_steps).

:class:`CourseOfAction`
    A named (label, score, strategy_index) course of action.

:func:`generate_coas`
    Convenience wrapper — run *n* episodes per strategy and return a ranked
    list of :class:`CourseOfAction`.

:func:`generate_corps_coas`
    Corps-level convenience wrapper.

Saliency
--------
:class:`SaliencyAnalyzer`
    High-level wrapper for computing and visualising policy saliency.

:func:`compute_gradient_saliency`
    Gradient-based saliency (vanilla gradient × observation).

:func:`compute_integrated_gradients`
    Integrated-gradients saliency (Axiomatic Attribution).

:func:`compute_shap_importance`
    SHAP-based feature importance (requires ``shap`` extra).

:func:`plot_saliency_map`
    Render a saliency heatmap over observation dimensions.

:func:`plot_feature_importance`
    Bar chart of mean absolute saliency per feature.

:data:`OBSERVATION_FEATURES`
    Tuple of human-readable feature names for the 12-dim BattalionEnv obs.
"""

from __future__ import annotations

# ── COA generation ────────────────────────────────────────────────────────
from analysis.coa_generator import (
    COAScore,
    CourseOfAction,
    COAGenerator,
    generate_coas,
    STRATEGY_LABELS,
    CorpsCOAScore,
    CorpsCourseOfAction,
    COAExplanation,
    COAModification,
    CorpsCOAGenerator,
    generate_corps_coas,
    CORPS_STRATEGY_LABELS,
)

# ── Saliency ──────────────────────────────────────────────────────────────
from analysis.saliency import (
    OBSERVATION_FEATURES,
    SaliencyAnalyzer,
    compute_gradient_saliency,
    compute_integrated_gradients,
    compute_shap_importance,
    plot_saliency_map,
    plot_feature_importance,
)

__all__ = [
    # COA generation
    "COAScore",
    "CourseOfAction",
    "COAGenerator",
    "generate_coas",
    "STRATEGY_LABELS",
    "CorpsCOAScore",
    "CorpsCourseOfAction",
    "COAExplanation",
    "COAModification",
    "CorpsCOAGenerator",
    "generate_corps_coas",
    "CORPS_STRATEGY_LABELS",
    # Saliency
    "OBSERVATION_FEATURES",
    "SaliencyAnalyzer",
    "compute_gradient_saliency",
    "compute_integrated_gradients",
    "compute_shap_importance",
    "plot_saliency_map",
    "plot_feature_importance",
]
