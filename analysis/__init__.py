"""Wargames Training — analysis public API (E12.2).

Stable interfaces for course-of-action generation and saliency analysis.
Import from this module to remain insulated from internal restructuring.

COA generation
--------------
:class:`~analysis.coa_generator.COAGenerator`
    Generate and evaluate courses of action for battalion-level scenarios.

:class:`~analysis.coa_generator.CorpsCOAGenerator`
    Generate and evaluate courses of action for corps-level scenarios.

:func:`~analysis.coa_generator.generate_coas` — convenience function.
:func:`~analysis.coa_generator.generate_corps_coas` — corps-level convenience.

Saliency
--------
:func:`~analysis.saliency.compute_gradient_saliency` — gradient-based saliency.
:func:`~analysis.saliency.compute_integrated_gradients` — integrated-gradients saliency.
:func:`~analysis.saliency.compute_shap_importance` — SHAP feature importance (optional).
"""

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

__all__ = [
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
]
