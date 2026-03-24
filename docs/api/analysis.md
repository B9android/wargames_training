# Analysis API

The `analysis` package exposes course-of-action generation and policy
saliency analysis as a clean Python API.

## Quick-start

```python
from analysis import COAGenerator, generate_coas, SaliencyAnalyzer

# Generate ranked courses of action
coas = generate_coas(env=env, policy=model, n_episodes=20)
for coa in coas:
    print(f"{coa.label}: win_rate={coa.score.win_rate:.1%}")

# Compute saliency for a policy
analyzer = SaliencyAnalyzer(policy=model, env=env)
saliency = analyzer.compute(obs, method="integrated_gradients")
analyzer.plot(saliency)
```

## COA generation

::: analysis.coa_generator.COAGenerator
::: analysis.coa_generator.CorpsCOAGenerator
::: analysis.coa_generator.COAScore
::: analysis.coa_generator.CourseOfAction
::: analysis.coa_generator.generate_coas
::: analysis.coa_generator.generate_corps_coas

## Saliency analysis

::: analysis.saliency.SaliencyAnalyzer
::: analysis.saliency.compute_gradient_saliency
::: analysis.saliency.compute_integrated_gradients
::: analysis.saliency.compute_shap_importance
::: analysis.saliency.plot_saliency_map
::: analysis.saliency.plot_feature_importance
