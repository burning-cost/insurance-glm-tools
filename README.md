# insurance-glm-tools

[![PyPI](https://img.shields.io/pypi/v/insurance-glm-tools)](https://pypi.org/project/insurance-glm-tools/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-glm-tools)](https://pypi.org/project/insurance-glm-tools/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

Two GLM tools for UK insurance pricing, combined into one package.

Pricing actuaries spend a lot of time on two tasks that should be automated: deciding how to band ordinal rating factors (vehicle age, NCD years) and building territory ratings that respect spatial structure. This package handles both.

## Quick Start

```bash
pip install insurance-glm-tools
```

```python
import numpy as np
import pandas as pd
from insurance_glm_tools.cluster import FactorClusterer

rng = np.random.default_rng(42)
n = 5000

# 16 vehicle age levels — true DGP has a break at year 8 and another at year 12
df = pd.DataFrame({
    "vehicle_age": rng.integers(0, 16, n),
    "ncd_years":   rng.integers(0, 10, n),
    "driver_age":  rng.integers(17, 75, n),
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    + 0.3 * (df["vehicle_age"].to_numpy() > 8).astype(float)
    + 0.5 * (df["vehicle_age"].to_numpy() > 12).astype(float)
    - 0.12 * df["ncd_years"].to_numpy()
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

# BIC selects regularisation strength; adjacent levels with zero difference are merged
fc = FactorClusterer(family="poisson", lambda_="bic", min_exposure=200)
fc.fit(df, y, exposure=exposure, ordinal_factors=["vehicle_age", "ncd_years"])

print(fc.level_map("vehicle_age").to_df())
#    original_level  merged_group  coefficient  group_exposure
# 0               0             0    -0.117359     1853.927617
# 1               1             0    -0.117359     1853.927617
# ...
# 8               8             0    -0.117359     1853.927617
# 9               9             1     0.192394     1401.495288
# 10             10             1     0.192394     1401.495288
# ...
# 15             15             1     0.192394     1401.495288
#
# BIC selects 2 groups: levels 0-8 (low-risk) and 9-15 (high-risk).
# The DGP has a second break at year 12, but with min_exposure=200 and
# n=5000 there is insufficient data to resolve three distinct bands at
# this sample size — BIC correctly prefers parsimony.

X_merged = fc.transform(df)
result = fc.refit_glm(X_merged, y, exposure=exposure)
print(result.summary())
```

## Subpackages

### `insurance_glm_tools.nested` — Nested GLM with entity embeddings

Implements the Wang, Shi, Cao (NAAJ 2025) framework for handling high-cardinality categoricals (vehicle make/model, postcode sector) in a GLM context. The idea: instead of dummy-coding 500 vehicle makes, train a neural network to learn a dense embedding for each one, then use those embeddings in a standard GLM.

The pipeline runs four phases:

1. **Base GLM** — fits a standard GLM on the structured factors (age band, NCD, etc.)
2. **Embedding** — trains a PyTorch CANN-style network on the GLM residuals; high-cardinality categoricals are mapped to dense vectors
3. **Territory clustering** — groups postcode sectors into territories using SKATER spatial clustering; contiguity is guaranteed by construction
4. **Outer GLM** — fits the final model with structured factors + embedding vectors + territory fixed effects

The spatial pipeline (`geo_gdf` parameter) requires `pip install "insurance-glm-tools[spatial]"` and a GeoDataFrame of postcode sector polygons. If you do not have spatial data, you can omit the `geo_gdf` argument and the pipeline skips territory clustering — this still gives you the embedding benefit for high-cardinality categoricals.

The example below shows the non-spatial variant, which works with any tabular dataset:

```python
import numpy as np
import pandas as pd
from insurance_glm_tools.nested import NestedGLMPipeline

rng = np.random.default_rng(42)
n = 1000

# High-cardinality vehicle make/model: 80 distinct values
vehicle_makes = [f"make_{i:03d}" for i in rng.integers(0, 80, n)]

df = pd.DataFrame({
    "age_band":          rng.choice(["17-25", "26-35", "36-50", "51-65", "66+"], n),
    "ncd_years":         rng.integers(0, 10, n),
    "vehicle_group":     rng.integers(1, 20, n),
    "vehicle_make_model": vehicle_makes,
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    + 0.03 * (df["ncd_years"].to_numpy() == 0).astype(float)
    - 0.02 * df["ncd_years"].to_numpy()
    + 0.02 * df["vehicle_group"].to_numpy()
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncd_years + vehicle_group",
    embedding_epochs=20,
    # n_territories and geo_gdf omitted: no spatial clustering
)
pipeline.fit(
    df, y, exposure,
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncd_years", "vehicle_group"],
)

relativities = pipeline.relativities()
```

You can also use the components independently:

```python
from insurance_glm_tools.nested import EmbeddingTrainer, TerritoryClusterer, NestedGLM
```

### `insurance_glm_tools.cluster` — R2VF factor-level clustering

Automates the process of banding ordinal GLM factors. Given a factor with 16 vehicle age levels, R2VF (Ben Dror 2025, arXiv:2503.01521) finds the optimal grouping by fitting a fused lasso on the split-coded design matrix. Adjacent levels whose difference shrinks to zero get merged.

The standard workflow is three lines:

```python
import numpy as np
import pandas as pd
from insurance_glm_tools.cluster import FactorClusterer

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "vehicle_age": rng.integers(0, 16, n),
    "driver_age":  rng.integers(17, 75, n),
    "ncd_years":   rng.integers(0, 10, n),
    "area_code":   rng.integers(1, 6, n),
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    + 0.04 * (df["vehicle_age"].to_numpy() > 8).astype(float)
    - 0.02 * df["ncd_years"].to_numpy()
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

fc = FactorClusterer(family='poisson', lambda_='bic', min_exposure=500)
fc.fit(df, y, exposure=exposure, ordinal_factors=['vehicle_age', 'ncd_years'])
X_merged = fc.transform(df)

# Inspect the groupings
print(fc.level_map('vehicle_age').to_df())

# Unpenalised refit on merged encoding
result = fc.refit_glm(X_merged, y, exposure=exposure)
```

BIC selects the regularisation strength automatically. The `min_exposure` constraint prevents groups with insufficient data from standing alone.

## Installation

```bash
pip install insurance-glm-tools
```

With spatial clustering support (geopandas, libpysal, spopt):

```bash
pip install insurance-glm-tools[spatial]
```

With plotting:

```bash
pip install insurance-glm-tools[plot]
```

## Dependencies

Core: numpy, pandas, scipy, scikit-learn, statsmodels, torch

Optional spatial: geopandas, libpysal, spopt

Optional plotting: matplotlib

## Design decisions

**Why one package?** Both subpackages target the same workflow: fitting GLMs on UK motor data. Keeping them together avoids duplication and makes it easy to combine them (e.g. use `cluster` to band vehicle age, then pass the banded factors into `nested` as the base formula).

**Why not simplify the nested GLM?** The four-phase structure is the point. It mirrors the actuarial workflow: structured factors first, then high-cardinality corrections, then geography. Collapsing it into a black box loses interpretability.

**Why IRLS + Lasso for R2VF?** sklearn's `PoissonRegressor` uses L2 (ridge) penalty — it cannot shrink coefficients to exactly zero, which is necessary for clean level fusion. The IRLS approach with a Lasso inner step gives true L1 penalisation.

## Source repos

This package consolidates two previously separate libraries:

- `insurance-nested-glm` — archived, merged into `insurance_glm_tools.nested`
- `insurance-glm-cluster` — archived, merged into `insurance_glm_tools.cluster`

---

## Performance

Benchmarked on Databricks serverless, 2026-03-16. DGP: 20,000 synthetic UK motor policies, 30 postcode districts, 7-band true territory structure. Baseline: fit a PoissonRegressor with all 30 districts as dummies, extract fitted relativities, sort into 5 quintile bands, refit. R2VF: fit FactorClusterer with BIC-selected fused lasso, refit unpenalised GLM on merged encoding.

**Note on ordering:** For R2VF to be meaningful on postcode districts, districts must be pre-ordered by fitted relativity (e.g. ascending GLM coefficient) before passing them as an ordinal factor. R2VF's fused lasso merges *adjacent* levels — it assumes that neighbouring indices are risk-neighbours. Without this ordering step, the fused lasso has no meaningful structure to exploit.

| Metric | Manual quintile banding | R2VF (BIC-selected) |
|--------|------------------------|----------------------|
| Territory bands produced | 5 (fixed) | 5 (BIC-selected, same here) |
| Test Poisson deviance | 2,389 | 2,474 |
| Train BIC | 7,866 | 7,980 |
| Adjusted Rand Index (vs true 7-band DGP) | 0.139 | 0.384 |
| Fit time | 0.7s | 229s |

**Honest result:** On this benchmark R2VF produces worse deviance and worse BIC than manual quintile banding, but substantially better ARI (0.384 vs 0.139). The ARI difference means R2VF recovers the true territory structure almost 3x more accurately than manual quintile banding — even though the predictive metrics are slightly worse.

**Interpreting the trade-off:** Manual quintile banding optimises the training data split; it does not care about whether the grouping matches the true underlying structure. R2VF optimises grouping quality (merges districts with similar true risk) at some cost to test deviance. In a real setting this means:

- R2VF produces groups that are more stable year-on-year (they reflect the underlying risk, not artefacts of one year's relativity ranking)
- R2VF groups are more defensible in FCA scrutiny (the structure is statistically derived, not drawn by hand)
- Manual quintiles may produce slightly better in-sample AIC/BIC on the current year but the boundaries are arbitrary

The fit time difference (0.7s vs 229s for 20k policies) reflects the IRLS + lambda grid search in R2VF vs a single sklearn fit. This is negligible in a weekly or nightly pricing run, but too slow for interactive model exploration on large datasets.

**When to use R2VF:** Territory grouping, vehicle group merging, occupation class consolidation — anywhere you have an ordinal factor where the manual grouping is currently done by eye. The ARI improvement is the headline: you get a grouping that matches the true risk structure more closely, even if the test deviance is marginally worse.

**When NOT to use:** When the factor has very unequal exposure per level (BIC penalisation per parameter is aggressive; low-exposure districts will be over-merged). The `min_exposure` parameter partially addresses this. Also not appropriate for nominal factors (unordered categoricals) — R2VF assumes ordinal structure.

See `notebooks/benchmark_databricks.py` for the runnable benchmark.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_glm_tools_demo.py).

## References

- Wang R, Shi H, Cao J (2025). A Nested GLM Framework with Neural Network Encoding and Spatially Constrained Clustering in Non-Life Insurance Ratemaking. *North American Actuarial Journal*, 29(3).
- Ben Dror I (2025). R2VF: Regularized Ratemaking via Variable Fusion. arXiv:2503.01521.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Generalised Additive Models — smooth non-linear effects as an alternative to discretised GLM factor levels |
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking — use GLM tools to band the resulting territory relativities into factor levels |
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Whittaker-Henderson graduation — smooth development triangles and rate level indices before GLM fitting |
