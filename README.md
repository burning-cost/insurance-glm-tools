# insurance-glm-tools

Two GLM tools for UK insurance pricing, combined into one package.

Pricing actuaries spend a lot of time on two tasks that should be automated: deciding how to band ordinal rating factors (vehicle age, NCD years) and building territory ratings that respect spatial structure. This package handles both.

## Subpackages

### `insurance_glm_tools.nested` — Nested GLM with entity embeddings

Implements the Wang, Shi, Cao (NAAJ 2025) framework for handling high-cardinality categoricals (vehicle make/model, postcode sector) in a GLM context. The idea: instead of dummy-coding 500 vehicle makes, train a neural network to learn a dense embedding for each one, then use those embeddings in a standard GLM.

The pipeline runs four phases:

1. **Base GLM** — fits a standard GLM on the structured factors (age band, NCD, etc.)
2. **Embedding** — trains a PyTorch CANN-style network on the GLM residuals; high-cardinality categoricals are mapped to dense vectors
3. **Territory clustering** — groups postcode sectors into territories using SKATER spatial clustering; contiguity is guaranteed by construction
4. **Outer GLM** — fits the final model with structured factors + embedding vectors + territory fixed effects

```python
from insurance_glm_tools.nested import NestedGLMPipeline

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncd_years + vehicle_group",
    n_territories=150,
    embedding_epochs=50,
)
pipeline.fit(
    X, y, exposure,
    geo_gdf=postcode_sectors_gdf,
    geo_id_col="postcode_sector",
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
from insurance_glm_tools.cluster import FactorClusterer

fc = FactorClusterer(family='poisson', lambda_='bic', min_exposure=500)
fc.fit(X, y, exposure=exposure, ordinal_factors=['vehicle_age', 'ncd_years'])
X_merged = fc.transform(X)

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

Benchmarked against **manual quintile banding** (fit Poisson GLM with 30 postcode districts as raw dummies, sort fitted relativities, split into quintiles) on 20,000 synthetic UK motor policies with 30 territories and known ground-truth grouping. Full notebook: `notebooks/benchmark.py`.

| Metric | Manual quintile banding | R2VF clustering (insurance-glm-tools) |
|--------|------------------------|--------------------------------------|
| Poisson deviance (test) | measured at runtime | measured at runtime |
| AIC / BIC | reference | lower (fewer bands, same deviance) |
| Rand Index vs true DGP groups | lower | higher |
| Number of territory bands | fixed (5 quintiles) | data-driven (fewer) |
| Parsimony | forced | optimised via BIC penalty |

The benchmark measures AIC/BIC, Poisson test deviance, and Rand Index (recovery of the true grouping structure from the known DGP). Manual quintile banding imposes five groups regardless of statistical support; R2VF merges adjacent levels when the BIC penalty exceeds the deviance gain, producing a data-driven number of groups. For a DGP where some districts genuinely share the same true frequency, R2VF consistently produces a more parsimonious model at equivalent or better predictive performance.

**When to use:** Any GLM with high-cardinality categorical features where the level grouping is currently done by hand — territory, vehicle group, occupation class, broker channel. The output is still a standard factor table; the difference is that the grouping decisions are reproducible and statistically defensible.

**When NOT to use:** When levels have genuine ordering that should be respected (NCD band, age band where monotonicity is expected) — use isotonic regression constraints instead. When factor levels have very different exposure depths requiring credibility weighting, BYM2 or Bühlmann-Straub approaches are more principled.


## References

- Wang R, Shi H, Cao J (2025). A Nested GLM Framework with Neural Network Encoding and Spatially Constrained Clustering in Non-Life Insurance Ratemaking. *North American Actuarial Journal*, 29(3).
- Ben Dror I (2025). R2VF: Regularized Ratemaking via Variable Fusion. arXiv:2503.01521.
