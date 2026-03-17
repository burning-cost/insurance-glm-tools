"""
Benchmark: insurance-glm-tools (FactorClusterer / R2VF)
=========================================================

Scenario: A UK motor insurer has 30 postcode districts. The true risk structure
has 7 territory bands (matching something like the ABI area bands A–G). The
actuary needs to group these 30 districts into rating territories.

We compare:
  Baseline: Manual quintile banding — sort districts by fitted GLM relativity,
            cut into 5 equal-exposure groups. Standard actuarial practice.

  Library:  R2VF via FactorClusterer (BIC-selected fused lasso). Merges adjacent
            districts (ordered by fitted relativity) where the difference shrinks
            to zero. Produces data-driven band boundaries.

IMPORTANT: R2VF operates on *adjacent* levels — it requires the factor to be
ordered. For postcode districts this means pre-ordering by fitted GLM
relativity (ascending) before calling FactorClusterer.

Metrics:
  - Test Poisson deviance
  - Adjusted Rand Index vs true 7-band territory structure
  - Train BIC
  - Number of bands produced

Seed: 42, n=20,000 policies, 30 districts.
"""

import time
import numpy as np
from sklearn.metrics import adjusted_rand_score
import statsmodels.api as sm
import pandas as pd

from insurance_glm_tools.cluster import FactorClusterer

print("=" * 60)
print("Benchmark: insurance-glm-tools (R2VF factor clustering)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 20_000

# 30 postcode districts, 7 true territory bands
# True structure: districts 0-3=band0, 4-7=band1, ..., 28-29=band6
n_districts = 30
n_true_bands = 7
true_band = np.array([min(d // (n_districts // n_true_bands), n_true_bands - 1)
                       for d in range(n_districts)])
# True log relativities for each band
band_log_rel = np.array([-0.30, -0.15, 0.00, 0.15, 0.35, 0.55, 0.80])
district_log_rel = band_log_rel[true_band]

vehicle_group = rng.integers(1, 20, n).astype(float)
ncd_years     = rng.integers(0, 9, n).astype(float)
district      = rng.integers(0, n_districts, n)
driver_age    = rng.integers(17, 75, n).astype(float)
exposure      = rng.uniform(0.3, 1.0, n)

log_rate = (
    -2.5
    + 0.015 * vehicle_group
    - 0.08 * ncd_years
    + district_log_rel[district]
    + 0.3 * (driver_age < 25).astype(float)
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

df = pd.DataFrame({
    "vehicle_group": vehicle_group,
    "ncd_years":     ncd_years,
    "district":      district.astype(float),
    "driver_age":    driver_age,
    "exposure":      exposure,
})

# Train/test split
n_train = int(0.8 * n)
idx_all = np.arange(n)
rng.shuffle(idx_all)
train_idx, test_idx = idx_all[:n_train], idx_all[n_train:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)
y_train  = y[train_idx]
y_test   = y[test_idx]
exp_train = exposure[train_idx]
exp_test  = exposure[test_idx]

print(f"\nPolicies: {n} (train={n_train}, test={len(test_idx)})")
print(f"Districts: {n_districts}, True territory bands: {n_true_bands}")
print(f"Mean frequency: {(y / exposure).mean():.4f}")


def poisson_deviance_sm(y_true, y_pred):
    y_pred_c = np.maximum(y_pred, 1e-10)
    mask = y_true > 0
    d = np.zeros_like(y_true, dtype=float)
    d[mask] = y_true[mask] * np.log(y_true[mask] / y_pred_c[mask]) - (y_true[mask] - y_pred_c[mask])
    d[~mask] = y_pred_c[~mask]
    return 2.0 * d.sum()


# ---------------------------------------------------------------------------
# Step 1: Fit initial GLM to get district-level fitted relativities
#         (both approaches use the same initial fit for ordering)
# ---------------------------------------------------------------------------
print("\n--- Step 1: Initial GLM for relativity ordering ---")

t0 = time.time()
# Fit unpenalised GLM with district dummies for relativity estimation
district_dummies_train = pd.get_dummies(df_train["district"].astype(int), prefix="d", drop_first=True)
X_init_train = pd.concat([
    df_train[["vehicle_group", "ncd_years"]],
    district_dummies_train,
], axis=1).astype(float)

X_sm_init = sm.add_constant(X_init_train)
glm_init = sm.GLM(
    y_train, X_sm_init,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_train, 1e-10)),
).fit(disp=False)
t_init = time.time() - t0
print(f"  Initial GLM fit time: {t_init:.2f}s")

# Extract district coefficients
d_cols = [c for c in X_init_train.columns if c.startswith("d_")]
d_coeffs = {int(c.split("_")[1]): glm_init.params.get(c, 0.0) for c in d_cols}
# District 0 is the reference (coefficient = 0)
d_coeffs[0] = 0.0
# Order districts by fitted relativity
district_order = sorted(range(n_districts), key=lambda d: d_coeffs.get(d, 0.0))
relativity_order_map = {d: i for i, d in enumerate(district_order)}
df_train["district_ordered"] = df_train["district"].astype(int).map(relativity_order_map)
df_test["district_ordered"]  = df_test["district"].astype(int).map(relativity_order_map)

# True band assignment in the ordered space
true_band_ordered = np.array([true_band[d] for d in district_order])

# ---------------------------------------------------------------------------
# Baseline: Manual quintile banding on ordered districts
# ---------------------------------------------------------------------------
print("\n--- Baseline: Manual quintile banding (5 bands) ---")

n_manual_bands = 5
district_sorted_relativity = sorted(d_coeffs.items(), key=lambda x: x[1])
# Assign each district to a quintile by relativity
quintile_map = {}
quintile_size = n_districts // n_manual_bands
for i, (d, _) in enumerate(district_sorted_relativity):
    quintile_map[d] = min(i // quintile_size, n_manual_bands - 1)

df_train["district_band_manual"] = df_train["district"].astype(int).map(quintile_map).astype(float)
df_test["district_band_manual"]  = df_test["district"].astype(int).map(quintile_map).astype(float)

t0 = time.time()
X_manual_train = pd.concat([
    df_train[["vehicle_group", "ncd_years", "district_band_manual"]],
    pd.get_dummies(df_train["district_band_manual"].astype(int), prefix="band", drop_first=True),
], axis=1).drop(columns=["district_band_manual"]).astype(float)

X_sm_manual_train = sm.add_constant(X_manual_train)
glm_manual = sm.GLM(
    y_train, X_sm_manual_train,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_train, 1e-10)),
).fit(disp=False)

X_manual_test = pd.concat([
    df_test[["vehicle_group", "ncd_years", "district_band_manual"]],
    pd.get_dummies(df_test["district_band_manual"].astype(int), prefix="band", drop_first=True),
], axis=1).drop(columns=["district_band_manual"]).astype(float)
# Align columns
X_manual_test = X_manual_test.reindex(columns=X_manual_train.columns, fill_value=0)
X_sm_manual_test = sm.add_constant(X_manual_test)
# Ensure test constant column matches
X_sm_manual_test = X_sm_manual_test.reindex(columns=X_sm_manual_train.columns, fill_value=0)

y_pred_manual = glm_manual.predict(X_sm_manual_test, offset=np.log(np.maximum(exp_test, 1e-10)))
t_manual = time.time() - t0

dev_manual = poisson_deviance_sm(y_test, y_pred_manual)
bic_manual = glm_manual.bic
n_manual_bands_actual = df_train["district_band_manual"].nunique()

# ARI: compare quintile band assignments vs true bands
pred_bands_manual = np.array([quintile_map[d] for d in range(n_districts)])
ari_manual = adjusted_rand_score(true_band, pred_bands_manual)

print(f"  Fit time: {t_manual:.2f}s")
print(f"  Bands produced: {n_manual_bands_actual}")
print(f"  Test Poisson deviance: {dev_manual:.1f}")
print(f"  Train BIC: {bic_manual:.1f}")
print(f"  Adjusted Rand Index vs true {n_true_bands}-band DGP: {ari_manual:.3f}")

# ---------------------------------------------------------------------------
# Library: R2VF via FactorClusterer (BIC-selected fused lasso)
# ---------------------------------------------------------------------------
print("\n--- Library: R2VF via FactorClusterer (BIC-selected) ---")

t0 = time.time()
fc = FactorClusterer(family="poisson", lambda_="bic", min_exposure=200)
fc.fit(
    df_train[["vehicle_group", "ncd_years", "district_ordered"]],
    y_train,
    exposure=exp_train,
    ordinal_factors=["district_ordered"],
)
t_r2vf = time.time() - t0

X_r2vf_train = fc.transform(df_train[["vehicle_group", "ncd_years", "district_ordered"]])
X_r2vf_test  = fc.transform(df_test[["vehicle_group", "ncd_years", "district_ordered"]])

result_r2vf = fc.refit_glm(X_r2vf_train, y_train, exposure=exp_train)

# Predict on test
X_r2vf_test_sm = sm.add_constant(X_r2vf_test)
X_r2vf_train_sm = sm.add_constant(X_r2vf_train)
X_r2vf_test_sm = X_r2vf_test_sm.reindex(columns=X_r2vf_train_sm.columns, fill_value=0)
y_pred_r2vf = result_r2vf.predict(X_r2vf_test_sm, offset=np.log(np.maximum(exp_test, 1e-10)))

dev_r2vf = poisson_deviance_sm(y_test, y_pred_r2vf)
bic_r2vf = result_r2vf.bic

# R2VF band count
lm = fc.level_map("district_ordered")
lm_df = lm.to_df()
n_r2vf_bands = lm_df["merged_group"].nunique()

# ARI: map district_ordered -> merged_group -> compare to true_band_ordered
r2vf_assignment = lm_df.set_index("original_level")["merged_group"].to_dict()
pred_bands_r2vf = np.array([r2vf_assignment.get(i, 0) for i in range(n_districts)])
ari_r2vf = adjusted_rand_score(true_band_ordered, pred_bands_r2vf)

print(f"  Fit time (IRLS + BIC grid): {t_r2vf:.1f}s")
print(f"  Bands produced: {n_r2vf_bands}")
print(f"  Test Poisson deviance: {dev_r2vf:.1f}")
print(f"  Train BIC: {bic_r2vf:.1f}")
print(f"  Adjusted Rand Index vs true {n_true_bands}-band DGP: {ari_r2vf:.3f}")

# Show level map
print(f"\n  Level map (district_ordered -> band):")
for _, row in lm_df.iterrows():
    print(f"    district_ordered={int(row['original_level']):2d} -> band={int(row['merged_group'])}  "
          f"exp={row.get('group_exposure', 0.0):.0f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Metric':<35} {'Manual quintiles':>16} {'R2VF (library)':>16}")
print("-" * 67)
print(f"{'Territory bands produced':<35} {n_manual_bands_actual:>16} {n_r2vf_bands:>16}")
print(f"{'Test Poisson deviance':<35} {dev_manual:>16.1f} {dev_r2vf:>16.1f}")
print(f"{'Train BIC':<35} {bic_manual:>16.1f} {bic_r2vf:>16.1f}")
print(f"{'Adjusted Rand Index (vs true DGP)':<35} {ari_manual:>16.3f} {ari_r2vf:>16.3f}")
print(f"{'Fit time':<35} {'<1s':>16} {t_r2vf:.0f}s")
print(f"\nTrue territory structure: {n_true_bands} bands")
print(f"ARI improvement (R2VF vs manual): {ari_r2vf - ari_manual:+.3f}")
print(f"\nNote: R2VF requires pre-ordering districts by fitted GLM relativity.")
print("Without ordering, the fused lasso has no meaningful adjacency structure.")
