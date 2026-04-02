# Databricks notebook source
# MAGIC %md
# MAGIC # RobustMMDGLM: MMD-Penalised GLMs for Robust Insurance Pricing
# MAGIC
# MAGIC **Reference**: Kang & Kang (2026), arXiv:2602.21132
# MAGIC
# MAGIC This notebook demonstrates `RobustMMDGLM` from `insurance-glm-tools v0.2.0`.
# MAGIC
# MAGIC The core idea: replace the log-likelihood with the Maximum Mean Discrepancy (MMD)
# MAGIC between the empirical distribution and the fitted model distribution. MMD is
# MAGIC robust to heavy-tailed contamination because the Gaussian kernel K_y(y1, y2)
# MAGIC down-weights large discrepancies geometrically rather than linearly.
# MAGIC
# MAGIC **Why this matters for insurance pricing:**
# MAGIC - Severity models are routinely contaminated by large claims that inflate MLE estimates
# MAGIC - The MMD loss caps the influence of any single extreme observation
# MAGIC - L1 regularisation gives automatic feature selection across large rating factor sets
# MAGIC - All four standard families supported: Gaussian, Logistic, Poisson, Gamma

# COMMAND ----------

# MAGIC %pip install insurance-glm-tools==0.2.0 --quiet

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from insurance_glm_tools.robust import RobustMMDGLM

print("insurance-glm-tools version:")
import insurance_glm_tools
print(insurance_glm_tools.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Gamma Severity Model (the key insurance use case)
# MAGIC
# MAGIC We simulate a Gamma severity dataset with:
# MAGIC - 500 policies, 4 rating factors (2 active, 2 noise)
# MAGIC - 5% large claim contamination (claims 100x the normal level)
# MAGIC - Vehicle years exposure

# COMMAND ----------

rng = np.random.default_rng(42)
n = 500

# Simulate rating factors
age_band = rng.choice([0.0, 0.1, -0.1, 0.2], size=n)        # vehicle age
region = rng.choice([-0.2, 0.0, 0.15, 0.3], size=n)          # region factor
noise1 = rng.standard_normal(n) * 0.01                        # irrelevant feature
noise2 = rng.standard_normal(n) * 0.01                        # irrelevant feature

X = np.column_stack([age_band, region, noise1, noise2])
feature_names = ["age_band", "region", "noise1", "noise2"]

true_beta = np.array([0.5, -0.3, 0.0, 0.0])
exposure = rng.uniform(0.5, 2.0, size=n)

# Gamma severity
mu_true = exposure * np.exp(X @ true_beta)
alpha_true = 3.0
y_severity = rng.gamma(shape=alpha_true, scale=mu_true / alpha_true)

# Contaminate 5% with catastrophic claims
outlier_idx = rng.choice(n, size=25, replace=False)
y_contaminated = y_severity.copy()
y_contaminated[outlier_idx] *= 100

print(f"Clean data: mean severity = {y_severity.mean():.2f}, 99th pct = {np.percentile(y_severity, 99):.2f}")
print(f"Contaminated: mean severity = {y_contaminated.mean():.2f}, 99th pct = {np.percentile(y_contaminated, 99):.2f}")
print(f"True beta: {true_beta}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit: MMD-Gamma vs standard Gamma GLM

# COMMAND ----------

# Fit RobustMMDGLM
mmd_model = RobustMMDGLM(
    family="gamma",
    lambda_="cv",
    cv_folds=5,
    n_quadrature=24,
    max_admm_iter=150,
    max_inner_iter=300,
    init="lasso",
    random_state=42,
)
mmd_model.fit(X, y_contaminated, exposure=exposure, feature_names=feature_names)

print("MMD-Gamma fitted.")
print(f"Selected lambda: {mmd_model._lambda_value:.4g}")
print(f"Selected features: {mmd_model.selected_features()}")
print()
print("Relativities:")
display(mmd_model.relativities())

# COMMAND ----------

# Compare with standard Gamma GLM (statsmodels)
from sklearn.linear_model import GammaRegressor

gamma_mle = GammaRegressor(alpha=0.0, fit_intercept=False, max_iter=500)
gamma_mle.fit(X, y_contaminated, sample_weight=exposure)

print("Standard Gamma GLM (MLE, contaminated data):")
for name, coef, true in zip(feature_names, gamma_mle.coef_, true_beta):
    print(f"  {name:12s}: MLE={coef:+.4f}  MMD={mmd_model.coef_[feature_names.index(name)]:+.4f}  True={true:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CV Path: Lambda selection

# COMMAND ----------

cv_df = mmd_model.cv_path()
if cv_df is not None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cv_df["lambda"], cv_df["mean_mmd_loss"], "b-o", markersize=3)
    ax.axvline(mmd_model._lambda_value, color="red", linestyle="--", label=f"Selected λ={mmd_model._lambda_value:.4g}")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Mean CV MMD Loss")
    ax.set_title("Gamma MMD-GLM: Cross-Validation Path")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/cv_path_gamma.png", dpi=100)
    plt.show()
    display(cv_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Poisson Frequency Model

# COMMAND ----------

rng2 = np.random.default_rng(99)
n_freq = 400

X_freq = rng2.standard_normal((n_freq, 5))
beta_freq = np.array([0.4, -0.3, 0.0, 0.0, 0.0])
exposure_freq = rng2.uniform(0.5, 3.0, size=n_freq)
mu_freq = exposure_freq * np.exp(X_freq @ beta_freq)
y_freq = rng2.poisson(mu_freq).astype(float)

print(f"Frequency data: {n_freq} policies, mean claim count = {y_freq.mean():.3f}")
print(f"True beta: {beta_freq}")

# COMMAND ----------

freq_model = RobustMMDGLM(
    family="poisson",
    lambda_="cv",
    cv_folds=5,
    max_admm_iter=100,
    max_inner_iter=200,
    init="lasso",
    random_state=42,
)
freq_model.fit(
    X_freq, y_freq,
    exposure=exposure_freq,
    feature_names=["age", "region", "nc", "bonus", "noise"],
)

print(f"Selected lambda: {freq_model._lambda_value:.4g}")
print(f"Selected features: {freq_model.selected_features()}")
print()
display(freq_model.relativities())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bootstrap Confidence Intervals
# MAGIC
# MAGIC For a fitted model we can get bootstrap CIs using `bootstrap_relativities_from_data`.
# MAGIC This refits on 100 bootstrap samples to quantify uncertainty.

# COMMAND ----------

# Use a smaller model for speed
rng3 = np.random.default_rng(7)
n_small = 200
X_small = rng3.standard_normal((n_small, 3))
beta_small = np.array([0.5, -0.3, 0.0])
exposure_small = np.ones(n_small)
mu_small = np.exp(X_small @ beta_small)
y_small = rng3.gamma(shape=3.0, scale=mu_small / 3.0)

boot_model = RobustMMDGLM(
    family="gamma",
    lambda_=0.05,
    n_quadrature=16,
    max_admm_iter=80,
    max_inner_iter=150,
    init="lasso",
    random_state=0,
)
boot_model.fit(X_small, y_small, feature_names=["x0", "x1", "x2"])

boot_df = boot_model.bootstrap_relativities_from_data(
    X_small, y_small, n_boot=50, ci=0.95
)
print("Bootstrap relativities (95% CI):")
display(boot_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gaussian Robustness Comparison
# MAGIC
# MAGIC Side-by-side comparison of MMD-GLM vs Lasso under 10% contamination.

# COMMAND ----------

from sklearn.linear_model import Lasso

rng4 = np.random.default_rng(2026)
n_gauss = 300
X_gauss = rng4.standard_normal((n_gauss, 5))
beta_gauss = np.array([2.0, -1.5, 1.0, 0.0, 0.0])
y_clean = X_gauss @ beta_gauss + rng4.standard_normal(n_gauss) * 0.5

# Add 10% contamination
y_noisy = y_clean.copy()
outlier_mask = rng4.random(n_gauss) < 0.1
y_noisy[outlier_mask] += rng4.choice([-25, 25], size=outlier_mask.sum())

print(f"Contamination: {outlier_mask.sum()} outliers ({100*outlier_mask.mean():.0f}%)")
print(f"True beta: {beta_gauss}")

# Fit both
mmd_gauss = RobustMMDGLM(
    family="gaussian",
    lambda_="cv",
    cv_folds=5,
    max_admm_iter=100,
    max_inner_iter=200,
    init="lasso",
    random_state=0,
)
mmd_gauss.fit(X_gauss, y_noisy, feature_names=["x0", "x1", "x2", "x3", "x4"])

lasso_gauss = Lasso(alpha=0.05, max_iter=5000, fit_intercept=False)
lasso_gauss.fit(X_gauss, y_noisy)

print("\nCoefficient comparison (clean vs contaminated):")
header = f"{'Feature':8s}  {'True':>8s}  {'MMD':>8s}  {'Lasso':>8s}"
print(header)
print("-" * len(header))
names = ["x0", "x1", "x2", "x3", "x4"]
for name, bt, bm, bl in zip(names, beta_gauss, mmd_gauss.coef_, lasso_gauss.coef_):
    print(f"{name:8s}  {bt:+8.3f}  {bm:+8.3f}  {bl:+8.3f}")

mmd_err = np.linalg.norm(mmd_gauss.coef_ - beta_gauss)
lasso_err = np.linalg.norm(lasso_gauss.coef_ - beta_gauss)
print(f"\n||beta_MMD - beta_true|| = {mmd_err:.3f}")
print(f"||beta_Lasso - beta_true|| = {lasso_err:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Logistic: Binary Classification with Leverage Outliers

# COMMAND ----------

rng5 = np.random.default_rng(55)
n_logit = 400
X_logit = rng5.standard_normal((n_logit, 4))
beta_logit = np.array([1.5, -1.0, 0.0, 0.0])
prob_logit = 1.0 / (1.0 + np.exp(-(X_logit @ beta_logit)))
y_logit = rng5.binomial(1, prob_logit).astype(float)

# Add leverage outliers: very high-risk profile but y=0
X_lev = np.zeros((20, 4))
X_lev[:, 0] = 10.0
y_lev = np.zeros(20)
X_logit_cont = np.vstack([X_logit, X_lev])
y_logit_cont = np.concatenate([y_logit, y_lev])

logit_model = RobustMMDGLM(
    family="logistic",
    lambda_="cv",
    cv_folds=5,
    max_admm_iter=100,
    max_inner_iter=200,
    init="lasso",
    random_state=0,
)
logit_model.fit(
    X_logit_cont, y_logit_cont,
    feature_names=["high_risk", "nc", "bonus", "age"],
)

print(f"Selected features: {logit_model.selected_features()}")
print()
display(logit_model.relativities())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Family    | Use case                          | Key advantage vs MLE          |
# MAGIC |-----------|-----------------------------------|-------------------------------|
# MAGIC | Gaussian  | Pure premium, additive factors    | Bounded influence function    |
# MAGIC | Logistic  | Binary classification             | Leverage point resistance     |
# MAGIC | Poisson   | Claim frequency with exposure     | Robust to count outliers      |
# MAGIC | Gamma     | Claim severity with exposure      | Robust to catastrophic claims |
# MAGIC
# MAGIC **Implementation notes:**
# MAGIC - Gaussian E[K_y]: closed form via convolution of Gaussians
# MAGIC - Logistic E[K_y]: exact (two-point mixture)
# MAGIC - Poisson E[K_y]: truncated series (1-1e-10 quantile cutoff)
# MAGIC - Gamma E[K_y]: Gauss-Laguerre quadrature (32 nodes by default)
# MAGIC
# MAGIC **ADMM algorithm:**
# MAGIC - Theta-step: AdaGrad gradient descent on (MMD loss + quadratic augmentation)
# MAGIC - Z-step: Soft thresholding for L1 penalty
# MAGIC - Dual update: standard scaled ADMM
# MAGIC
# MAGIC **Lambda selection:**
# MAGIC - CV criterion: MMD loss (not deviance) — consistent with the estimator
# MAGIC - Default grid: 60 log-spaced values from lambda_max to lambda_max/1000
# MAGIC - lambda_max from null-model gradient norm
