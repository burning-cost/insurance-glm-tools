# Changelog

## 0.2.0 (2026-04-02)

### New: `insurance_glm_tools.robust` subpackage

Implements `RobustMMDGLM` — an MMD-penalised GLM with L1 regularisation,
based on Kang & Kang (2026), arXiv:2602.21132.

Four GLM families supported:
- **Gaussian** (identity link): closed-form E[K_y] via Gaussian convolution
- **Logistic** (logit link): exact E[K_y] as two-point mixture
- **Poisson** (log link): E[K_y] via truncated summation with exposure offset
- **Gamma** (log link): E[K_y] via Gauss-Laguerre quadrature with exposure offset

Key features:
- L1 regularisation via ADMM with AdaGrad theta-step
- Cross-validation lambda selection using MMD loss (not deviance)
- Bootstrap confidence intervals for relativities
- `selected_features()`, `relativities()`, `cv_path()` convenience methods
- Warm start from sklearn Lasso/LogisticRegression/PoissonRegressor/GammaRegressor

Top-level import added: `from insurance_glm_tools import RobustMMDGLM`

Demo notebook: `notebooks/demo_robust_mmdglm.py`

## 0.1.7 and earlier

See git history.
