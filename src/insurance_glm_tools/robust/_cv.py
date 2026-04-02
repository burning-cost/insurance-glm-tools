"""Cross-validation for lambda selection in RobustMMDGLM.

We use the MMD loss (not deviance) as the CV criterion. This is deliberate:
the whole point of MMD-GLM is robustness to mis-specification, so we should
also validate using the same loss.

The lambda grid is log-spaced from lambda_max to lambda_max/1000:
- lambda_max is the smallest lambda that zeroes out all coefficients.
  For MMD loss, lambda_max = max |gradient of MMD loss at theta=0| / n.
  (This is analogous to the Lasso lambda_max = max |X^T y| / n.)

Reference: Kang & Kang (2026), arXiv:2602.21132, Section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_glm_tools.robust._families import Family


def _compute_lambda_max(
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    exposure_log: Optional[np.ndarray],
) -> float:
    """Compute lambda_max: the null-model gradient norm.

    At theta=0, the MMD gradient gives the scale of the problem.
    lambda_max = max |d/d_theta_j (1/n) * sum_i l_MMD|_{theta=0}

    We use the absolute max over features.
    """
    from insurance_glm_tools.robust._admm import _mmd_gradient

    theta_zero = np.zeros(X.shape[1])
    grad = _mmd_gradient(theta_zero, X, y, family, h_y, exposure_log)
    lambda_max = float(np.max(np.abs(grad)))

    # Avoid zero (can happen with degenerate data)
    if lambda_max < 1e-8:
        lambda_max = 1.0

    return lambda_max


def make_lambda_grid(
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    exposure_log: Optional[np.ndarray],
    n_lambdas: int = 60,
    lambda_ratio: float = 1e-3,
) -> np.ndarray:
    """Build a log-spaced lambda grid.

    Args:
        X: Design matrix, shape (n, p).
        y: Response vector.
        family: GLM family.
        h_y: Response kernel bandwidth.
        exposure_log: log(exposure) offset or None.
        n_lambdas: Number of grid points.
        lambda_ratio: lambda_min / lambda_max ratio.

    Returns:
        Lambda grid from lambda_max down to lambda_max * lambda_ratio, shape (n_lambdas,).
    """
    lam_max = _compute_lambda_max(X, y, family, h_y, exposure_log)
    lam_min = lam_max * lambda_ratio
    return np.geomspace(lam_max, lam_min, n_lambdas)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    exposure_log: Optional[np.ndarray],
    lambda_grid: np.ndarray,
    cv_folds: int = 5,
    rho: float = 1.0,
    adagrad_lr: float = 0.1,
    max_admm_iter: int = 200,
    max_inner_iter: int = 500,
    tol_primal: float = 1e-3,
    tol_dual: float = 1e-3,
    theta_init: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """K-fold CV over the lambda grid using MMD loss.

    Args:
        X: Design matrix, shape (n, p).
        y: Response vector.
        family: GLM family.
        h_y: Response kernel bandwidth.
        exposure_log: log(exposure) offset or None.
        lambda_grid: Array of lambda values to evaluate.
        cv_folds: Number of folds.
        rho: ADMM penalty.
        adagrad_lr: AdaGrad learning rate.
        max_admm_iter: ADMM iteration limit.
        max_inner_iter: Inner AdaGrad iteration limit.
        tol_primal, tol_dual: ADMM convergence tolerances.
        theta_init: Warm start for theta (if None, zeros).
        random_state: Seed for fold assignment.

    Returns:
        DataFrame with columns: lambda, mean_mmd_loss, std_mmd_loss, n_nonzero.
    """
    from insurance_glm_tools.robust._admm import admm_fit
    from insurance_glm_tools.robust._kernels import total_mmd_loss

    n, p = X.shape
    rng = np.random.default_rng(random_state)
    fold_ids = rng.integers(0, cv_folds, size=n)

    results = []

    for lam in lambda_grid:
        fold_losses = []

        for fold in range(cv_folds):
            val_mask = fold_ids == fold
            train_mask = ~val_mask

            if np.sum(train_mask) < 2 or np.sum(val_mask) < 1:
                continue

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            exp_tr = exposure_log[train_mask] if exposure_log is not None else None
            exp_val = exposure_log[val_mask] if exposure_log is not None else None

            t_init = np.zeros(p) if theta_init is None else theta_init.copy()

            theta_hat, _ = admm_fit(
                theta_init=t_init,
                X=X_tr,
                y=y_tr,
                family=family,
                h_y=h_y,
                lam=lam,
                exposure_log=exp_tr,
                rho=rho,
                adagrad_lr=adagrad_lr,
                max_admm_iter=max_admm_iter,
                max_inner_iter=max_inner_iter,
                tol_primal=tol_primal,
                tol_dual=tol_dual,
            )

            # Evaluate MMD loss on validation fold
            eta_val = X_val @ theta_hat
            if exp_val is not None:
                eta_val += exp_val
            mu_val = family.mean(eta_val)
            ek_val = family.ek_y(mu_val, y_val, h_y)
            val_loss = total_mmd_loss(ek_val)
            fold_losses.append(val_loss)

        if len(fold_losses) > 0:
            n_nonzero = int(np.sum(np.abs(theta_hat) > 1e-8))  # from last fold
            results.append({
                "lambda": float(lam),
                "mean_mmd_loss": float(np.mean(fold_losses)),
                "std_mmd_loss": float(np.std(fold_losses)),
                "n_nonzero": n_nonzero,
            })

    df = pd.DataFrame(results)
    return df


def select_lambda(cv_df: pd.DataFrame) -> float:
    """Select lambda that minimises mean CV MMD loss.

    If the DataFrame is empty, returns 0.01 as a fallback.
    """
    if cv_df.empty:
        return 0.01
    idx = cv_df["mean_mmd_loss"].idxmin()
    return float(cv_df.loc[idx, "lambda"])
