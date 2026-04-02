"""ADMM solver for the MMD-penalised GLM.

The optimisation problem is:

    min_{theta} (1/n) * sum_i l_MMD(theta; x_i, y_i) + lambda * ||theta||_1

where l_MMD(theta; x_i, y_i) = 1 - 2 * E_{Y~F(theta,x_i)}[K_y(Y, y_i)].

ADMM splitting: introduce z = theta, augmented Lagrangian:
    L_rho(theta, z, u) = f(theta) + lambda*||z||_1
                         + (rho/2) * ||theta - z + u||^2

Steps:
    theta-step: min_theta f(theta) + (rho/2)||theta - z + u||^2
                => AdaGrad gradient descent (smooth, non-convex in theta)
    z-step:     soft-threshold(theta + u, lambda/rho)
    u-step:     u += theta - z

Convergence: primal residual ||theta - z|| < tol_primal
             dual residual   rho * ||z_new - z_old|| < tol_dual

Note on AdaGrad: we use a per-parameter accumulated gradient for adaptive
step sizes. This is important because the MMD gradient can vary wildly in
scale across features. We reset the AdaGrad accumulator at each ADMM
iteration to avoid over-dampening.

Reference: Kang & Kang (2026), arXiv:2602.21132, Algorithm 1.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_glm_tools.robust._families import Family


# ---------------------------------------------------------------------------
# Gradient of MMD loss wrt theta
# ---------------------------------------------------------------------------

def _mmd_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    exposure_log: Optional[np.ndarray],
) -> np.ndarray:
    """Gradient of (1/n) * sum_i l_MMD wrt theta.

    Chain rule:
        d/d_theta l_MMD_i = -2 * (d E[K_y] / d mu_i) * (d mu_i / d eta_i) * x_i

    For log-link families: d mu_i / d eta_i = mu_i
    For identity link (Gaussian): d mu_i / d eta_i = 1
    For logit link: d mu_i / d eta_i = mu_i * (1 - mu_i)

    We compute d mu / d eta numerically to avoid family-specific code here.
    """
    n, p = X.shape
    eps = 1e-5

    # Compute eta and mu
    eta = X @ theta
    if exposure_log is not None:
        eta = eta + exposure_log

    mu = family.mean(eta)

    # dE[K_y]/dmu — family computes this
    dek = family.dek_dmu(mu, y, h_y)  # shape (n,)

    # dmu/deta via finite differences on eta
    eta_up = eta + eps
    mu_up = family.mean(eta_up)
    dmu_deta = (mu_up - mu) / eps  # shape (n,)

    # Chain: d l_MMD_i / d theta = -2 * dek_i * dmu_deta_i * x_i
    weight = -2.0 * dek * dmu_deta  # shape (n,)

    # Gradient: (1/n) * X^T @ weight
    grad = (X.T @ weight) / n  # shape (p,)

    return grad


# ---------------------------------------------------------------------------
# AdaGrad theta-step
# ---------------------------------------------------------------------------

def _adagrad_step(
    theta_init: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    exposure_log: Optional[np.ndarray],
    rho: float,
    lr: float,
    max_iter: int,
    tol: float = 1e-5,
) -> np.ndarray:
    """Minimise f(theta) + (rho/2)||theta - z + u||^2 via AdaGrad.

    Args:
        theta_init: Starting point.
        z: Current z variable.
        u: Current dual variable.
        X: Design matrix (n, p).
        y: Response (n,).
        family: GLM family.
        h_y: Response kernel bandwidth.
        exposure_log: log(exposure) offset or None.
        rho: ADMM penalty parameter.
        lr: AdaGrad base learning rate.
        max_iter: Maximum inner iterations.
        tol: Gradient norm tolerance for early stopping.

    Returns:
        Updated theta, shape (p,).
    """
    theta = theta_init.copy()
    p = len(theta)
    G = np.zeros(p)  # AdaGrad accumulated squared gradient
    eps_adagrad = 1e-8

    target = z - u

    for _ in range(max_iter):
        # MMD gradient
        g_mmd = _mmd_gradient(theta, X, y, family, h_y, exposure_log)

        # Quadratic augmentation gradient
        g_aug = rho * (theta - target)

        g = g_mmd + g_aug

        # AdaGrad update
        G += g ** 2
        step = lr / (np.sqrt(G) + eps_adagrad)
        theta = theta - step * g

        if np.linalg.norm(g) < tol:
            break

    return theta


# ---------------------------------------------------------------------------
# Soft thresholding
# ---------------------------------------------------------------------------

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft thresholding: sign(x) * max(|x| - threshold, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


# ---------------------------------------------------------------------------
# Main ADMM loop
# ---------------------------------------------------------------------------

def admm_fit(
    theta_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    family: "Family",
    h_y: float,
    lam: float,
    exposure_log: Optional[np.ndarray],
    rho: float = 1.0,
    adagrad_lr: float = 0.1,
    max_admm_iter: int = 200,
    max_inner_iter: int = 500,
    tol_primal: float = 1e-3,
    tol_dual: float = 1e-3,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run ADMM to solve the L1-penalised MMD-GLM.

    Args:
        theta_init: Initial coefficient vector, shape (p,).
        X: Design matrix, shape (n, p).
        y: Response vector, shape (n,).
        family: GLM family object.
        h_y: Response kernel bandwidth.
        lam: L1 regularisation strength.
        exposure_log: log(exposure) offset, shape (n,), or None.
        rho: ADMM augmented Lagrangian penalty.
        adagrad_lr: AdaGrad learning rate for theta-step.
        max_admm_iter: Maximum outer ADMM iterations.
        max_inner_iter: Maximum inner AdaGrad iterations per ADMM step.
        tol_primal: Primal residual convergence tolerance.
        tol_dual: Dual residual convergence tolerance.
        verbose: Print convergence info.

    Returns:
        (theta, info_dict) where info_dict contains convergence diagnostics.
    """
    theta = theta_init.copy()
    z = theta.copy()
    u = np.zeros_like(theta)

    n, p = X.shape
    info = {
        "converged": False,
        "n_iter": max_admm_iter,
        "primal_residuals": [],
        "dual_residuals": [],
    }

    for iteration in range(max_admm_iter):
        z_old = z.copy()

        # --- theta-step ---
        theta = _adagrad_step(
            theta_init=theta,
            z=z,
            u=u,
            X=X,
            y=y,
            family=family,
            h_y=h_y,
            exposure_log=exposure_log,
            rho=rho,
            lr=adagrad_lr,
            max_iter=max_inner_iter,
        )

        # --- z-step (proximal L1) ---
        z = soft_threshold(theta + u, lam / rho)

        # --- dual update ---
        u = u + theta - z

        # --- Convergence check ---
        primal_res = float(np.linalg.norm(theta - z))
        dual_res = float(rho * np.linalg.norm(z - z_old))

        info["primal_residuals"].append(primal_res)
        info["dual_residuals"].append(dual_res)

        if verbose:
            print(
                f"  ADMM iter {iteration+1:3d}: "
                f"primal={primal_res:.4e}, dual={dual_res:.4e}, "
                f"nnz={np.sum(np.abs(z) > 1e-8)}"
            )

        if primal_res < tol_primal and dual_res < tol_dual:
            info["converged"] = True
            info["n_iter"] = iteration + 1
            break

    # Return z (the proximal-regularised variable) as the final coefficients
    # theta and z should be close at convergence
    return z, info
