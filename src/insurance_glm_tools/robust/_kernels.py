"""Kernel functions and bandwidth selection for RobustMMDGLM.

We use Gaussian (RBF) kernels throughout — both K_y (on the response) and K_x
(on the covariates, used to down-weight leverage points in the MMD loss).

The median heuristic is the standard bandwidth selector for MMD: set h to the
median pairwise distance. It's not optimal but it's robust and parameter-free.
For large n we subsample to avoid O(n^2) computation.

Reference: Kang & Kang (2026), arXiv:2602.21132, Section 3.1.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Kernel evaluations
# ---------------------------------------------------------------------------

def gaussian_kernel(a: np.ndarray, b: np.ndarray, h: float) -> np.ndarray:
    """K(a, b) = exp(-(a-b)^2 / 2h^2). Vectorised over matching axes."""
    return np.exp(-((a - b) ** 2) / (2.0 * h ** 2))


def gaussian_kernel_matrix(X: np.ndarray, h: float) -> np.ndarray:
    """Full n x n Gram matrix for X (1D array). Used in testing, not in ADMM."""
    diff = X[:, None] - X[None, :]
    return np.exp(-(diff ** 2) / (2.0 * h ** 2))


# ---------------------------------------------------------------------------
# Median heuristic
# ---------------------------------------------------------------------------

def median_bandwidth_y(y: np.ndarray, max_pairs: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
    """Estimate bandwidth for response kernel via median pairwise distance.

    For n > max_pairs we subsample rather than computing all n*(n-1)/2 pairs.
    The median of |y_i - y_j| is taken over a random subset of pairs.

    Args:
        y: Response vector, shape (n,).
        max_pairs: Maximum number of pairs to compute.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        Bandwidth h_y > 0.
    """
    n = len(y)
    if rng is None:
        rng = np.random.default_rng()

    if n <= 1:
        return 1.0

    # Sample pairs
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        # All pairs
        idx_i, idx_j = np.tril_indices(n, k=-1)
    else:
        idx_i = rng.integers(0, n, size=max_pairs)
        idx_j = rng.integers(0, n, size=max_pairs)
        # Avoid self-pairs
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        if len(idx_i) == 0:
            return 1.0

    dists = np.abs(y[idx_i] - y[idx_j])
    h = float(np.median(dists))

    # Fallback: if all points are identical, use std or unit bandwidth
    if h < 1e-10:
        h = float(np.std(y))
    if h < 1e-10:
        h = 1.0

    return h


def median_bandwidth_x(X: np.ndarray, max_pairs: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
    """Estimate bandwidth for covariate kernel via median pairwise Euclidean distance.

    Args:
        X: Design matrix, shape (n, p).
        max_pairs: Maximum number of pairs.
        rng: Optional numpy random Generator.

    Returns:
        Bandwidth h_x > 0.
    """
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    if n <= 1:
        return 1.0

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        idx_i, idx_j = np.tril_indices(n, k=-1)
    else:
        idx_i = rng.integers(0, n, size=max_pairs)
        idx_j = rng.integers(0, n, size=max_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        if len(idx_i) == 0:
            return 1.0

    diffs = X[idx_i] - X[idx_j]
    dists = np.linalg.norm(diffs, axis=1)
    h = float(np.median(dists))

    if h < 1e-10:
        h = 1.0

    return h


# ---------------------------------------------------------------------------
# MMD loss (per-observation, O(n) approximation)
# ---------------------------------------------------------------------------

def mmd_loss_per_obs(ek_y: np.ndarray) -> np.ndarray:
    """Per-observation MMD surrogate loss.

    l_tilde(theta, x_i, y_i) = 1 - 2 * E_{Y~F(theta,x_i)}[K_y(Y, y_i)]

    The constant term K_y(y_i, y_i) = 1 is dropped from the gradient.

    Args:
        ek_y: E[K_y(Y, y_i)] for each observation, shape (n,).

    Returns:
        Per-obs loss, shape (n,). Mean over n gives scalar MMD loss.
    """
    return 1.0 - 2.0 * ek_y


def total_mmd_loss(ek_y: np.ndarray) -> float:
    """Mean MMD loss across observations."""
    return float(np.mean(mmd_loss_per_obs(ek_y)))
