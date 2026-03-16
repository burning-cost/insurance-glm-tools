"""
Low-level split-coding matrix utilities.

These are the matrix-level primitives underlying the fused lasso reparameterisation.
The higher-level make_split_coded_matrix in penalties.py takes a pd.Series and
ordered level list and is what FactorClusterer uses directly. These functions
operate on raw arrays and are useful for testing, external integration, or when
you already have a one-hot encoded design matrix you want to transform.

Background
----------
The split-coding trick reduces the fused lasso penalty on adjacent differences
(λ·Σ|β_i - β_{i-1}|) to a plain L1 penalty after a change of basis. The
transformation matrix T is a lower-triangular matrix of ones: β = T @ δ.

The split-coded design matrix is X_split = X_onehot @ T.  Fitting with an L1
penalty on the δ coefficients in this basis recovers the fused lasso on β.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_split_coding_matrix(n_levels: int) -> NDArray[np.float64]:
    """
    Build the split-coding transformation matrix for fused lasso.

    The split-coding trick converts a standard fused lasso (penalty on
    differences β_i - β_{i-1}) into a standard L1 problem. Define
    δ_i = β_i - β_{i-1} for i >= 1, and δ_0 = β_0. Then β_i = Σ_{s≤i} δ_s.

    This function returns the matrix T such that β = T @ δ, i.e. a lower
    triangular matrix of ones.

    Parameters
    ----------
    n_levels : int
        Number of factor levels.

    Returns
    -------
    NDArray[np.float64]
        Lower triangular matrix of shape (n_levels, n_levels).

    Examples
    --------
    >>> T = build_split_coding_matrix(3)
    >>> T
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]])
    """
    return np.tril(np.ones((n_levels, n_levels), dtype=np.float64))


def apply_split_coding(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform a one-hot design matrix from β-space to δ-space using split coding.

    If X_onehot has columns [x_0, x_1, ..., x_{K-1}] corresponding to level
    indicators, the split-coded design matrix is X_split = X_onehot @ T, where
    T is the lower-triangular-ones matrix.  Fitting an L1-penalised GLM on the
    δ coefficients in this basis is equivalent to the fused lasso on the
    original β coefficients.

    In this encoding, column j of X_split has a 1 for every observation whose
    factor level is >= j.  Equivalently, X_split[:, j] = sum(X_onehot[:, j:],
    axis=1).

    Parameters
    ----------
    X : NDArray[np.float64]
        One-hot design matrix of shape (n_samples, n_levels). Each row sums
        to at most 1.

    Returns
    -------
    NDArray[np.float64]
        Transformed design matrix of shape (n_samples, n_levels).

    Examples
    --------
    >>> X = np.eye(3)  # identity = one-hot for 3 levels
    >>> apply_split_coding(X)
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]])
    """
    # X_split[:, j] = sum(X[:, j:], axis=1)
    # Vectorised: reverse cumsum along axis=1, then reverse back
    return np.cumsum(X[:, ::-1], axis=1)[:, ::-1].astype(np.float64)
