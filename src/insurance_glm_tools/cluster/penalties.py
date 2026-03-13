"""
Split-coding reparameterisation and fusion logic for ordinal factor clustering.

The R2VF algorithm (Ben Dror 2025) exploits the fact that the fused lasso
penalty on adjacent differences of ordinal factor coefficients reduces to a
standard L1 penalty after a change of basis — the "split-coding" transform.

For a factor with K ordered levels and coefficient vector β = (β₁, …, βK):
  δ₁ = β₁
  δⱼ = βⱼ - βⱼ₋₁  for j = 2, …, K

Then βⱼ = Σ_{s≤j} δₛ (cumulative sum of δ), and the fused lasso penalty
λ·Σⱼ|βⱼ - βⱼ₋₁| = λ·Σ_{j≥2}|δⱼ|, which is a plain L1 penalty on δ₂,...,δK.

The design matrix is constructed so that a column for level j has 1s for
all observations with factor level ≥ j, and 0s otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def make_split_coded_matrix(
    factor_series: pd.Series,
    ordered_levels: list,
) -> NDArray[np.float64]:
    """
    Build the split-coded design matrix for an ordinal factor.

    For K levels, produces K columns. Column j has value 1 for every
    observation whose level is at position ≥ j in ordered_levels.

    Parameters
    ----------
    factor_series : pd.Series
        The factor column (values must all appear in ordered_levels).
    ordered_levels : list
        The ordered levels, lowest to highest (e.g. [0, 1, 2, ..., 15]).

    Returns
    -------
    NDArray[np.float64]
        Array of shape (n_obs, K).
    """
    K = len(ordered_levels)
    n = len(factor_series)
    level_to_idx: dict = {lv: i for i, lv in enumerate(ordered_levels)}

    obs_idx = np.array([level_to_idx[v] for v in factor_series], dtype=np.int32)

    # Column j has 1 for obs_idx >= j
    # Vectorised: X[i, j] = 1 if obs_idx[i] >= j
    X = np.zeros((n, K), dtype=np.float64)
    for j in range(K):
        X[:, j] = (obs_idx >= j).astype(np.float64)

    return X


def delta_to_beta(delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert split-coded coefficients δ back to level coefficients β.

    β = cumsum(δ)

    Parameters
    ----------
    delta : NDArray[np.float64]
        Array of length K (δ₁, δ₂, ..., δK).

    Returns
    -------
    NDArray[np.float64]
        Array of length K (β₁, β₂, ..., βK).
    """
    return np.cumsum(delta)


def beta_to_delta(beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert level coefficients β to split-coded differences δ.

    δ₁ = β₁, δⱼ = βⱼ - βⱼ₋₁ for j ≥ 2.

    Parameters
    ----------
    beta : NDArray[np.float64]
        Array of length K.

    Returns
    -------
    NDArray[np.float64]
        Array of length K.
    """
    delta = np.empty_like(beta)
    delta[0] = beta[0]
    delta[1:] = np.diff(beta)
    return delta


def identify_merged_groups(
    delta: NDArray[np.float64],
    tol: float = 1e-8,
) -> NDArray[np.int32]:
    """
    Identify which levels are merged based on the δ solution.

    When δⱼ ≈ 0, levels j-1 and j share the same coefficient and are merged.
    Returns an integer group label for each level, starting from 0.

    Parameters
    ----------
    delta : NDArray[np.float64]
        Split-coded coefficient vector of length K.
    tol : float
        Threshold below which |δⱼ| is treated as zero (merged).

    Returns
    -------
    NDArray[np.int32]
        Group labels of length K, integers 0, 1, 2, ...
    """
    K = len(delta)
    groups = np.zeros(K, dtype=np.int32)
    current_group = 0
    for j in range(1, K):
        if abs(delta[j]) > tol:
            current_group += 1
        groups[j] = current_group
    return groups


def split_coded_columns_for_factor(
    factor_col: str,
    ordered_levels: list,
) -> list[str]:
    """
    Generate canonical column names for split-coded columns.

    Parameters
    ----------
    factor_col : str
        Name of the original factor column.
    ordered_levels : list
        Ordered levels.

    Returns
    -------
    list[str]
        Column names of length K.
    """
    return [f"{factor_col}__split_{j}" for j in range(len(ordered_levels))]


def build_full_split_matrix(
    X: pd.DataFrame,
    ordinal_factors: list[str],
    ordered_levels_map: dict[str, list],
    other_cols: list[str] | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Construct the full design matrix including split-coded ordinal factors
    and pass-through columns for any other predictors.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame.
    ordinal_factors : list[str]
        Names of columns to split-code.
    ordered_levels_map : dict[str, list]
        Maps each factor name to its ordered level list.
    other_cols : list[str] or None
        Additional columns to include as-is (not penalised).

    Returns
    -------
    X_split : NDArray[np.float64]
        Design matrix.
    col_names : list[str]
        Column names matching columns of X_split.
    """
    blocks = []
    col_names: list[str] = []

    for factor in ordinal_factors:
        levels = ordered_levels_map[factor]
        block = make_split_coded_matrix(X[factor], levels)
        blocks.append(block)
        col_names.extend(split_coded_columns_for_factor(factor, levels))

    if other_cols:
        for col in other_cols:
            blocks.append(X[col].values.reshape(-1, 1).astype(np.float64))
            col_names.append(col)

    X_split = np.hstack(blocks) if blocks else np.empty((len(X), 0))
    return X_split, col_names
