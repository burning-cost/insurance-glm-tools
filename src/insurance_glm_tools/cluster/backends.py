"""
Statsmodels adapter for the unpenalised GLM refit (Step 3 of R2VF).

After fusion determines merged groups, we fit an ordinary (unpenalised) GLM
on the merged factor encoding. This removes the shrinkage bias introduced by
the lasso and gives us proper MLE estimates for inference.

The design matrix here uses simple group-indicator columns — one dummy per
merged group per factor, with one dropped for identifiability. The intercept
is always included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from numpy.typing import NDArray


def build_refit_matrix(
    X: pd.DataFrame,
    factor_group_maps: dict[str, dict],
    ordinal_factors: list[str],
    drop_first: bool = True,
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Build the design matrix for the unpenalised refit GLM.

    For each factor, creates one-hot dummy columns for the merged groups,
    with the first group dropped for identifiability. Adds an intercept.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature data.
    factor_group_maps : dict[str, dict]
        Maps factor name → dict of (original_level → group_label).
    ordinal_factors : list[str]
        Factor names to include.
    drop_first : bool
        Whether to drop first group dummy (default True for identifiability).

    Returns
    -------
    X_refit : NDArray[np.float64]
        Design matrix with intercept as first column.
    col_names : list[str]
        Column names.
    """
    cols = []
    col_names: list[str] = ["intercept"]

    for factor in ordinal_factors:
        level_to_group = factor_group_maps[factor]
        groups = X[factor].map(level_to_group)
        unique_groups = sorted(groups.unique())

        start = 1 if drop_first else 0
        for g in unique_groups[start:]:
            cols.append((groups == g).astype(np.float64).values)
            col_names.append(f"{factor}_g{g}")

    intercept = np.ones(len(X), dtype=np.float64)
    matrices = [intercept.reshape(-1, 1)] + [c.reshape(-1, 1) for c in cols]
    X_refit = np.hstack(matrices)
    return X_refit, col_names


def fit_poisson_refit(
    X_refit: NDArray[np.float64],
    y: NDArray[np.float64],
    exposure: NDArray[np.float64] | None = None,
) -> GLMResults:
    """
    Fit an unpenalised Poisson GLM with log link.

    Parameters
    ----------
    X_refit : NDArray[np.float64]
        Design matrix (with intercept).
    y : NDArray[np.float64]
        Response (claim counts).
    exposure : NDArray[np.float64] or None
        Exposure vector. Used as offset: log(exposure).

    Returns
    -------
    GLMResults
        Fitted statsmodels GLM result object.
    """
    offset = np.log(exposure) if exposure is not None else None
    model = sm.GLM(
        y,
        X_refit,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset,
    )
    return model.fit()


def fit_gamma_refit(
    X_refit: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> GLMResults:
    """
    Fit an unpenalised Gamma GLM with log link.

    Parameters
    ----------
    X_refit : NDArray[np.float64]
        Design matrix (with intercept).
    y : NDArray[np.float64]
        Response (claim severity / average cost).
    weights : NDArray[np.float64] or None
        Frequency weights (e.g. claim counts).

    Returns
    -------
    GLMResults
        Fitted statsmodels GLM result object.
    """
    model = sm.GLM(
        y,
        X_refit,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=weights,
    )
    return model.fit()


def extract_group_coefficients(
    result: GLMResults,
    col_names: list[str],
    factor: str,
    n_groups: int,
) -> NDArray[np.float64]:
    """
    Extract per-group coefficients for a specific factor from refit results.

    Group 0 is the reference (coefficient = 0.0 relative to intercept,
    but we return the absolute log-rate for that group).

    Parameters
    ----------
    result : GLMResults
        Fitted statsmodels result.
    col_names : list[str]
        Column names used in build_refit_matrix.
    factor : str
        Factor name to extract.
    n_groups : int
        Total number of merged groups for this factor.

    Returns
    -------
    NDArray[np.float64]
        Coefficient for each group (length n_groups).
        Group 0 = 0.0 (reference); others are log-relativities vs group 0.
    """
    coefs = np.zeros(n_groups, dtype=np.float64)
    for g in range(1, n_groups):
        col = f"{factor}_g{g}"
        if col in col_names:
            idx = col_names.index(col)
            coefs[g] = result.params[idx]
    return coefs
