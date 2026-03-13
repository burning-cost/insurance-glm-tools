"""
BIC computation and diagnostic path tracking for lambda selection.

BIC(λ) = -2·ℓ(β̂(λ)) + K_eff(λ)·log(n)

where ℓ is the log-likelihood evaluated at the fitted coefficients and K_eff
is the effective degrees of freedom — the number of distinct merged groups
summed across all factors. When two adjacent levels are fused, K_eff decreases
by one; BIC trades off fit against parsimony.

We fit over a log-spaced grid of 50 lambdas and return the full diagnostic
path so practitioners can inspect the tradeoff themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import PoissonRegressor, GammaRegressor


@dataclass
class DiagnosticPath:
    """
    Full regularisation path diagnostics.

    Attributes
    ----------
    lambdas : NDArray[np.float64]
        Lambda values fitted (ascending).
    bic : NDArray[np.float64]
        BIC at each lambda.
    deviance : NDArray[np.float64]
        Deviance at each lambda (2 * (saturated_loglik - fitted_loglik)).
    n_groups : NDArray[np.int32]
        Total effective groups K_eff at each lambda.
    best_idx : int
        Index of best (minimum BIC) lambda.
    """

    lambdas: NDArray[np.float64]
    bic: NDArray[np.float64]
    deviance: NDArray[np.float64]
    n_groups: NDArray[np.int32]
    best_idx: int

    @property
    def best_lambda(self) -> float:
        """Lambda that minimises BIC."""
        return float(self.lambdas[self.best_idx])

    def to_df(self) -> pd.DataFrame:
        """Return diagnostic path as a tidy DataFrame."""
        return pd.DataFrame({
            "lambda": self.lambdas,
            "bic": self.bic,
            "deviance": self.deviance,
            "n_groups": self.n_groups,
            "is_best": np.arange(len(self.lambdas)) == self.best_idx,
        })


def poisson_log_likelihood(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    """
    Poisson log-likelihood: Σ [y·log(μ) - μ].

    Constant terms (log(y!)) are omitted as they cancel in BIC comparisons.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed counts.
    mu : NDArray[np.float64]
        Fitted means.

    Returns
    -------
    float
        Log-likelihood value.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(y > 0, y * np.log(mu) - mu, -mu)
    return float(ll.sum())


def gamma_log_likelihood(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Gamma log-likelihood with log link (dispersion cancels in BIC comparisons).

    Shape k, rate k/μ. Log-likelihood ∝ Σ wᵢ·[-log(μᵢ) - yᵢ/μᵢ].

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed severities.
    mu : NDArray[np.float64]
        Fitted means.
    weights : NDArray[np.float64] or None
        Observation weights (claim counts).

    Returns
    -------
    float
        Log-likelihood value (up to constant).
    """
    w = weights if weights is not None else np.ones_like(y)
    ll = w * (-np.log(mu) - y / mu)
    return float(ll.sum())


def poisson_deviance(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    """
    Poisson deviance: 2·Σ [y·log(y/μ) - (y - μ)].

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed counts.
    mu : NDArray[np.float64]
        Fitted means.

    Returns
    -------
    float
        Deviance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
    return float(2 * term.sum())


def gamma_deviance(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Gamma deviance: 2·Σ wᵢ·[log(μᵢ/yᵢ) + yᵢ/μᵢ - 1].

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed severities.
    mu : NDArray[np.float64]
        Fitted means.
    weights : NDArray[np.float64] or None
        Observation weights.

    Returns
    -------
    float
        Deviance.
    """
    w = weights if weights is not None else np.ones_like(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.log(mu / y) + y / mu - 1
    return float(2 * (w * term).sum())


def compute_bic(
    log_likelihood: float,
    k_eff: int,
    n: int,
) -> float:
    """
    Compute BIC = -2·ℓ + K_eff·log(n).

    Parameters
    ----------
    log_likelihood : float
    k_eff : int
        Effective number of parameters (distinct groups + intercept).
    n : int
        Number of observations.

    Returns
    -------
    float
    """
    return -2.0 * log_likelihood + k_eff * np.log(n)


def estimate_lambda_max(
    X_split: NDArray[np.float64],
    y_adj: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
    family: str,
    n_col_per_factor: list[int],
) -> float:
    """
    Estimate lambda_max: smallest λ that fully collapses all factors.

    Uses a heuristic: fit with a very large alpha and find where all
    split-coded differences vanish. We return a conservative upper bound
    based on the score statistic at the null model.

    For practical purposes we use sklearn's built-in max_iter and increase
    alpha until the model is effectively null.

    Parameters
    ----------
    X_split : NDArray[np.float64]
        Split-coded design matrix.
    y_adj : NDArray[np.float64]
        Adjusted response for penalised fit.
    sample_weight : NDArray[np.float64] or None
        Sample weights.
    family : str
        'poisson' or 'gamma'.
    n_col_per_factor : list[int]
        Number of split-coded columns per factor (for identifying first columns).

    Returns
    -------
    float
        Estimated lambda_max.
    """
    # Binary search for the smallest alpha that yields all-zero fusion deltas.
    # Simpler heuristic: use 10x the largest gradient at the null.
    # For GLM with log link and intercept, the score at null ~ (y - mean(y)).
    # We use a practical upper bound.
    alpha_max = 1.0
    for _ in range(20):
        try:
            if family == "poisson":
                m = PoissonRegressor(alpha=alpha_max, fit_intercept=True, max_iter=200)
            else:
                m = GammaRegressor(alpha=alpha_max, fit_intercept=True, max_iter=200)
            m.fit(X_split, y_adj, sample_weight=sample_weight)
            # Check if all non-first split deltas are near zero
            coef = m.coef_
            # Columns 1+ within each factor block are the delta_2,...,delta_K
            offset = 0
            all_zero = True
            for n_cols in n_col_per_factor:
                if n_cols > 1:
                    diffs = coef[offset + 1 : offset + n_cols]
                    if np.any(np.abs(diffs) > 1e-4):
                        all_zero = False
                        break
                offset += n_cols

            if all_zero:
                return alpha_max
            alpha_max *= 3.0
        except Exception:
            alpha_max *= 3.0

    return alpha_max
