"""
BIC computation and diagnostic path tracking for lambda selection.

BIC(λ) = -2·ℓ(β̂(λ)) + K_eff(λ)·log(n)

where ℓ is the log-likelihood evaluated at the fitted coefficients and K_eff
is the effective degrees of freedom, computed as:

    K_eff = 1 + sum(G_f - 1 for each factor f)

where G_f is the number of distinct merged groups for factor f, and the leading
1 accounts for the intercept. Each factor contributes G_f - 1 free parameters
because one group is absorbed into the intercept as the reference level.
When two adjacent levels are fused, K_eff decreases by one; BIC trades off
fit against parsimony.

We fit over a log-spaced grid of 50 lambdas and return the full diagnostic
path so practitioners can inspect the tradeoff themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
