"""
FactorClusterer: the main interface for R2VF ordinal factor clustering.

Workflow:
  1. Build split-coded design matrix for all ordinal factors.
  2. Fit penalised GLM (IRLS with L1 Lasso step) over a grid of lambda values.
  3. For each lambda, decode merged groups from the δ solution.
  4. Compute BIC; select best lambda.
  5. Apply min_exposure constraint.
  6. Fit unpenalised refit GLM (statsmodels) on merged encoding.
  7. Expose LevelMap objects for each factor.

Implementation note on penalisation:
  sklearn's PoissonRegressor uses L2 (Ridge) penalty, which does NOT shrink
  coefficients to exactly zero. The fused lasso requires L1. We implement
  Poisson/Gamma GLM with L1 penalty via IRLS: at each Newton step we solve a
  weighted Lasso on the working response, using sklearn's coordinate descent.

  Crucially, δ₁ = β₁ (the anchor/reference level for each factor) is NOT
  penalised — only δ₂,...,δ_K (adjacent differences) receive the L1 penalty.
  The IRLS inner step partitions columns into penalised (δ₂+) and unpenalised
  (δ₁), updating each group appropriately.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Lasso, ElasticNet
from statsmodels.genmod.generalized_linear_model import GLMResults

from .penalties import (
    build_full_split_matrix,
    delta_to_beta,
    identify_merged_groups,
)
from .constraints import enforce_min_exposure, relabel_groups_contiguous
from .backends import (
    build_refit_matrix,
    fit_poisson_refit,
    fit_gamma_refit,
    extract_group_coefficients,
)
from .diagnostics import (
    DiagnosticPath,
    poisson_log_likelihood,
    gamma_log_likelihood,
    poisson_deviance,
    gamma_deviance,
    compute_bic,
)
from .level_map import LevelMap, build_level_map


Family = Literal["poisson", "gamma"]

_LASSO_TOL = 1e-6


def _build_penalised_mask(n_col_per_factor: list[int]) -> NDArray[np.bool_]:
    """
    Build a boolean mask of length sum(n_col_per_factor).

    True  => column is penalised (δ₂, δ₃, ..., δ_K for each factor).
    False => column is unpenalised (δ₁, the anchor for each factor).

    For a factor with K levels the split-coded block has K columns.
    Column 0 within the block is δ₁ and must be excluded from the L1 penalty.
    """
    mask = []
    for k in n_col_per_factor:
        mask.append(False)       # δ₁ unpenalised
        mask.extend([True] * (k - 1))  # δ₂,...,δ_K penalised
    return np.array(mask, dtype=bool)


def _update_unpenalised_coef(
    X_unpen: NDArray[np.float64],
    z_adj: NDArray[np.float64],
    W: NDArray[np.float64],
    pen_coef_contrib: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Update unpenalised coefficients via weighted OLS (no penalty).

    Given working response z_adj and current penalised contribution
    pen_coef_contrib = X_pen @ coef_pen, solve:

        argmin_c  sum_i W_i (z_adj_i - pen_coef_contrib_i - X_unpen_i @ c)^2

    Returns the WLS solution.
    """
    residual_for_unpen = z_adj - pen_coef_contrib
    # Weighted normal equations: (X_unpen.T @ diag(W) @ X_unpen) c = X_unpen.T @ diag(W) @ r
    XtW = X_unpen.T * W[np.newaxis, :]  # (p_unpen, n)
    XtWX = XtW @ X_unpen               # (p_unpen, p_unpen)
    XtWr = XtW @ residual_for_unpen    # (p_unpen,)
    # Add tiny ridge for numerical stability
    XtWX += 1e-10 * np.eye(XtWX.shape[0])
    return np.linalg.solve(XtWX, XtWr)


def _poisson_irls_lasso(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    exposure: NDArray[np.float64] | None,
    alpha: float,
    n_col_per_factor: list[int],
    max_iter_irls: int = 20,
    max_iter_lasso: int = 2000,
    tol_irls: float = 1e-5,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """
    Fit Poisson GLM with L1 lasso penalty via IRLS.

    At each IRLS step, the penalised Poisson GLM reduces to a weighted Lasso
    on the working response. We use sklearn's Lasso (coordinate descent) for
    the inner step.

    The L1 penalty is applied only to δ₂,...,δ_K for each factor (the
    adjacent differences). δ₁ (the anchor level) is unpenalised and its
    coefficient is updated via weighted OLS at each step.

    Parameters
    ----------
    X : array (n, p)
        Split-coded design matrix (no intercept column — we fit intercept separately).
    y : array (n,)
        Claim counts.
    exposure : array (n,) or None
        Exposure (years at risk).
    alpha : float
        L1 penalty strength.
    n_col_per_factor : list[int]
        Number of split-coded columns per factor.  Used to identify which
        columns are δ₁ anchors (unpenalised) vs δ₂+ differences (penalised).
    max_iter_irls : int
        Maximum IRLS iterations.
    max_iter_lasso : int
        Maximum Lasso iterations per IRLS step.
    tol_irls : float
        Convergence tolerance on coefficient change.

    Returns
    -------
    coef : array (p,)
        Fitted split-coded coefficients (excluding intercept).
    intercept : float
        Fitted intercept.
    mu : array (n,)
        Fitted means (E[y]).
    """
    n, p = X.shape
    if exposure is None:
        exposure = np.ones(n)

    # Identify penalised vs unpenalised columns
    pen_mask = _build_penalised_mask(n_col_per_factor)
    unpen_mask = ~pen_mask

    X_pen = X[:, pen_mask]     # columns to be L1-penalised (δ₂+)
    X_unpen = X[:, unpen_mask] # columns to be updated freely (δ₁ anchors)

    # Initialise: use log(y+0.5) - log(exposure) as starting linear predictor
    y_safe = np.maximum(y, 0.5)
    mu = np.maximum(y_safe, 0.1)

    coef = np.zeros(p)
    # P1-3 fix: use log of total rate, not log of ratio of means
    intercept = float(np.log(y.sum() / exposure.sum()))

    for _ in range(max_iter_irls):
        # IRLS working response and weights
        # For Poisson: W = mu, z = eta + (y - mu) / mu
        W = mu  # working weights
        eta_full = np.log(np.maximum(mu, 1e-10))
        z = eta_full + (y - mu) / mu  # working response (log scale)

        # Working response relative to current intercept
        z_adj = z - intercept

        sqrt_W = np.sqrt(W)

        # --- Penalised step: Lasso on penalised columns only ---
        X_pen_w = X_pen * sqrt_W[:, np.newaxis]
        # Remove unpenalised contribution from working response for the Lasso
        unpen_contrib = X_unpen @ coef[unpen_mask]
        z_adj_for_pen = z_adj - unpen_contrib
        z_pen_w = z_adj_for_pen * sqrt_W

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = Lasso(
                alpha=alpha,
                fit_intercept=False,
                max_iter=max_iter_lasso,
                tol=_LASSO_TOL,
                warm_start=False,
            )
            lasso.fit(X_pen_w, z_pen_w)

        new_coef_pen = lasso.coef_

        # --- Unpenalised step: WLS on anchor columns (δ₁ per factor) ---
        pen_contrib = X_pen @ new_coef_pen
        new_coef_unpen = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)

        # Assemble full coefficient vector
        new_coef = np.zeros(p)
        new_coef[pen_mask] = new_coef_pen
        new_coef[unpen_mask] = new_coef_unpen

        # Update intercept: weighted mean of residuals
        residual = z - X @ new_coef
        new_intercept = float(np.average(residual, weights=W))

        # Update eta and mu
        eta = new_intercept + X @ new_coef
        eta = np.clip(eta, -20, 20)
        mu = exposure * np.exp(eta)
        mu = np.maximum(mu, 1e-10)

        # Check convergence
        if np.max(np.abs(new_coef - coef)) < tol_irls:
            coef = new_coef
            intercept = new_intercept
            break
        coef = new_coef
        intercept = new_intercept

    return coef, intercept, mu


def _gamma_irls_lasso(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
    alpha: float,
    n_col_per_factor: list[int],
    max_iter_irls: int = 20,
    max_iter_lasso: int = 2000,
    tol_irls: float = 1e-5,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """
    Fit Gamma GLM with log link and L1 lasso penalty via IRLS.

    For Gamma with log link: V(mu)*g'(mu)^2 = mu^2*(1/mu)^2 = 1, so W = obs_weights
    (frequency weights). Working response: z = eta + (y - mu) / mu.

    The L1 penalty is applied only to δ₂,...,δ_K for each factor.
    δ₁ anchors are unpenalised and updated via WLS.

    Parameters
    ----------
    X : array (n, p)
    y : array (n,)
        Claim severities.
    weights : array (n,) or None
        Frequency weights (claim counts).
    alpha : float
        L1 penalty strength.
    n_col_per_factor : list[int]
        Number of split-coded columns per factor.

    Returns
    -------
    coef : array (p,)
    intercept : float
    mu : array (n,)
    """
    n, p = X.shape
    obs_weights = weights if weights is not None else np.ones(n)

    # Identify penalised vs unpenalised columns
    pen_mask = _build_penalised_mask(n_col_per_factor)
    unpen_mask = ~pen_mask

    X_pen = X[:, pen_mask]
    X_unpen = X[:, unpen_mask]

    # Initialise
    mu = np.maximum(y, 1e-6)

    coef = np.zeros(p)
    intercept = float(np.log(np.average(y, weights=obs_weights)))

    for _ in range(max_iter_irls):
        # IRLS: for Gamma with log link, W = obs_weights (canonical weight is 1)
        eta_full = np.log(np.maximum(mu, 1e-10))
        z = eta_full + (y - mu) / mu  # working response

        W = obs_weights

        z_adj = z - intercept
        sqrt_W = np.sqrt(W)

        # --- Penalised step ---
        X_pen_w = X_pen * sqrt_W[:, np.newaxis]
        unpen_contrib = X_unpen @ coef[unpen_mask]
        z_adj_for_pen = z_adj - unpen_contrib
        z_pen_w = z_adj_for_pen * sqrt_W

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = Lasso(
                alpha=alpha,
                fit_intercept=False,
                max_iter=max_iter_lasso,
                tol=_LASSO_TOL,
                warm_start=False,
            )
            lasso.fit(X_pen_w, z_pen_w)

        new_coef_pen = lasso.coef_

        # --- Unpenalised step ---
        pen_contrib = X_pen @ new_coef_pen
        new_coef_unpen = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)

        new_coef = np.zeros(p)
        new_coef[pen_mask] = new_coef_pen
        new_coef[unpen_mask] = new_coef_unpen

        residual = z - X @ new_coef
        new_intercept = float(np.average(residual, weights=W))

        eta = new_intercept + X @ new_coef
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)

        if np.max(np.abs(new_coef - coef)) < tol_irls:
            coef = new_coef
            intercept = new_intercept
            break
        coef = new_coef
        intercept = new_intercept

    return coef, intercept, mu


class FactorClusterer:
    """
    Automated GLM factor-level clustering via R2VF (Step 2 + Step 3).

    Fits a fused lasso on the split-coded design matrix to merge adjacent
    ordinal factor levels, then refits an unpenalised GLM on the merged
    encoding.

    The penalised step uses IRLS with a Lasso inner solver, ensuring true
    L1 (not Ridge) penalisation — necessary for coefficients to shrink to
    exactly zero and enable clean level fusion.

    The L1 penalty is applied only to δ₂,...,δ_K (adjacent differences)
    for each factor.  δ₁ (the anchor level) is always unpenalised.

    Parameters
    ----------
    family : {'poisson', 'gamma'}
        Response distribution.
    method : str
        Algorithm. Currently only 'r2vf' is supported.
    lambda_ : float or 'bic'
        Regularisation strength. If 'bic', selected by BIC over a grid.
    n_lambda : int
        Grid size for BIC search. Default 50.
    min_exposure : float
        Minimum exposure per merged group. Default 0.0 (disabled).
    tol : float
        Zero-threshold for delta coefficients (merged). Default 1e-8.
    max_iter_irls : int
        IRLS iterations for each penalised fit. Default 20.
    random_state : int or None
        Unused; reserved for future stochastic extensions.

    Examples
    --------
    >>> fc = FactorClusterer(family='poisson', lambda_='bic', min_exposure=500)
    >>> fc.fit(X, y, exposure=exposure, ordinal_factors=['vehicle_age', 'ncd_years'])
    >>> X_merged = fc.transform(X)
    >>> result = fc.refit_glm(X_merged, y, exposure=exposure)
    >>> lm = fc.level_map('vehicle_age')
    >>> print(lm.to_df())
    """

    def __init__(
        self,
        family: Family = "poisson",
        method: str = "r2vf",
        lambda_: float | Literal["bic"] = "bic",
        n_lambda: int = 50,
        min_exposure: float = 0.0,
        tol: float = 1e-8,
        max_iter_irls: int = 20,
        random_state: int | None = None,
    ) -> None:
        if family not in ("poisson", "gamma"):
            raise ValueError(f"family must be 'poisson' or 'gamma', got {family!r}")
        if method != "r2vf":
            raise ValueError(f"method must be 'r2vf', got {method!r}")
        if n_lambda < 2:
            raise ValueError("n_lambda must be at least 2")

        self.family = family
        self.method = method
        self.lambda_ = lambda_
        self.n_lambda = n_lambda
        self.min_exposure = min_exposure
        self.tol = tol
        self.max_iter_irls = max_iter_irls
        self.random_state = random_state

        # Set after fit()
        self._ordinal_factors: list[str] = []
        self._ordered_levels_map: dict[str, list] = {}
        self._level_maps: dict[str, LevelMap] = {}
        self._factor_group_maps: dict[str, dict] = {}
        self._diagnostic_path: DiagnosticPath | None = None
        self._best_lambda: float | None = None
        self._refit_result: GLMResults | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | NDArray[np.float64],
        exposure: pd.Series | NDArray[np.float64] | None = None,
        ordinal_factors: list[str] | None = None,
    ) -> "FactorClusterer":
        """
        Fit the R2VF clustering algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data. Must contain all columns in ordinal_factors.
        y : array-like
            Response (claim counts for Poisson, severity for Gamma).
        exposure : array-like or None
            Exposure (years at risk for Poisson; claim counts for Gamma).
        ordinal_factors : list[str] or None
            Factor columns to cluster. If None, uses all columns.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        if ordinal_factors is None:
            ordinal_factors = list(X.columns)
        self._ordinal_factors = ordinal_factors

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=np.float64)

        for col in ordinal_factors:
            if col not in X.columns:
                raise ValueError(f"Column {col!r} not found in X")

        # Build ordered levels for each factor
        self._ordered_levels_map = {}
        for col in ordinal_factors:
            self._ordered_levels_map[col] = sorted(X[col].unique())

        # Build split-coded design matrix
        X_split, _ = build_full_split_matrix(
            X, ordinal_factors, self._ordered_levels_map
        )

        n_col_per_factor = [len(self._ordered_levels_map[f]) for f in ordinal_factors]

        # Select and fit at best lambda
        if self.lambda_ == "bic":
            self._diagnostic_path = self._fit_lambda_path(
                X_split, y, exposure, n_col_per_factor, n
            )
            self._best_lambda = self._diagnostic_path.best_lambda
        else:
            self._best_lambda = float(self.lambda_)
            self._diagnostic_path = None

        coef, intercept, mu = self._fit_at_lambda(
            X_split, y, exposure, self._best_lambda, n_col_per_factor
        )

        self._decode_groups(X, coef, n_col_per_factor, exposure)

        self._factor_group_maps = {
            factor: self._level_maps[factor].level_to_group
            for factor in ordinal_factors
        }

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Recode factor columns to merged group labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data with original factor values.

        Returns
        -------
        pd.DataFrame
            Copy of X with ordinal factor columns replaced by group labels.
        """
        self._check_fitted()
        X_out = X.copy()
        for factor in self._ordinal_factors:
            X_out[factor] = self._level_maps[factor].apply(X[factor])
        return X_out

    def refit_glm(
        self,
        X: pd.DataFrame,
        y: pd.Series | NDArray[np.float64],
        exposure: pd.Series | NDArray[np.float64] | None = None,
    ) -> GLMResults:
        """
        Fit an unpenalised GLM on the merged factor encoding (Step 3).

        Parameters
        ----------
        X : pd.DataFrame
            Feature data with original factor values.
        y : array-like
            Response variable.
        exposure : array-like or None
            Exposure vector.

        Returns
        -------
        GLMResults
            Fitted statsmodels GLM result.
        """
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)

        X_refit, col_names = build_refit_matrix(
            X, self._factor_group_maps, self._ordinal_factors
        )

        if self.family == "poisson":
            exposure_arr = (
                np.asarray(exposure, dtype=np.float64)
                if exposure is not None
                else np.ones(len(y))
            )
            result = fit_poisson_refit(X_refit, y, exposure=exposure_arr)
        else:
            weights = (
                np.asarray(exposure, dtype=np.float64)
                if exposure is not None
                else None
            )
            result = fit_gamma_refit(X_refit, y, weights=weights)

        self._refit_result = result
        self._update_level_map_coefficients(result, col_names)
        return result

    def level_map(self, factor: str) -> LevelMap:
        """
        Return the LevelMap for a specific factor.

        Parameters
        ----------
        factor : str
            Factor column name.

        Returns
        -------
        LevelMap
        """
        self._check_fitted()
        if factor not in self._level_maps:
            raise ValueError(f"Factor {factor!r} not in fitted factors")
        return self._level_maps[factor]

    @property
    def diagnostic_path(self) -> DiagnosticPath | None:
        """Diagnostic path from BIC lambda selection. None if lambda was fixed."""
        return self._diagnostic_path

    @property
    def best_lambda(self) -> float | None:
        """Selected regularisation strength."""
        return self._best_lambda

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_at_lambda(
        self,
        X_split: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None,
        alpha: float,
        n_col_per_factor: list[int],
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
        """
        Fit penalised GLM at one alpha via IRLS + Lasso.

        Returns (coef, intercept, mu).
        """
        if self.family == "poisson":
            return _poisson_irls_lasso(
                X_split, y, exposure, alpha, n_col_per_factor,
                max_iter_irls=self.max_iter_irls,
            )
        else:
            return _gamma_irls_lasso(
                X_split, y, exposure, alpha, n_col_per_factor,
                max_iter_irls=self.max_iter_irls,
            )

    def _estimate_lambda_max(
        self,
        X_split: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None,
        n_col_per_factor: list[int],
    ) -> float:
        """
        Find the smallest lambda that collapses all factors to a single group.

        Binary search: start large, halve until some factor has > 1 group.
        """
        # Start with a candidate lambda_max via gradient norm heuristic.
        # For GLM at null model: gradient w.r.t. delta_j ≈ X_j^T (y - mu_null).
        # The L1 KKT condition: lambda_max = max |gradient|.
        if exposure is not None:
            mu_null = exposure * (y.sum() / exposure.sum())
        else:
            mu_null = np.full(len(y), y.mean())

        residual = y - mu_null
        grad = np.abs(X_split.T @ residual) / len(y)

        # Exclude first column per factor (delta_1 is not penalised by fused lasso)
        penalised_grad = np.concatenate([
            grad[offset + 1 : offset + n_cols]
            for offset, n_cols in zip(
                np.cumsum([0] + n_col_per_factor[:-1]),
                n_col_per_factor,
            )
            if n_cols > 1
        ])

        if len(penalised_grad) == 0:
            return 1.0

        lambda_max = float(penalised_grad.max()) * 10  # generous upper bound
        return max(lambda_max, 0.1)

    def _fit_lambda_path(
        self,
        X_split: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None,
        n_col_per_factor: list[int],
        n: int,
    ) -> DiagnosticPath:
        """Fit over full lambda grid and compute BIC at each point."""
        lambda_max = self._estimate_lambda_max(X_split, y, exposure, n_col_per_factor)
        lambda_min = lambda_max / 1000.0

        lambdas = np.logspace(
            np.log10(lambda_max), np.log10(lambda_min), self.n_lambda
        )  # descending: large lambda first (fewest groups)

        bic_values = np.full(self.n_lambda, np.inf)
        deviance_values = np.full(self.n_lambda, np.inf)
        n_groups_values = np.zeros(self.n_lambda, dtype=np.int32)

        for i, alpha in enumerate(lambdas):
            try:
                coef, intercept, mu = self._fit_at_lambda(
                    X_split, y, exposure, alpha, n_col_per_factor
                )
                k_eff = self._count_effective_groups(coef, n_col_per_factor)

                if self.family == "poisson":
                    ll = poisson_log_likelihood(y, mu)
                    dev = poisson_deviance(y, mu)
                else:
                    ll = gamma_log_likelihood(y, mu, weights=exposure)
                    dev = gamma_deviance(y, mu, weights=exposure)

                bic_values[i] = compute_bic(ll, k_eff, n)
                deviance_values[i] = dev
                n_groups_values[i] = k_eff
            except Exception as exc:
                warnings.warn(
                    f"Lambda path: fit failed at alpha={alpha:.4g}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        best_idx = int(np.argmin(bic_values))

        return DiagnosticPath(
            lambdas=lambdas,
            bic=bic_values,
            deviance=deviance_values,
            n_groups=n_groups_values,
            best_idx=best_idx,
        )

    def _count_effective_groups(
        self,
        coef: NDArray[np.float64],
        n_col_per_factor: list[int],
    ) -> int:
        """
        Count total effective free parameters across all factors.

        For a factor with G merged groups, the dummy-coded GLM has G-1 free
        parameters (reference level is absorbed into the intercept). We therefore
        add G-1 per factor, plus 1 for the intercept itself.
        """
        k_eff = 1  # intercept
        offset = 0
        for n_cols in n_col_per_factor:
            delta = coef[offset : offset + n_cols]
            groups = identify_merged_groups(delta, tol=self.tol)
            n_groups = len(np.unique(groups))
            k_eff += n_groups - 1  # P1-1 fix: G-1 free params per factor
            offset += n_cols
        return k_eff

    def _decode_groups(
        self,
        X: pd.DataFrame,
        coef: NDArray[np.float64],
        n_col_per_factor: list[int],
        exposure: NDArray[np.float64] | None,
    ) -> None:
        """Decode merged groups, apply min_exposure, build LevelMaps."""
        offset = 0
        for factor, n_cols in zip(self._ordinal_factors, n_col_per_factor):
            levels = self._ordered_levels_map[factor]
            delta = coef[offset : offset + n_cols]
            beta = delta_to_beta(delta)

            groups = identify_merged_groups(delta, tol=self.tol)

            if exposure is not None:
                level_to_exp = (
                    pd.DataFrame({"level": X[factor].values, "exp": exposure})
                    .groupby("level")["exp"]
                    .sum()
                )
                exposure_by_level = np.array(
                    [float(level_to_exp.get(lv, 0.0)) for lv in levels]
                )
            else:
                level_counts = X[factor].value_counts()
                exposure_by_level = np.array(
                    [float(level_counts.get(lv, 0)) for lv in levels]
                )

            if self.min_exposure > 0:
                groups = enforce_min_exposure(
                    groups=groups,
                    exposure_by_level=exposure_by_level,
                    coefficients_by_level=beta,
                    min_exposure=self.min_exposure,
                )

            groups = relabel_groups_contiguous(groups)

            n_groups = len(np.unique(groups))
            coef_per_group = np.zeros(n_groups, dtype=np.float64)
            for g in range(n_groups):
                mask = groups == g
                w = exposure_by_level[mask]
                if w.sum() > 0:
                    coef_per_group[g] = np.average(beta[mask], weights=w)
                else:
                    coef_per_group[g] = float(np.mean(beta[mask]))

            self._level_maps[factor] = build_level_map(
                factor=factor,
                ordered_levels=levels,
                groups=groups,
                coefficients=coef_per_group,
                exposure_by_level=exposure_by_level,
            )

            offset += n_cols

    def _update_level_map_coefficients(
        self,
        result: GLMResults,
        col_names: list[str],
    ) -> None:
        """Update LevelMap coefficients from the unpenalised refit result."""
        for factor in self._ordinal_factors:
            lm = self._level_maps[factor]
            refit_coefs = extract_group_coefficients(
                result, col_names, factor, lm.n_groups
            )
            self._level_maps[factor] = LevelMap(
                factor=factor,
                ordered_levels=lm.ordered_levels,
                groups=lm.groups,
                coefficients=tuple(float(c) for c in refit_coefs),
                group_exposure=lm.group_exposure,
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method")
