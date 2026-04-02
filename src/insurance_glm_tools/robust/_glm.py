"""RobustMMDGLM: MMD-penalised GLM with L1 regularisation.

This implements the estimator from Kang & Kang (2026), arXiv:2602.21132.

The core idea: replace the log-likelihood with the Maximum Mean Discrepancy
(MMD) between the empirical distribution and the fitted model distribution.
MMD is a proper scoring rule but much more robust to heavy-tailed contamination
and outliers than log-likelihood, because the Gaussian kernel K_y(y1, y2)
down-weights large discrepancies geometrically rather than linearly.

Why this matters for insurance:
- Severity models are routinely contaminated by large claims that inflate
  MLE coefficient estimates and distort the rating factors.
- The MMD loss caps the influence of any single observation.
- The L1 penalty gives automatic feature selection — useful when you have
  50+ potential rating factors and need a parsimonious model.

Usage::

    from insurance_glm_tools.robust import RobustMMDGLM

    model = RobustMMDGLM(family="gamma", lambda_="cv", cv_folds=5)
    model.fit(X_train, y_train, exposure=exposure_train)
    print(model.relativities())

The ``exposure`` parameter applies as a multiplicative offset for Poisson/Gamma
(so mu_i = exposure_i * exp(x_i^T theta)). For Gaussian and Logistic families
it is ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union

from insurance_glm_tools.robust._families import Family, get_family
from insurance_glm_tools.robust._kernels import (
    median_bandwidth_y,
    median_bandwidth_x,
    total_mmd_loss,
)
from insurance_glm_tools.robust._admm import admm_fit
from insurance_glm_tools.robust._cv import (
    make_lambda_grid,
    cross_validate,
    select_lambda,
)


class RobustMMDGLM:
    """MMD-penalised GLM with L1 regularisation.

    Parameters
    ----------
    family : str
        Response distribution. One of "gaussian", "logistic", "poisson", "gamma".
    link : None
        Reserved for future custom link support. Currently ignored.
    lambda_ : float or "cv"
        L1 penalty strength. "cv" uses K-fold cross-validation to select from
        a log-spaced grid.
    lambda_grid : array-like or None
        Custom lambda grid for CV. If None and lambda_="cv", auto-constructed.
    h_y : float or "median"
        Response kernel bandwidth. "median" uses the median heuristic on y.
    h_x : float or "median"
        Covariate kernel bandwidth. "median" uses the median heuristic on X.
        (Currently used only in the bootstrap; reserved for weighted MMD variants.)
    rho : float
        ADMM augmented Lagrangian penalty parameter. Default 1.0 works for most
        problems; increase for ill-conditioned design matrices.
    adagrad_lr : float
        AdaGrad base learning rate for the theta-step. 0.1 is a good default.
    max_admm_iter : int
        Maximum outer ADMM iterations.
    max_inner_iter : int
        Maximum inner AdaGrad iterations per ADMM step.
    tol_primal : float
        ADMM primal residual convergence tolerance.
    tol_dual : float
        ADMM dual residual convergence tolerance.
    n_quadrature : int
        Number of Gauss-Laguerre nodes for Gamma family E[K_y] integral.
    cv_folds : int
        Number of CV folds when lambda_="cv".
    init : str
        Initialisation strategy. "lasso" uses sklearn Lasso/LogisticRegression
        for warm start. "zeros" starts from zero vector.
    random_state : int or None
        Random seed for median heuristic subsampling and CV fold assignment.
    """

    def __init__(
        self,
        family: str = "gaussian",
        link: None = None,
        lambda_: Union[float, str] = "cv",
        lambda_grid: Optional[np.ndarray] = None,
        h_y: Union[float, str] = "median",
        h_x: Union[float, str] = "median",
        rho: float = 1.0,
        adagrad_lr: float = 0.1,
        max_admm_iter: int = 200,
        max_inner_iter: int = 500,
        tol_primal: float = 1e-3,
        tol_dual: float = 1e-3,
        n_quadrature: int = 32,
        cv_folds: int = 5,
        init: str = "lasso",
        random_state: Optional[int] = None,
    ):
        self.family_name = family
        self.link = link
        self.lambda_ = lambda_
        self.lambda_grid = lambda_grid
        self.h_y = h_y
        self.h_x = h_x
        self.rho = rho
        self.adagrad_lr = adagrad_lr
        self.max_admm_iter = max_admm_iter
        self.max_inner_iter = max_inner_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.n_quadrature = n_quadrature
        self.cv_folds = cv_folds
        self.init = init
        self.random_state = random_state

        # Set after fit
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list] = None
        self._family: Optional[Family] = None
        self._h_y_value: Optional[float] = None
        self._h_x_value: Optional[float] = None
        self._lambda_value: Optional[float] = None
        self._cv_df: Optional[pd.DataFrame] = None
        self._admm_info: Optional[dict] = None
        self._n_features: Optional[int] = None
        self._exposure_log: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> "RobustMMDGLM":
        """Fit the model.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Design matrix. Should be pre-processed (encoded, scaled as needed).
        y : array-like, shape (n,)
            Response vector.
        exposure : array-like, shape (n,) or None
            Exposure (e.g. earned premium years). Applied as multiplicative offset
            for Poisson and Gamma families via log(exposure) added to eta.
        feature_names : list or None
            Names for each column of X. Used in relativities() output.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if y.shape[0] != n:
            raise ValueError(f"X has {n} rows but y has {len(y)} elements.")

        # Store exposure as log offset
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if np.any(exposure <= 0):
                raise ValueError("All exposure values must be positive.")
            self._exposure_log = np.log(exposure)
        else:
            self._exposure_log = None

        # Feature names
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"x{j}" for j in range(p)]
        self._n_features = p

        rng = np.random.default_rng(self.random_state)

        # Instantiate family
        family_kwargs = {}
        if self.family_name.lower() == "gamma":
            family_kwargs["n_quadrature"] = self.n_quadrature
        self._family = get_family(self.family_name, **family_kwargs)

        # Bandwidths
        if self.h_y == "median":
            self._h_y_value = median_bandwidth_y(y, rng=rng)
        else:
            self._h_y_value = float(self.h_y)

        if self.h_x == "median":
            self._h_x_value = median_bandwidth_x(X, rng=rng)
        else:
            self._h_x_value = float(self.h_x)

        # Initial theta
        theta_init = self._initialise_theta(X, y, p)

        # Update family dispersion parameters from init
        if hasattr(self._family, "update_sigma"):
            mu_init = self._family.mean(X @ theta_init + (self._exposure_log if self._exposure_log is not None else 0.0))
            self._family.update_sigma(y, mu_init)
        if hasattr(self._family, "update_alpha"):
            mu_init = self._family.mean(X @ theta_init + (self._exposure_log if self._exposure_log is not None else 0.0))
            self._family.update_alpha(y, mu_init)

        # Lambda selection
        if self.lambda_ == "cv":
            grid = self.lambda_grid
            if grid is None:
                grid = make_lambda_grid(
                    X, y, self._family, self._h_y_value, self._exposure_log
                )
            else:
                grid = np.asarray(grid)

            self._cv_df = cross_validate(
                X=X,
                y=y,
                family=self._family,
                h_y=self._h_y_value,
                exposure_log=self._exposure_log,
                lambda_grid=grid,
                cv_folds=self.cv_folds,
                rho=self.rho,
                adagrad_lr=self.adagrad_lr,
                max_admm_iter=self.max_admm_iter,
                max_inner_iter=self.max_inner_iter,
                tol_primal=self.tol_primal,
                tol_dual=self.tol_dual,
                theta_init=theta_init,
                random_state=self.random_state,
            )
            self._lambda_value = select_lambda(self._cv_df)
        else:
            self._lambda_value = float(self.lambda_)
            self._cv_df = None

        # Final fit on full data
        coef, info = admm_fit(
            theta_init=theta_init,
            X=X,
            y=y,
            family=self._family,
            h_y=self._h_y_value,
            lam=self._lambda_value,
            exposure_log=self._exposure_log,
            rho=self.rho,
            adagrad_lr=self.adagrad_lr,
            max_admm_iter=self.max_admm_iter,
            max_inner_iter=self.max_inner_iter,
            tol_primal=self.tol_primal,
            tol_dual=self.tol_dual,
        )
        self.coef_ = coef
        self._admm_info = info

        return self

    def _initialise_theta(
        self, X: np.ndarray, y: np.ndarray, p: int
    ) -> np.ndarray:
        """Warm-start theta from sklearn Lasso / LogisticRegression, or zeros."""
        if self.init != "lasso":
            return np.zeros(p)

        try:
            if self.family_name.lower() in ("gaussian",):
                from sklearn.linear_model import Lasso

                lasso = Lasso(alpha=0.01, max_iter=2000, fit_intercept=False)
                lasso.fit(X, y)
                return lasso.coef_.copy()

            elif self.family_name.lower() in ("logistic", "binomial"):
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(
                    penalty="l1", C=10.0, solver="liblinear", fit_intercept=False, max_iter=500
                )
                lr.fit(X, y)
                return lr.coef_.ravel().copy()

            elif self.family_name.lower() in ("poisson",):
                from sklearn.linear_model import PoissonRegressor

                # PoissonRegressor doesn't support L1 directly; use Ridge as proxy
                pr = PoissonRegressor(alpha=0.01, fit_intercept=False, max_iter=500)
                pr.fit(X, y)
                return pr.coef_.copy()

            elif self.family_name.lower() in ("gamma",):
                from sklearn.linear_model import GammaRegressor

                gr = GammaRegressor(alpha=0.01, fit_intercept=False, max_iter=500)
                gr.fit(X, y)
                return gr.coef_.copy()

        except Exception:
            pass

        return np.zeros(p)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self, X: np.ndarray, exposure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict mean response.

        Parameters
        ----------
        X : array-like, shape (m, p)
        exposure : array-like, shape (m,) or None

        Returns
        -------
        mu : np.ndarray, shape (m,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        eta = X @ self.coef_
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            eta = eta + np.log(exposure)
        return self._family.mean(eta)

    # ------------------------------------------------------------------
    # relativities
    # ------------------------------------------------------------------

    def relativities(self) -> pd.DataFrame:
        """Return coefficient table with relativities (exp of coefficients).

        For log-link families (Poisson, Gamma) this gives the multiplicative
        rating factors. For Gaussian (identity link) the 'relativity' column
        equals the raw coefficient.

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient, relativity, nonzero.
        """
        self._check_fitted()
        rows = []
        for name, coef in zip(self.feature_names_, self.coef_):
            if self.family_name.lower() in ("gaussian",):
                relativity = coef  # identity link: no exp transform
            else:
                relativity = float(np.exp(coef))
            rows.append({
                "feature": name,
                "coefficient": float(coef),
                "relativity": relativity,
                "nonzero": bool(abs(coef) > 1e-8),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # bootstrap_relativities
    # ------------------------------------------------------------------

    def bootstrap_relativities(
        self, n_boot: int = 200, ci: float = 0.95
    ) -> pd.DataFrame:
        """Bootstrap confidence intervals for relativities.

        Refits the model (with the same lambda_) on n_boot bootstrap samples
        and returns percentile CIs.

        Parameters
        ----------
        n_boot : int
            Number of bootstrap replicates.
        ci : float
            Confidence level (e.g. 0.95 for 95% CI).

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient, relativity, lower, upper.
        """
        self._check_fitted()
        rng = np.random.default_rng(self.random_state)

        # We need the original X, y stored — but we don't cache them.
        # This method is only meaningful if called after fit. Raise a clear error
        # if the user tries to call it without re-providing data.
        raise NotImplementedError(
            "bootstrap_relativities requires re-fitting on each bootstrap sample. "
            "Use bootstrap_relativities_from_data(X, y, n_boot=200) instead."
        )

    def bootstrap_relativities_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        n_boot: int = 200,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """Bootstrap confidence intervals for relativities, given training data.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        exposure : array-like or None
        n_boot : int
        ci : float

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient, relativity, lower, upper.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        boot_coefs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            X_b = X[idx]
            y_b = y[idx]
            exp_b = exposure[idx] if exposure is not None else None

            coef_b, _ = admm_fit(
                theta_init=np.zeros(self._n_features),
                X=X_b,
                y=y_b,
                family=self._family,
                h_y=self._h_y_value,
                lam=self._lambda_value,
                exposure_log=np.log(exp_b) if exp_b is not None else None,
                rho=self.rho,
                adagrad_lr=self.adagrad_lr,
                max_admm_iter=self.max_admm_iter,
                max_inner_iter=self.max_inner_iter,
                tol_primal=self.tol_primal,
                tol_dual=self.tol_dual,
            )
            boot_coefs.append(coef_b)

        boot_coefs = np.array(boot_coefs)  # (n_boot, p)
        alpha = (1.0 - ci) / 2.0
        lower_pct = 100.0 * alpha
        upper_pct = 100.0 * (1.0 - alpha)

        rows = []
        for j, name in enumerate(self.feature_names_):
            coef = float(self.coef_[j])
            if self.family_name.lower() in ("gaussian",):
                rel = coef
                lo = float(np.percentile(boot_coefs[:, j], lower_pct))
                hi = float(np.percentile(boot_coefs[:, j], upper_pct))
            else:
                rel = float(np.exp(coef))
                lo = float(np.exp(np.percentile(boot_coefs[:, j], lower_pct)))
                hi = float(np.exp(np.percentile(boot_coefs[:, j], upper_pct)))
            rows.append({
                "feature": name,
                "coefficient": coef,
                "relativity": rel,
                "lower": lo,
                "upper": hi,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # selected_features
    # ------------------------------------------------------------------

    def selected_features(self, tol: float = 1e-8) -> list:
        """Return list of feature names with non-zero coefficients.

        Parameters
        ----------
        tol : float
            Threshold for treating a coefficient as zero.
        """
        self._check_fitted()
        return [
            name
            for name, coef in zip(self.feature_names_, self.coef_)
            if abs(coef) > tol
        ]

    # ------------------------------------------------------------------
    # cv_path
    # ------------------------------------------------------------------

    def cv_path(self) -> Optional[pd.DataFrame]:
        """Return the CV path DataFrame, or None if lambda was specified directly.

        Columns: lambda, mean_mmd_loss, std_mmd_loss, n_nonzero.
        """
        return self._cv_df

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.coef_ is not None
        lam = f"{self._lambda_value:.4g}" if self._lambda_value is not None else "?"
        return (
            f"RobustMMDGLM(family={self.family_name!r}, "
            f"lambda={lam}, fitted={fitted})"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
