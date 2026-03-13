"""
Tests for BIC computation and diagnostic path.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_glm_tools.cluster.diagnostics import (
    poisson_log_likelihood,
    gamma_log_likelihood,
    poisson_deviance,
    gamma_deviance,
    compute_bic,
    DiagnosticPath,
)


class TestPoissonLogLikelihood:
    def test_perfect_fit(self):
        """When mu == y, ll = Σ [y·log(y) - y]."""
        y = np.array([2.0, 3.0, 5.0])
        mu = y.copy()
        ll = poisson_log_likelihood(y, mu)
        expected = float(np.sum(y * np.log(y) - y))
        assert ll == pytest.approx(expected)

    def test_zero_counts(self):
        """y=0 observations contribute -mu to log-likelihood."""
        y = np.array([0.0, 0.0])
        mu = np.array([1.0, 2.0])
        ll = poisson_log_likelihood(y, mu)
        assert ll == pytest.approx(-3.0)

    def test_worse_fit_has_lower_ll(self):
        y = np.array([5.0, 5.0, 5.0])
        mu_good = np.array([5.0, 5.0, 5.0])
        mu_bad = np.array([1.0, 1.0, 1.0])
        assert poisson_log_likelihood(y, mu_good) > poisson_log_likelihood(y, mu_bad)


class TestGammaLogLikelihood:
    def test_perfect_fit(self):
        """When mu == y, ll = Σ w·[-log(y) - 1]."""
        y = np.array([100.0, 200.0, 150.0])
        mu = y.copy()
        w = np.array([5.0, 10.0, 8.0])
        ll = gamma_log_likelihood(y, mu, weights=w)
        expected = float(np.sum(w * (-np.log(y) - 1.0)))
        assert ll == pytest.approx(expected)

    def test_worse_fit_has_lower_ll(self):
        y = np.array([100.0, 100.0])
        mu_good = np.array([100.0, 100.0])
        mu_bad = np.array([10.0, 10.0])
        assert gamma_log_likelihood(y, mu_good) > gamma_log_likelihood(y, mu_bad)


class TestPoissonDeviance:
    def test_perfect_fit_zero_deviance(self):
        y = np.array([3.0, 4.0, 5.0])
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_positive_deviance(self):
        y = np.array([5.0, 5.0])
        mu = np.array([3.0, 3.0])
        assert poisson_deviance(y, mu) > 0

    def test_zero_counts(self):
        """y=0 contributes 2·mu to deviance."""
        y = np.array([0.0])
        mu = np.array([2.0])
        assert poisson_deviance(y, mu) == pytest.approx(2 * 2.0)


class TestGammaDeviance:
    def test_perfect_fit_zero_deviance(self):
        y = np.array([100.0, 200.0])
        assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_positive_deviance(self):
        y = np.array([100.0])
        mu = np.array([80.0])
        assert gamma_deviance(y, mu) > 0


class TestComputeBIC:
    def test_basic_formula(self):
        ll = -500.0
        k_eff = 5
        n = 1000
        expected = -2 * (-500.0) + 5 * np.log(1000)
        assert compute_bic(ll, k_eff, n) == pytest.approx(expected)

    def test_higher_k_penalised(self):
        """More parameters → higher BIC."""
        ll = -500.0
        n = 1000
        bic_simple = compute_bic(ll, 3, n)
        bic_complex = compute_bic(ll, 10, n)
        assert bic_complex > bic_simple

    def test_better_fit_rewarded(self):
        """Better log-likelihood → lower BIC."""
        k_eff = 5
        n = 1000
        bic_good = compute_bic(-400.0, k_eff, n)
        bic_bad = compute_bic(-600.0, k_eff, n)
        assert bic_good < bic_bad


class TestDiagnosticPath:
    def make_path(self, n=10):
        lambdas = np.logspace(-3, 0, n)
        bic = np.random.default_rng(0).uniform(100, 200, n)
        bic[4] = 50.0  # best at index 4
        deviance = np.ones(n) * 10
        n_groups = np.ones(n, dtype=np.int32) * 3
        return DiagnosticPath(
            lambdas=lambdas,
            bic=bic,
            deviance=deviance,
            n_groups=n_groups,
            best_idx=4,
        )

    def test_best_lambda(self):
        path = self.make_path()
        assert path.best_lambda == pytest.approx(path.lambdas[4])

    def test_to_df_shape(self):
        path = self.make_path(n=10)
        df = path.to_df()
        assert len(df) == 10
        assert "lambda" in df.columns
        assert "bic" in df.columns
        assert "n_groups" in df.columns
        assert df["is_best"].sum() == 1
