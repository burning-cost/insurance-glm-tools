"""
Extended tests for cluster/diagnostics.py.

The existing test_diagnostics.py covers the happy path; this file adds:
  - Edge cases: all-zero y, single observation, very small/large mu
  - DiagnosticPath invariants beyond basic shape checks
  - compute_bic boundary behaviour
  - gamma_deviance with weights
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


# ---------------------------------------------------------------------------
# poisson_log_likelihood — edge cases
# ---------------------------------------------------------------------------


class TestPoissonLogLikelihoodEdgeCases:
    def test_single_observation(self):
        y = np.array([3.0])
        mu = np.array([3.0])
        ll = poisson_log_likelihood(y, mu)
        expected = float(3.0 * np.log(3.0) - 3.0)
        assert ll == pytest.approx(expected)

    def test_all_zero_y(self):
        """All-zero response is valid (all terms contribute -mu)."""
        y = np.zeros(5)
        mu = np.ones(5) * 2.0
        ll = poisson_log_likelihood(y, mu)
        assert ll == pytest.approx(-10.0)

    def test_large_y(self):
        """Large counts — should not overflow."""
        y = np.array([1000.0, 2000.0])
        mu = y.copy()
        ll = poisson_log_likelihood(y, mu)
        assert np.isfinite(ll)

    def test_returns_float(self):
        y = np.array([1.0, 2.0])
        mu = np.array([1.5, 1.5])
        ll = poisson_log_likelihood(y, mu)
        assert isinstance(ll, float)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0])
    def test_scaling_mu_changes_ll(self, scale):
        """Changing mu should change the log-likelihood."""
        y = np.array([3.0, 5.0, 2.0])
        mu_base = np.array([3.0, 5.0, 2.0])
        mu_scaled = mu_base * scale
        ll_base = poisson_log_likelihood(y, mu_base)
        ll_scaled = poisson_log_likelihood(y, mu_scaled)
        if scale != 1.0:
            assert ll_base != ll_scaled


# ---------------------------------------------------------------------------
# gamma_log_likelihood — edge cases
# ---------------------------------------------------------------------------


class TestGammaLogLikelihoodEdgeCases:
    def test_no_weights_defaults_to_ones(self):
        y = np.array([100.0, 200.0])
        mu = y.copy()
        ll_none = gamma_log_likelihood(y, mu, weights=None)
        ll_ones = gamma_log_likelihood(y, mu, weights=np.ones(2))
        assert ll_none == pytest.approx(ll_ones)

    def test_zero_weight_observation_excluded(self):
        """An observation with zero weight contributes nothing."""
        y = np.array([100.0, 200.0, 50.0])
        mu = y.copy()
        w_full = np.array([1.0, 1.0, 1.0])
        w_zero = np.array([1.0, 1.0, 0.0])
        ll_full = gamma_log_likelihood(y, mu, weights=w_full)
        ll_zero = gamma_log_likelihood(y, mu, weights=w_zero)
        # Third term: -log(50) - 1 = -(log(50)+1), excluded in zero version
        assert ll_full != pytest.approx(ll_zero)

    def test_returns_float(self):
        y = np.array([100.0])
        mu = np.array([100.0])
        assert isinstance(gamma_log_likelihood(y, mu), float)

    def test_higher_weight_amplifies_ll(self):
        y = np.array([100.0, 100.0])
        mu = np.array([80.0, 80.0])
        w_low = np.ones(2)
        w_high = np.ones(2) * 5.0
        ll_low = gamma_log_likelihood(y, mu, weights=w_low)
        ll_high = gamma_log_likelihood(y, mu, weights=w_high)
        assert ll_high == pytest.approx(ll_low * 5.0)


# ---------------------------------------------------------------------------
# poisson_deviance — edge cases
# ---------------------------------------------------------------------------


class TestPoissonDevianceEdgeCases:
    def test_single_observation(self):
        y = np.array([4.0])
        mu = np.array([4.0])
        assert poisson_deviance(y, mu) == pytest.approx(0.0, abs=1e-10)

    def test_all_zero_y(self):
        """y=0: deviance contribution = 2·mu (since 0·log(0/mu) - (0-mu) = mu)."""
        y = np.zeros(3)
        mu = np.array([1.0, 2.0, 3.0])
        expected = 2.0 * mu.sum()
        assert poisson_deviance(y, mu) == pytest.approx(expected)

    def test_deviance_is_non_negative(self):
        rng = np.random.default_rng(5)
        y = rng.poisson(2.0, 20).astype(float)
        mu = rng.uniform(0.5, 5.0, 20)
        assert poisson_deviance(y, mu) >= 0

    def test_symmetry_not_expected(self):
        """Poisson deviance is not symmetric: D(y, mu) != D(mu, y)."""
        y = np.array([5.0])
        mu = np.array([3.0])
        d1 = poisson_deviance(y, mu)
        d2 = poisson_deviance(mu, y)
        assert d1 != pytest.approx(d2)


# ---------------------------------------------------------------------------
# gamma_deviance — edge cases
# ---------------------------------------------------------------------------


class TestGammaDevianceEdgeCases:
    def test_perfect_fit_zero_deviance_with_weights(self):
        y = np.array([100.0, 200.0])
        weights = np.array([5.0, 10.0])
        assert gamma_deviance(y, y, weights=weights) == pytest.approx(0.0, abs=1e-10)

    def test_no_weights_same_as_unit_weights(self):
        y = np.array([100.0, 200.0])
        mu = np.array([110.0, 190.0])
        d_none = gamma_deviance(y, mu, weights=None)
        d_ones = gamma_deviance(y, mu, weights=np.ones(2))
        assert d_none == pytest.approx(d_ones)

    def test_returns_float(self):
        y = np.array([100.0])
        mu = np.array([120.0])
        assert isinstance(gamma_deviance(y, mu), float)

    @pytest.mark.parametrize("mu_scale", [0.8, 1.2, 2.0])
    def test_nonzero_deviance_when_mu_differs(self, mu_scale):
        y = np.array([100.0, 200.0, 150.0])
        mu = y * mu_scale
        dev = gamma_deviance(y, mu)
        if mu_scale != 1.0:
            assert dev > 0


# ---------------------------------------------------------------------------
# compute_bic — additional checks
# ---------------------------------------------------------------------------


class TestComputeBICExtended:
    def test_formula_matches_definition(self):
        """BIC = -2 * ll + k * log(n)."""
        ll = -123.4
        k = 7
        n = 500
        expected = -2 * ll + k * np.log(n)
        assert compute_bic(ll, k, n) == pytest.approx(expected)

    def test_single_observation(self):
        """n=1 → log(1) = 0 → BIC = -2 * ll."""
        ll = -5.0
        assert compute_bic(ll, k_eff=3, n=1) == pytest.approx(-2 * ll)

    def test_zero_parameters(self):
        """k=0 is degenerate but should not error."""
        ll = -100.0
        n = 100
        result = compute_bic(ll, k_eff=0, n=n)
        assert result == pytest.approx(-2 * ll)

    def test_returns_float(self):
        assert isinstance(compute_bic(-100.0, 5, 200), float)

    @pytest.mark.parametrize("n", [10, 100, 1000, 10000])
    def test_penalty_grows_with_n(self, n):
        """For fixed ll and k, BIC should increase with n (log(n) grows)."""
        ll = -500.0
        k = 5
        bics = [compute_bic(ll, k, n_i) for n_i in [10, 100, 1000]]
        assert bics[0] < bics[1] < bics[2]


# ---------------------------------------------------------------------------
# DiagnosticPath — additional checks
# ---------------------------------------------------------------------------


class TestDiagnosticPathExtended:
    def _make_path(self, n: int = 10, best_at: int = 5):
        rng = np.random.default_rng(0)
        lambdas = np.logspace(-3, 0, n)
        bic = rng.uniform(100, 200, n)
        bic[best_at] = 50.0
        deviance = np.ones(n) * 10.0
        n_groups = np.arange(1, n + 1, dtype=np.int32)
        return DiagnosticPath(
            lambdas=lambdas,
            bic=bic,
            deviance=deviance,
            n_groups=n_groups,
            best_idx=best_at,
        )

    def test_best_lambda_is_lambda_at_best_idx(self):
        path = self._make_path(n=20, best_at=7)
        assert path.best_lambda == pytest.approx(path.lambdas[7])

    def test_to_df_has_correct_number_of_rows(self):
        for n in [5, 10, 50]:
            path = self._make_path(n=n, best_at=n // 2)
            df = path.to_df()
            assert len(df) == n

    def test_to_df_is_best_exactly_one_true(self):
        path = self._make_path()
        df = path.to_df()
        assert df["is_best"].sum() == 1

    def test_to_df_is_best_at_correct_row(self):
        path = self._make_path(n=10, best_at=3)
        df = path.to_df()
        assert df.loc[3, "is_best"]

    def test_to_df_columns(self):
        path = self._make_path()
        df = path.to_df()
        assert set(df.columns) == {"lambda", "bic", "deviance", "n_groups", "is_best"}

    def test_best_lambda_coincides_with_min_bic(self):
        path = self._make_path(n=15, best_at=9)
        df = path.to_df()
        min_bic_row = df.loc[df["bic"].idxmin()]
        assert min_bic_row["is_best"]
