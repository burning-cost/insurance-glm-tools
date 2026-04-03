"""
Tests for cluster/backends.py — the statsmodels refit layer.

No test for backends existed before; these give direct coverage of:
  - build_refit_matrix: design matrix construction for merged groups
  - fit_poisson_refit: Poisson GLM with exposure offset
  - fit_gamma_refit: Gamma GLM with frequency weights
  - extract_group_coefficients: reading back per-group params from results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.backends import (
    build_refit_matrix,
    fit_poisson_refit,
    fit_gamma_refit,
    extract_group_coefficients,
)


# ---------------------------------------------------------------------------
# build_refit_matrix
# ---------------------------------------------------------------------------


class TestBuildRefitMatrix:
    def _simple_setup(self):
        """Three-level factor merged to two groups: {0:0, 1:0, 2:1}."""
        X = pd.DataFrame({"region": [0, 1, 2, 0, 1, 2]})
        factor_group_maps = {"region": {0: 0, 1: 0, 2: 1}}
        return X, factor_group_maps

    def test_shape_two_groups_one_factor(self):
        X, fgm = self._simple_setup()
        X_refit, col_names = build_refit_matrix(X, fgm, ["region"])
        # intercept + (2 groups - 1 dropped) = 2 columns
        assert X_refit.shape == (6, 2)
        assert len(col_names) == 2

    def test_intercept_column_all_ones(self):
        X, fgm = self._simple_setup()
        X_refit, col_names = build_refit_matrix(X, fgm, ["region"])
        assert col_names[0] == "intercept"
        np.testing.assert_array_equal(X_refit[:, 0], np.ones(6))

    def test_group_indicator_correct_values(self):
        """Column for group 1 should be 1 only for observations with region=2."""
        X, fgm = self._simple_setup()
        X_refit, col_names = build_refit_matrix(X, fgm, ["region"])
        # observations 2 and 5 have region=2 → group 1
        expected_g1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(X_refit[:, 1], expected_g1)

    def test_drop_first_false_includes_reference_dummy(self):
        """drop_first=False should produce one column per group (no reference drop)."""
        X, fgm = self._simple_setup()
        X_refit, col_names = build_refit_matrix(X, fgm, ["region"], drop_first=False)
        # intercept + 2 group dummies
        assert X_refit.shape == (6, 3)

    def test_two_factors(self):
        X = pd.DataFrame({
            "age": [0, 1, 2, 0, 1],
            "ncd": [0, 0, 1, 1, 0],
        })
        fgm = {
            "age": {0: 0, 1: 0, 2: 1},  # 2 groups
            "ncd": {0: 0, 1: 1},         # 2 groups
        }
        X_refit, col_names = build_refit_matrix(X, fgm, ["age", "ncd"])
        # intercept + (2-1) age + (2-1) ncd = 3 columns
        assert X_refit.shape == (5, 3)

    def test_single_group_produces_intercept_only(self):
        """When all levels merge to one group, only intercept column remains."""
        X = pd.DataFrame({"age": [0, 1, 2]})
        fgm = {"age": {0: 0, 1: 0, 2: 0}}  # all in group 0
        X_refit, col_names = build_refit_matrix(X, fgm, ["age"])
        assert X_refit.shape == (3, 1)
        assert col_names == ["intercept"]

    def test_col_names_follow_convention(self):
        """Column names should follow '{factor}_g{group}' convention."""
        X = pd.DataFrame({"va": [0, 1, 2]})
        fgm = {"va": {0: 0, 1: 1, 2: 2}}  # 3 distinct groups
        _, col_names = build_refit_matrix(X, fgm, ["va"])
        assert "va_g1" in col_names
        assert "va_g2" in col_names
        assert "va_g0" not in col_names  # reference dropped

    def test_dtype_float64(self):
        X, fgm = self._simple_setup()
        X_refit, _ = build_refit_matrix(X, fgm, ["region"])
        assert X_refit.dtype == np.float64

    @pytest.mark.parametrize("n_groups", [2, 3, 5])
    def test_varying_group_counts(self, n_groups):
        X = pd.DataFrame({"f": list(range(n_groups)) * 4})
        fgm = {"f": {v: v for v in range(n_groups)}}
        X_refit, col_names = build_refit_matrix(X, fgm, ["f"])
        # intercept + (n_groups - 1) dummies
        assert X_refit.shape[1] == n_groups


# ---------------------------------------------------------------------------
# fit_poisson_refit
# ---------------------------------------------------------------------------


class TestFitPoissonRefit:
    def _make_data(self):
        """Simple two-group Poisson data with known rates."""
        rng = np.random.default_rng(42)
        n = 300
        group = rng.integers(0, 2, n)
        exposure = rng.uniform(0.5, 1.5, n)
        true_rate = np.where(group == 0, 0.05, 0.15)
        y = rng.poisson(true_rate * exposure).astype(np.float64)
        X = np.column_stack([np.ones(n), group.astype(np.float64)])
        return X, y, exposure

    def test_returns_glmresults(self):
        from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
        X, y, exposure = self._make_data()
        result = fit_poisson_refit(X, y, exposure=exposure)
        # model.fit() returns GLMResultsWrapper, a thin proxy; accept either
        assert isinstance(result, (GLMResults, GLMResultsWrapper))

    def test_converged(self):
        X, y, exposure = self._make_data()
        result = fit_poisson_refit(X, y, exposure=exposure)
        assert result.converged

    def test_param_count(self):
        """Should have 2 parameters: intercept + one group coefficient."""
        X, y, exposure = self._make_data()
        result = fit_poisson_refit(X, y, exposure=exposure)
        assert len(result.params) == 2

    def test_without_exposure(self):
        """Should work with exposure=None (unit exposure)."""
        X, y, _ = self._make_data()
        result = fit_poisson_refit(X, y, exposure=None)
        assert result.converged

    def test_fitted_values_positive(self):
        X, y, exposure = self._make_data()
        result = fit_poisson_refit(X, y, exposure=exposure)
        assert np.all(result.fittedvalues > 0)

    def test_group_coefficient_direction(self):
        """Group 1 has higher rate than group 0 → positive second param."""
        X, y, exposure = self._make_data()
        result = fit_poisson_refit(X, y, exposure=exposure)
        assert result.params[1] > 0


# ---------------------------------------------------------------------------
# fit_gamma_refit
# ---------------------------------------------------------------------------


class TestFitGammaRefit:
    def _make_data(self):
        """Two-group Gamma severity data."""
        rng = np.random.default_rng(7)
        n = 400
        group = rng.integers(0, 2, n)
        weights = rng.integers(1, 10, n).astype(np.float64)
        true_sev = np.where(group == 0, 500.0, 1200.0)
        y = rng.gamma(shape=4.0, scale=true_sev / 4.0, size=n)
        X = np.column_stack([np.ones(n), group.astype(np.float64)])
        return X, y, weights

    def test_returns_glmresults(self):
        from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper
        X, y, weights = self._make_data()
        result = fit_gamma_refit(X, y, weights=weights)
        # model.fit() returns GLMResultsWrapper, a thin proxy; accept either
        assert isinstance(result, (GLMResults, GLMResultsWrapper))

    def test_converged(self):
        X, y, weights = self._make_data()
        result = fit_gamma_refit(X, y, weights=weights)
        assert result.converged

    def test_without_weights(self):
        X, y, _ = self._make_data()
        result = fit_gamma_refit(X, y, weights=None)
        assert result.converged

    def test_fitted_values_positive(self):
        X, y, weights = self._make_data()
        result = fit_gamma_refit(X, y, weights=weights)
        assert np.all(result.fittedvalues > 0)

    def test_higher_group_has_positive_coefficient(self):
        """Group 1 severity > group 0 → log-coefficient should be positive."""
        X, y, weights = self._make_data()
        result = fit_gamma_refit(X, y, weights=weights)
        assert result.params[1] > 0


# ---------------------------------------------------------------------------
# extract_group_coefficients
# ---------------------------------------------------------------------------


class TestExtractGroupCoefficients:
    def _fit_two_group(self):
        """Fit a Poisson with two groups and return result + col_names."""
        rng = np.random.default_rng(99)
        n = 300
        group = rng.integers(0, 2, n)
        exposure = rng.uniform(0.5, 1.5, n)
        true_rate = np.where(group == 0, 0.1, 0.3)
        y = rng.poisson(true_rate * exposure).astype(np.float64)
        X = np.column_stack([np.ones(n), group.astype(np.float64)])
        col_names = ["intercept", "factor_g1"]
        result = fit_poisson_refit(X, y, exposure=exposure)
        return result, col_names

    def test_reference_group_is_zero(self):
        result, col_names = self._fit_two_group()
        coefs = extract_group_coefficients(result, col_names, "factor", n_groups=2)
        assert coefs[0] == pytest.approx(0.0)

    def test_group1_matches_param(self):
        result, col_names = self._fit_two_group()
        coefs = extract_group_coefficients(result, col_names, "factor", n_groups=2)
        # Should match the second parameter in the result
        assert coefs[1] == pytest.approx(result.params[1], rel=1e-6)

    def test_length_equals_n_groups(self):
        result, col_names = self._fit_two_group()
        for n_groups in [2, 3]:
            coefs = extract_group_coefficients(result, col_names, "factor", n_groups=n_groups)
            assert len(coefs) == n_groups

    def test_missing_column_gives_zero(self):
        """If a group column is absent from col_names, coefficient defaults to 0."""
        result, col_names = self._fit_two_group()
        # Ask for 3 groups but only col for group 1 exists
        coefs = extract_group_coefficients(result, col_names, "factor", n_groups=3)
        assert coefs[0] == pytest.approx(0.0)
        assert coefs[2] == pytest.approx(0.0)  # missing → 0

    def test_single_group_returns_zeros(self):
        result, col_names = self._fit_two_group()
        coefs = extract_group_coefficients(result, col_names, "factor", n_groups=1)
        assert len(coefs) == 1
        assert coefs[0] == pytest.approx(0.0)

    def test_dtype_float64(self):
        result, col_names = self._fit_two_group()
        coefs = extract_group_coefficients(result, col_names, "factor", n_groups=2)
        assert coefs.dtype == np.float64
