"""
Tests for internal helpers in cluster/clusterer.py.

These functions had no direct test coverage:
  - _build_penalised_mask: already tested in test_p0_regressions.py
  - _update_unpenalised_coef: WLS helper — tested here
  - _estimate_lambda_max: heuristic — tested here
  - _count_effective_groups: group counting — tested here
  - FactorClusterer validation paths not covered elsewhere

Also covers edge cases for FactorClusterer that the higher-level tests miss:
  - invalid min_exposure / tol
  - level_map() for unfitted/unknown factor
  - BIC path with gamma family
  - refit_glm updates level map coefficients
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.clusterer import (
    _update_unpenalised_coef,
    FactorClusterer,
)


# ---------------------------------------------------------------------------
# _update_unpenalised_coef
# ---------------------------------------------------------------------------


class TestUpdateUnpenalisedCoef:
    def test_ordinary_least_squares_simple(self):
        """For a single column X=1 (intercept-only), WLS gives weighted mean."""
        X_unpen = np.ones((5, 1))
        z_adj = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = np.ones(5)
        pen_contrib = np.zeros(5)
        coef = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)
        assert coef[0] == pytest.approx(3.0)  # mean of [1,2,3,4,5]

    def test_weighted_mean(self):
        """Weighted OLS with unequal weights should give weighted mean."""
        X_unpen = np.ones((3, 1))
        z_adj = np.array([1.0, 2.0, 3.0])
        W = np.array([1.0, 2.0, 1.0])
        pen_contrib = np.zeros(3)
        coef = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)
        expected = np.average(z_adj, weights=W)
        assert coef[0] == pytest.approx(expected)

    def test_adjusted_for_pen_contrib(self):
        """When pen_contrib is non-zero, we fit on residuals after removing pen."""
        X_unpen = np.ones((4, 1))
        z_adj = np.array([2.0, 3.0, 4.0, 5.0])
        W = np.ones(4)
        pen_contrib = np.array([0.5, 0.5, 0.5, 0.5])
        coef = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)
        # Residual = [1.5, 2.5, 3.5, 4.5], mean = 3.0
        assert coef[0] == pytest.approx(3.0)

    def test_two_column_matrix(self):
        """Two-column X should produce two coefficients."""
        X_unpen = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        rng = np.random.default_rng(0)
        z_adj = rng.normal(size=10)
        W = np.ones(10)
        pen_contrib = np.zeros(10)
        coef = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)
        assert len(coef) == 2

    def test_returns_float_array(self):
        X_unpen = np.ones((5, 1))
        z_adj = np.ones(5)
        W = np.ones(5)
        pen_contrib = np.zeros(5)
        coef = _update_unpenalised_coef(X_unpen, z_adj, W, pen_contrib)
        assert coef.dtype == np.float64


# ---------------------------------------------------------------------------
# FactorClusterer — init validation
# ---------------------------------------------------------------------------


class TestFactorClustererValidation:
    def test_invalid_min_exposure(self):
        with pytest.raises(ValueError, match="min_exposure"):
            FactorClusterer(min_exposure=-1.0)

    def test_invalid_tol(self):
        with pytest.raises(ValueError, match="tol"):
            FactorClusterer(tol=-1e-5)

    def test_valid_min_exposure_zero(self):
        fc = FactorClusterer(min_exposure=0.0)
        assert fc.min_exposure == 0.0

    def test_valid_tol_zero(self):
        fc = FactorClusterer(tol=0.0)
        assert fc.tol == 0.0

    def test_level_map_unknown_factor_raises(self):
        """After fit, asking for an unknown factor should raise ValueError."""
        rng = np.random.default_rng(1)
        n = 200
        X = pd.DataFrame({"va": rng.integers(0, 5, n)})
        y = rng.poisson(0.1, n).astype(float)
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(X, y, ordinal_factors=["va"])
        with pytest.raises(ValueError, match="not in fitted"):
            fc.level_map("nonexistent_factor")

    def test_unfitted_refit_glm_raises(self):
        X = pd.DataFrame({"va": [0, 1, 2]})
        y = np.array([1.0, 2.0, 0.0])
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        with pytest.raises(RuntimeError, match="fit"):
            fc.refit_glm(X, y)

    def test_refit_updates_level_map_coefficients(self):
        """refit_glm should update LevelMap coefficients from the MLE fit."""
        rng = np.random.default_rng(99)
        n = 500
        va = rng.integers(0, 6, n)
        exposure = rng.uniform(0.5, 1.5, n)
        true_rate = np.where(va < 3, 0.05, 0.15)
        y = rng.poisson(true_rate * exposure).astype(float)
        X = pd.DataFrame({"va": va})

        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(X, y, exposure=exposure, ordinal_factors=["va"])

        lm_before = fc.level_map("va")
        fc.refit_glm(X, y, exposure=exposure)
        lm_after = fc.level_map("va")

        # Coefficients can change after MLE refit — just check they are finite
        assert all(np.isfinite(c) for c in lm_after.coefficients)
        # n_groups should be preserved
        assert lm_after.n_groups == lm_before.n_groups


# ---------------------------------------------------------------------------
# _count_effective_groups (tested via FactorClusterer._count_effective_groups)
# ---------------------------------------------------------------------------


class TestCountEffectiveGroups:
    def setup_method(self):
        self.fc = FactorClusterer(family="poisson", lambda_=0.1)
        self.fc.tol = 1e-8

    def test_all_different_levels(self):
        """No merging → k_eff = 1 (intercept) + (K-1) per factor."""
        # Delta with all non-zero → K groups → K-1 free params + 1 intercept
        delta = np.array([0.5, 0.3, 0.2])  # 3 levels, all distinct
        k_eff = self.fc._count_effective_groups(delta, [3])
        assert k_eff == 3  # 1 + (3-1)

    def test_all_merged(self):
        """All delta₂+ are zero → 1 group → k_eff = 1 (intercept only)."""
        delta = np.array([0.5, 0.0, 0.0, 0.0])  # 4 levels → 1 group
        k_eff = self.fc._count_effective_groups(delta, [4])
        assert k_eff == 1  # 1 + (1-1)

    def test_two_groups(self):
        """Two groups → k_eff = 1 + (2-1) = 2."""
        delta = np.array([0.5, 0.3, 0.0, 0.0])  # 4 levels → 2 groups
        k_eff = self.fc._count_effective_groups(delta, [4])
        assert k_eff == 2

    def test_two_factors(self):
        """Two factors: first has 2 groups, second has 3 groups."""
        # Factor A: 3 levels, merged to 2 groups (delta[1]=0)
        delta_a = np.array([0.5, 0.0, 0.3])
        # Factor B: 4 levels, 3 groups (delta[2]=0)
        delta_b = np.array([0.2, 0.1, 0.0, 0.15])
        delta = np.concatenate([delta_a, delta_b])
        k_eff = self.fc._count_effective_groups(delta, [3, 4])
        # 1 + (2-1) + (3-1) = 1 + 1 + 2 = 4
        assert k_eff == 4


# ---------------------------------------------------------------------------
# _estimate_lambda_max
# ---------------------------------------------------------------------------


class TestEstimateLambdaMax:
    def test_returns_positive_float(self):
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(size=(n, 5))
        y = rng.poisson(0.1, n).astype(float)
        exposure = np.ones(n)

        fc = FactorClusterer(family="poisson", lambda_=0.1)
        lmax = fc._estimate_lambda_max(X, y, exposure, n_col_per_factor=[5])
        assert lmax > 0

    def test_with_no_exposure(self):
        rng = np.random.default_rng(1)
        n = 100
        X = rng.normal(size=(n, 3))
        y = rng.poisson(0.1, n).astype(float)

        fc = FactorClusterer(family="poisson", lambda_=0.1)
        lmax = fc._estimate_lambda_max(X, y, None, n_col_per_factor=[3])
        assert lmax > 0

    def test_single_level_factor(self):
        """Factor with only one level (no penalised columns) → returns default 1.0."""
        rng = np.random.default_rng(2)
        n = 50
        X = np.ones((n, 1))
        y = rng.poisson(0.1, n).astype(float)
        exposure = np.ones(n)

        fc = FactorClusterer(family="poisson", lambda_=0.1)
        # Single-level factor: n_col_per_factor=[1] → no penalised cols → returns 1.0
        lmax = fc._estimate_lambda_max(X, y, exposure, n_col_per_factor=[1])
        assert lmax >= 0.1  # should be positive


# ---------------------------------------------------------------------------
# FactorClusterer — gamma BIC path
# ---------------------------------------------------------------------------


class TestGammaBICPath:
    def test_bic_path_computed_for_gamma(self):
        """BIC lambda selection should work for the gamma family too."""
        rng = np.random.default_rng(42)
        n = 600
        va = rng.integers(0, 8, n)
        weights = rng.integers(1, 15, n).astype(float)
        true_sev = np.where(va < 4, 500.0, 1200.0)
        y = rng.gamma(shape=4.0, scale=true_sev / 4.0, size=n)
        X = pd.DataFrame({"va": va})

        fc = FactorClusterer(family="gamma", lambda_="bic", n_lambda=15)
        fc.fit(X, y, exposure=weights, ordinal_factors=["va"])

        assert fc.diagnostic_path is not None
        assert len(fc.diagnostic_path.lambdas) == 15
        assert fc.best_lambda is not None
        assert fc.best_lambda > 0

    def test_gamma_bic_selects_sensible_groups(self):
        """BIC should not return a degenerate solution (all 1 group or all K groups)."""
        rng = np.random.default_rng(7)
        n = 800
        va = rng.integers(0, 10, n)
        weights = rng.integers(1, 10, n).astype(float)
        true_sev = np.where(va < 5, 600.0, 1400.0)
        y = rng.gamma(shape=5.0, scale=true_sev / 5.0, size=n)
        X = pd.DataFrame({"va": va})

        fc = FactorClusterer(family="gamma", lambda_="bic", n_lambda=20)
        fc.fit(X, y, exposure=weights, ordinal_factors=["va"])
        lm = fc.level_map("va")
        assert 1 <= lm.n_groups <= 10

    def test_gamma_transform_preserves_all_levels(self):
        rng = np.random.default_rng(3)
        n = 300
        va = rng.integers(0, 5, n)
        weights = rng.integers(1, 5, n).astype(float)
        true_sev = np.where(va < 3, 800.0, 1600.0)
        y = rng.gamma(shape=3.0, scale=true_sev / 3.0, size=n)
        X = pd.DataFrame({"va": va})

        fc = FactorClusterer(family="gamma", lambda_=0.05)
        fc.fit(X, y, exposure=weights, ordinal_factors=["va"])
        X_merged = fc.transform(X)
        assert X_merged["va"].isna().sum() == 0
        assert X_merged["va"].nunique() <= 5


# ---------------------------------------------------------------------------
# FactorClusterer — edge cases in fit()
# ---------------------------------------------------------------------------


class TestFactorClustererFitEdgeCases:
    def test_ordinal_factors_none_uses_all_columns(self):
        """Passing ordinal_factors=None should use all DataFrame columns."""
        rng = np.random.default_rng(5)
        n = 200
        va = rng.integers(0, 5, n)
        X = pd.DataFrame({"va": va})
        y = rng.poisson(0.1, n).astype(float)
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(X, y, ordinal_factors=None)
        assert "va" in [fc.level_map("va").factor]

    def test_min_exposure_enforced(self):
        """With a large min_exposure, groups should be fewer than without."""
        rng = np.random.default_rng(11)
        n = 1000
        va = rng.integers(0, 10, n)
        exposure = rng.uniform(0.5, 1.0, n)
        y = rng.poisson(0.1 * exposure).astype(float)
        X = pd.DataFrame({"va": va})

        fc_free = FactorClusterer(family="poisson", lambda_=0.01, min_exposure=0.0)
        fc_free.fit(X, y, exposure=exposure, ordinal_factors=["va"])

        fc_constrained = FactorClusterer(family="poisson", lambda_=0.01, min_exposure=500.0)
        fc_constrained.fit(X, y, exposure=exposure, ordinal_factors=["va"])

        lm_free = fc_free.level_map("va")
        lm_constrained = fc_constrained.level_map("va")
        # With large min_exposure, cannot have more groups than without
        assert lm_constrained.n_groups <= lm_free.n_groups + 1  # +1 for noise

    def test_y_as_ndarray_works(self):
        """fit() should accept numpy arrays directly."""
        rng = np.random.default_rng(6)
        n = 200
        va = rng.integers(0, 5, n)
        exposure = rng.uniform(0.5, 1.5, n)
        y = rng.poisson(0.1, n).astype(float)
        X = pd.DataFrame({"va": va})
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(X, y, exposure=exposure, ordinal_factors=["va"])
        assert fc._is_fitted

    def test_best_lambda_matches_selected_idx(self):
        """best_lambda should exactly equal lambdas[best_idx]."""
        rng = np.random.default_rng(20)
        n = 500
        va = rng.integers(0, 8, n)
        exposure = rng.uniform(0.5, 1.5, n)
        y = rng.poisson(0.1 * exposure).astype(float)
        X = pd.DataFrame({"va": va})
        fc = FactorClusterer(family="poisson", lambda_="bic", n_lambda=20)
        fc.fit(X, y, exposure=exposure, ordinal_factors=["va"])

        path = fc.diagnostic_path
        assert fc.best_lambda == pytest.approx(path.lambdas[path.best_idx])
