"""Unit tests for cluster/backends.py and cluster/level_map.py.

These modules have no dedicated test files. backends.py handles the
unpenalised refit GLM (design matrix construction + statsmodels fitting).
level_map.py handles the LevelMap artefact and build_level_map constructor.

All tests are self-contained: no fixtures from conftest.py required.
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
from insurance_glm_tools.cluster.level_map import LevelMap, build_level_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_poisson_data(n: int = 300, seed: int = 0):
    """Two-level factor, Poisson response."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 2, n)
    exposure = rng.uniform(0.5, 1.5, n)
    rate = np.where(levels == 0, 0.05, 0.15)
    y = rng.poisson(rate * exposure).astype(float)
    X = pd.DataFrame({"va": levels})
    return X, y, exposure


def _make_three_group_data(n: int = 400, seed: int = 1):
    """Three-level factor with known group mapping."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 6, n)  # 6 original levels
    # Map: 0,1 → group 0; 2,3 → group 1; 4,5 → group 2
    group_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    groups = np.array([group_map[v] for v in levels])
    rates = [0.05, 0.10, 0.20]
    exposure = rng.uniform(0.5, 1.5, n)
    rate = np.array([rates[g] for g in groups])
    y = rng.poisson(rate * exposure).astype(float)
    X = pd.DataFrame({"va": levels})
    return X, y, exposure, group_map


# ---------------------------------------------------------------------------
# build_refit_matrix
# ---------------------------------------------------------------------------


class TestBuildRefitMatrix:

    def test_shape_two_groups(self):
        """Two groups: intercept + 1 dummy column (group 1, group 0 dropped)."""
        X, y, _ = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}  # 2 groups
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        assert X_refit.shape == (len(X), 2)  # intercept + 1 dummy
        assert col_names[0] == "intercept"
        assert "va_g1" in col_names

    def test_shape_three_groups(self):
        """Three groups → intercept + 2 dummy columns."""
        _, _, _, group_map = _make_three_group_data()
        rng = np.random.default_rng(0)
        n = 100
        va = rng.integers(0, 6, n)
        X = pd.DataFrame({"va": va})
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        assert X_refit.shape == (n, 3)  # intercept + 2 dummies
        assert "va_g1" in col_names
        assert "va_g2" in col_names
        assert "va_g0" not in col_names  # reference group dropped

    def test_intercept_column_all_ones(self):
        X, _, _ = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        np.testing.assert_array_equal(X_refit[:, 0], np.ones(len(X)))

    def test_drop_first_false(self):
        """drop_first=False keeps all group dummies (including group 0)."""
        X, _, _ = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"], drop_first=False
        )
        assert X_refit.shape == (len(X), 3)  # intercept + 2 dummies
        assert "va_g0" in col_names
        assert "va_g1" in col_names

    def test_dummy_column_is_binary(self):
        """Each dummy column should contain only 0s and 1s."""
        X, _, _ = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        dummy_col = X_refit[:, 1]
        assert set(dummy_col).issubset({0.0, 1.0})

    def test_two_factors(self):
        """Two ordinal factors: columns for both should appear."""
        rng = np.random.default_rng(0)
        n = 100
        X = pd.DataFrame({"va": rng.integers(0, 3, n), "ncd": rng.integers(0, 2, n)})
        va_map = {0: 0, 1: 1, 2: 2}
        ncd_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": va_map, "ncd": ncd_map}, ordinal_factors=["va", "ncd"]
        )
        # intercept + 2 va dummies + 1 ncd dummy = 4 columns
        assert X_refit.shape == (n, 4)
        assert "va_g1" in col_names
        assert "va_g2" in col_names
        assert "ncd_g1" in col_names

    def test_single_group(self):
        """Single group means no dummy columns; just intercept."""
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame({"va": np.zeros(n, dtype=int)})
        group_map = {0: 0}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        assert X_refit.shape == (n, 1)  # intercept only
        assert col_names == ["intercept"]

    def test_float64_dtype(self):
        X, _, _ = _make_simple_poisson_data(n=50)
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        assert X_refit.dtype == np.float64


# ---------------------------------------------------------------------------
# fit_poisson_refit
# ---------------------------------------------------------------------------


class TestFitPoissonRefit:

    def test_fit_returns_results(self):
        X, y, exposure = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        result = fit_poisson_refit(X_refit, y, exposure)
        assert result is not None
        assert hasattr(result, "params")

    def test_coef_count(self):
        """Fitted params should match number of columns."""
        X, y, exposure = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        assert len(result.params) == len(col_names)

    def test_coef_finite(self):
        X, y, exposure = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        result = fit_poisson_refit(X_refit, y, exposure)
        assert np.all(np.isfinite(result.params))

    def test_without_exposure(self):
        X, y, _ = _make_simple_poisson_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        result = fit_poisson_refit(X_refit, y, exposure=None)
        assert np.all(np.isfinite(result.params))

    def test_coefficient_sign(self):
        """Group 1 (higher rate) should have positive coefficient vs group 0."""
        X, y, exposure = _make_simple_poisson_data(n=1000)
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        idx = col_names.index("va_g1")
        # True log(0.15) - log(0.05) = log(3) ≈ 1.1
        assert result.params[idx] > 0

    def test_three_groups_fit(self):
        X, y, exposure, group_map = _make_three_group_data()
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        assert len(result.params) == 3


# ---------------------------------------------------------------------------
# fit_gamma_refit
# ---------------------------------------------------------------------------


class TestFitGammaRefit:

    def _make_gamma_data(self, n: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        levels = rng.integers(0, 2, n)
        weights = rng.uniform(1.0, 10.0, n)
        true_sev = np.where(levels == 0, 500.0, 1500.0)
        y = rng.gamma(shape=5.0, scale=true_sev / 5.0, size=n)
        X = pd.DataFrame({"va": levels})
        return X, y, weights

    def test_fit_returns_results(self):
        X, y, weights = self._make_gamma_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        result = fit_gamma_refit(X_refit, y, weights)
        assert result is not None
        assert hasattr(result, "params")

    def test_fit_without_weights(self):
        X, y, _ = self._make_gamma_data()
        group_map = {0: 0, 1: 1}
        X_refit, _ = build_refit_matrix(X, {"va": group_map}, ordinal_factors=["va"])
        result = fit_gamma_refit(X_refit, y, weights=None)
        assert np.all(np.isfinite(result.params))

    def test_coefficient_direction(self):
        """Group 1 (higher severity) should have positive coefficient."""
        X, y, weights = self._make_gamma_data(n=800)
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_gamma_refit(X_refit, y, weights)
        idx = col_names.index("va_g1")
        # log(1500/500) = log(3) ≈ 1.1 → positive coefficient
        assert result.params[idx] > 0


# ---------------------------------------------------------------------------
# extract_group_coefficients
# ---------------------------------------------------------------------------


class TestExtractGroupCoefficients:

    def test_basic_extraction(self):
        """Extract 2-group coefficients from a fitted Poisson refit."""
        X, y, exposure = _make_simple_poisson_data(n=500)
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)

        coefs = extract_group_coefficients(result, col_names, "va", n_groups=2)
        assert len(coefs) == 2
        assert coefs[0] == pytest.approx(0.0)  # reference group

    def test_reference_group_is_zero(self):
        """Group 0 coefficient is always 0 (reference)."""
        X, y, exposure = _make_simple_poisson_data(n=300)
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        coefs = extract_group_coefficients(result, col_names, "va", n_groups=2)
        assert coefs[0] == 0.0

    def test_three_groups(self):
        """Three-group extraction should return 3 coefficients."""
        X, y, exposure, group_map = _make_three_group_data(n=600)
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        coefs = extract_group_coefficients(result, col_names, "va", n_groups=3)
        assert len(coefs) == 3
        assert coefs[0] == 0.0  # reference
        assert np.isfinite(coefs[1]) and np.isfinite(coefs[2])

    def test_coefficient_ordering(self):
        """Higher-rate group should have larger coefficient."""
        X, y, exposure = _make_simple_poisson_data(n=1000)
        group_map = {0: 0, 1: 1}
        X_refit, col_names = build_refit_matrix(
            X, {"va": group_map}, ordinal_factors=["va"]
        )
        result = fit_poisson_refit(X_refit, y, exposure)
        coefs = extract_group_coefficients(result, col_names, "va", n_groups=2)
        # Group 1 has higher rate (0.15 vs 0.05), so coef[1] > coef[0]=0
        assert coefs[1] > coefs[0]


# ---------------------------------------------------------------------------
# LevelMap
# ---------------------------------------------------------------------------


class TestLevelMap:

    def _make_simple_levelmap(self):
        """Two groups: levels 0,1 → group 0; levels 2,3 → group 1."""
        return LevelMap(
            factor="va",
            ordered_levels=(0, 1, 2, 3),
            groups=(0, 0, 1, 1),
            coefficients=(0.0, 0.5),
            group_exposure=(200.0, 150.0),
        )

    def test_n_levels(self):
        lm = self._make_simple_levelmap()
        assert lm.n_levels == 4

    def test_n_groups(self):
        lm = self._make_simple_levelmap()
        assert lm.n_groups == 2

    def test_level_to_group(self):
        lm = self._make_simple_levelmap()
        ltg = lm.level_to_group
        assert ltg[0] == 0
        assert ltg[1] == 0
        assert ltg[2] == 1
        assert ltg[3] == 1

    def test_apply_series(self):
        lm = self._make_simple_levelmap()
        s = pd.Series([0, 2, 1, 3, 0])
        result = lm.apply(s)
        assert list(result) == [0, 1, 0, 1, 0]

    def test_apply_preserves_index(self):
        lm = self._make_simple_levelmap()
        s = pd.Series([0, 2, 1], index=[10, 20, 30])
        result = lm.apply(s)
        assert list(result.index) == [10, 20, 30]

    def test_apply_unseen_level_returns_nan(self):
        """Levels not in the map produce NaN (pandas map behaviour)."""
        lm = self._make_simple_levelmap()
        s = pd.Series([0, 99])  # 99 is unseen
        result = lm.apply(s)
        assert result.isna().sum() == 1

    def test_to_df_structure(self):
        lm = self._make_simple_levelmap()
        df = lm.to_df()
        assert list(df.columns) == ["original_level", "merged_group", "coefficient", "group_exposure"]
        assert len(df) == 4

    def test_to_df_coefficient_values(self):
        lm = self._make_simple_levelmap()
        df = lm.to_df()
        # Levels 0,1 → group 0 → coefficient 0.0
        coef_g0 = df[df["original_level"].isin([0, 1])]["coefficient"].values
        np.testing.assert_allclose(coef_g0, [0.0, 0.0])
        # Levels 2,3 → group 1 → coefficient 0.5
        coef_g1 = df[df["original_level"].isin([2, 3])]["coefficient"].values
        np.testing.assert_allclose(coef_g1, [0.5, 0.5])

    def test_to_df_exposure_values(self):
        lm = self._make_simple_levelmap()
        df = lm.to_df()
        exp_g0 = df[df["original_level"].isin([0, 1])]["group_exposure"].values
        np.testing.assert_allclose(exp_g0, [200.0, 200.0])

    def test_group_summary_length(self):
        lm = self._make_simple_levelmap()
        summary = lm.group_summary()
        assert len(summary) == 2  # 2 groups

    def test_group_summary_columns(self):
        lm = self._make_simple_levelmap()
        summary = lm.group_summary()
        assert "merged_group" in summary.columns
        assert "levels" in summary.columns
        assert "coefficient" in summary.columns
        assert "group_exposure" in summary.columns

    def test_group_summary_levels_lists(self):
        lm = self._make_simple_levelmap()
        summary = lm.group_summary()
        summary = summary.set_index("merged_group")
        assert sorted(summary.loc[0, "levels"]) == [0, 1]
        assert sorted(summary.loc[1, "levels"]) == [2, 3]

    def test_repr_contains_factor_name(self):
        lm = self._make_simple_levelmap()
        r = repr(lm)
        assert "va" in r

    def test_repr_contains_n_levels(self):
        lm = self._make_simple_levelmap()
        r = repr(lm)
        assert "n_levels=4" in r

    def test_repr_contains_n_groups(self):
        lm = self._make_simple_levelmap()
        r = repr(lm)
        assert "n_groups=2" in r

    def test_single_group_levelmap(self):
        """All levels in one group."""
        lm = LevelMap(
            factor="ncd",
            ordered_levels=(0, 1, 2),
            groups=(0, 0, 0),
            coefficients=(0.0,),
            group_exposure=(500.0,),
        )
        assert lm.n_groups == 1
        assert lm.n_levels == 3
        ltg = lm.level_to_group
        assert all(v == 0 for v in ltg.values())

    def test_all_levels_distinct(self):
        """Each level in its own group (no merging)."""
        lm = LevelMap(
            factor="va",
            ordered_levels=(0, 1, 2),
            groups=(0, 1, 2),
            coefficients=(0.0, 0.3, 0.7),
            group_exposure=(100.0, 200.0, 300.0),
        )
        assert lm.n_groups == 3
        assert lm.n_levels == 3

    def test_string_levels(self):
        """LevelMap should support non-integer levels."""
        lm = LevelMap(
            factor="make",
            ordered_levels=("Ford", "Vauxhall", "Toyota"),
            groups=(0, 0, 1),
            coefficients=(0.0, 0.4),
            group_exposure=(300.0, 150.0),
        )
        ltg = lm.level_to_group
        assert ltg["Ford"] == 0
        assert ltg["Toyota"] == 1

    def test_to_df_no_nan_coefficient_when_groups_contiguous(self):
        """Coefficient lookup uses group index; no NaN expected for valid groups."""
        lm = self._make_simple_levelmap()
        df = lm.to_df()
        assert df["coefficient"].isna().sum() == 0

    def test_frozen_dataclass(self):
        """LevelMap is frozen — mutation should raise."""
        lm = self._make_simple_levelmap()
        with pytest.raises(Exception):
            lm.factor = "new_name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_level_map
# ---------------------------------------------------------------------------


class TestBuildLevelMap:

    def test_basic_construction(self):
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1, 2, 3, 4],
            groups=np.array([0, 0, 1, 1, 2]),
            coefficients=np.array([0.0, 0.3, 0.8]),
            exposure_by_level=np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
        )
        assert isinstance(lm, LevelMap)
        assert lm.factor == "va"
        assert lm.n_levels == 5
        assert lm.n_groups == 3

    def test_group_exposure_aggregation(self):
        """build_level_map should sum exposure per group."""
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1, 2, 3],
            groups=np.array([0, 0, 1, 1]),
            coefficients=np.array([0.0, 0.5]),
            exposure_by_level=np.array([100.0, 80.0, 60.0, 40.0]),
        )
        # Group 0: 100 + 80 = 180
        # Group 1: 60 + 40 = 100
        assert lm.group_exposure[0] == pytest.approx(180.0)
        assert lm.group_exposure[1] == pytest.approx(100.0)

    def test_single_group(self):
        lm = build_level_map(
            factor="ncd",
            ordered_levels=[0, 1, 2],
            groups=np.array([0, 0, 0]),
            coefficients=np.array([0.0]),
            exposure_by_level=np.array([50.0, 60.0, 70.0]),
        )
        assert lm.n_groups == 1
        assert lm.group_exposure[0] == pytest.approx(180.0)

    def test_all_distinct_groups(self):
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1, 2],
            groups=np.array([0, 1, 2]),
            coefficients=np.array([0.0, 0.3, 0.7]),
            exposure_by_level=np.array([100.0, 200.0, 300.0]),
        )
        assert lm.n_groups == 3
        assert lm.group_exposure[0] == pytest.approx(100.0)
        assert lm.group_exposure[1] == pytest.approx(200.0)
        assert lm.group_exposure[2] == pytest.approx(300.0)

    def test_groups_tuple_type(self):
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1, 2],
            groups=np.array([0, 0, 1]),
            coefficients=np.array([0.0, 0.5]),
            exposure_by_level=np.array([100.0, 80.0, 60.0]),
        )
        assert isinstance(lm.groups, tuple)
        assert all(isinstance(g, int) for g in lm.groups)

    def test_ordered_levels_preserved(self):
        levels = [5, 3, 8, 1, 0]
        lm = build_level_map(
            factor="va",
            ordered_levels=levels,
            groups=np.array([0, 0, 1, 1, 1]),
            coefficients=np.array([0.0, 0.5]),
            exposure_by_level=np.array([10.0] * 5),
        )
        assert list(lm.ordered_levels) == levels

    def test_coefficients_tuple_float(self):
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1],
            groups=np.array([0, 1]),
            coefficients=np.array([0.0, 0.5]),
            exposure_by_level=np.array([100.0, 50.0]),
        )
        assert isinstance(lm.coefficients, tuple)
        assert all(isinstance(c, float) for c in lm.coefficients)

    def test_apply_after_build(self):
        lm = build_level_map(
            factor="va",
            ordered_levels=[0, 1, 2, 3],
            groups=np.array([0, 0, 1, 1]),
            coefficients=np.array([0.0, 0.5]),
            exposure_by_level=np.array([100.0, 80.0, 60.0, 40.0]),
        )
        s = pd.Series([0, 1, 2, 3, 0])
        result = lm.apply(s)
        assert list(result) == [0, 0, 1, 1, 0]

    def test_large_exposure_by_level(self):
        """Handles large exposure values without overflow."""
        lm = build_level_map(
            factor="va",
            ordered_levels=list(range(10)),
            groups=np.zeros(10, dtype=int),
            coefficients=np.array([0.0]),
            exposure_by_level=np.ones(10) * 1e7,
        )
        assert lm.group_exposure[0] == pytest.approx(1e8)


# ---------------------------------------------------------------------------
# Edge cases — diagnostics interactions
# ---------------------------------------------------------------------------


class TestDiagnosticsEdgeCases:
    """Additional edge case tests for diagnostics functions not in test_diagnostics.py."""

    def test_poisson_ll_single_obs(self):
        from insurance_glm_tools.cluster.diagnostics import poisson_log_likelihood
        y = np.array([5.0])
        mu = np.array([5.0])
        ll = poisson_log_likelihood(y, mu)
        expected = float(5.0 * np.log(5.0) - 5.0)
        assert ll == pytest.approx(expected)

    def test_gamma_ll_no_weights(self):
        """gamma_log_likelihood with weights=None should use unit weights."""
        from insurance_glm_tools.cluster.diagnostics import gamma_log_likelihood
        y = np.array([100.0, 200.0])
        mu = y.copy()
        ll_none = gamma_log_likelihood(y, mu, weights=None)
        ll_ones = gamma_log_likelihood(y, mu, weights=np.ones(2))
        assert ll_none == pytest.approx(ll_ones)

    def test_poisson_deviance_single_obs(self):
        from insurance_glm_tools.cluster.diagnostics import poisson_deviance
        y = np.array([3.0])
        mu = np.array([2.0])
        d = poisson_deviance(y, mu)
        # 2 * (y*log(y/mu) - (y - mu)) = 2 * (3*log(1.5) - 1)
        expected = 2.0 * (3.0 * np.log(1.5) - 1.0)
        assert d == pytest.approx(expected)

    def test_gamma_deviance_single_obs(self):
        from insurance_glm_tools.cluster.diagnostics import gamma_deviance
        y = np.array([200.0])
        mu = np.array([100.0])
        d = gamma_deviance(y, mu)
        # 2 * (log(mu/y) + y/mu - 1) = 2 * (log(0.5) + 2 - 1)
        expected = 2.0 * (np.log(100.0 / 200.0) + 200.0 / 100.0 - 1.0)
        assert d == pytest.approx(expected)

    def test_bic_zero_parameters(self):
        """k_eff=0 means no penalty term — BIC = -2 * ll."""
        from insurance_glm_tools.cluster.diagnostics import compute_bic
        ll = -300.0
        bic = compute_bic(ll, k_eff=0, n=1000)
        assert bic == pytest.approx(-2.0 * ll)

    def test_diagnostic_path_best_idx_out_of_range_handled(self):
        """DiagnosticPath with best_idx at boundary."""
        from insurance_glm_tools.cluster.diagnostics import DiagnosticPath
        n = 5
        lambdas = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
        bic = np.array([100.0, 80.0, 60.0, 70.0, 90.0])
        path = DiagnosticPath(
            lambdas=lambdas,
            bic=bic,
            deviance=np.ones(n),
            n_groups=np.ones(n, dtype=int),
            best_idx=2,
        )
        assert path.best_lambda == pytest.approx(0.1)
        df = path.to_df()
        assert df["is_best"].sum() == 1
        assert df.loc[2, "is_best"]
