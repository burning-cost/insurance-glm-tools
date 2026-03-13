"""
End-to-end tests for ordinal factor clustering.

These tests verify:
1. The algorithm runs without error on synthetic data.
2. With BIC selection, it selects a lambda that yields sensible groupings.
3. The transform() method correctly recodes factor values.
4. The algorithm recovers (approximately) true group structure on
   synthetic data with clearly separated groups.
5. Multiple factors are handled correctly.
6. API edge cases (invalid family, unfitted calls, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster import FactorClusterer, LevelMap
from insurance_glm_tools.cluster.level_map import build_level_map


class TestFactorClustererBasicAPI:
    def test_invalid_family(self):
        with pytest.raises(ValueError, match="family"):
            FactorClusterer(family="tweedie")

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            FactorClusterer(method="unknown")

    def test_invalid_n_lambda(self):
        with pytest.raises(ValueError, match="n_lambda"):
            FactorClusterer(n_lambda=1)

    def test_unfitted_transform_raises(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        with pytest.raises(RuntimeError, match="fit"):
            fc.transform(d["X"])

    def test_unfitted_level_map_raises(self):
        fc = FactorClusterer()
        with pytest.raises(RuntimeError, match="fit"):
            fc.level_map("vehicle_age")

    def test_column_not_found(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        with pytest.raises(ValueError, match="not found"):
            fc.fit(d["X"], d["y"], ordinal_factors=["nonexistent"])


class TestOrdinalClusteringFit:
    def test_fit_returns_self(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        result = fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        assert result is fc

    def test_level_map_returned(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert isinstance(lm, LevelMap)
        assert lm.factor == "vehicle_age"
        assert lm.n_levels == 10
        assert lm.n_groups >= 1
        assert lm.n_groups <= 10

    def test_transform_valid_columns(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(d["X"])
        assert "vehicle_age" in X_merged.columns
        assert X_merged["vehicle_age"].nunique() <= d["X"]["vehicle_age"].nunique()

    def test_transform_no_nans(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(d["X"])
        assert X_merged["vehicle_age"].isna().sum() == 0

    def test_groups_are_integers(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert all(isinstance(g, (int, np.integer)) for g in lm.groups)


class TestBICSelection:
    def test_bic_path_computed(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic", n_lambda=20)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        assert fc.diagnostic_path is not None
        assert len(fc.diagnostic_path.lambdas) == 20

    def test_bic_path_dataframe(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic", n_lambda=15)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        df = fc.diagnostic_path.to_df()
        assert len(df) == 15
        assert df["is_best"].sum() == 1

    def test_best_lambda_set(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic", n_lambda=20)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        assert fc.best_lambda is not None
        assert fc.best_lambda > 0

    def test_fixed_lambda_no_path(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        assert fc.diagnostic_path is None
        assert fc.best_lambda == pytest.approx(0.1)


class TestMultipleFactors:
    def test_two_factor_fit(self, poisson_dataset):
        d = poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05, n_lambda=20)
        fc.fit(
            d["X"], d["y"],
            exposure=d["exposure"],
            ordinal_factors=["vehicle_age", "ncd_years"],
        )
        lm_va = fc.level_map("vehicle_age")
        lm_ncd = fc.level_map("ncd_years")
        assert lm_va.n_levels == 16
        assert lm_ncd.n_levels == 10

    def test_two_factor_transform(self, poisson_dataset):
        d = poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            d["X"], d["y"],
            exposure=d["exposure"],
            ordinal_factors=["vehicle_age", "ncd_years"],
        )
        X_merged = fc.transform(d["X"])
        assert set(X_merged.columns) == {"vehicle_age", "ncd_years"}
        assert X_merged["vehicle_age"].nunique() <= 16
        assert X_merged["ncd_years"].nunique() <= 10

    def test_refit_two_factors(self, poisson_dataset):
        d = poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            d["X"], d["y"],
            exposure=d["exposure"],
            ordinal_factors=["vehicle_age", "ncd_years"],
        )
        result = fc.refit_glm(d["X"], d["y"], exposure=d["exposure"])
        assert result is not None
        assert result.converged


class TestGroupRecovery:
    """
    Verify the algorithm recovers approximately the right number of groups
    on synthetic data with clear separation. We don't require exact level
    assignments — that depends on lambda tuning — but BIC selection should
    yield a reasonable number of groups.
    """

    def test_bic_finds_sensible_groups_poisson(self, small_poisson_dataset):
        """True structure: 2 groups. BIC should not return 10 groups or 1 group."""
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic", n_lambda=30)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        # True structure has 2 groups. Allow 1-4 given noise in 1000 obs
        assert 1 <= lm.n_groups <= 4

    def test_higher_lambda_fewer_groups(self, small_poisson_dataset):
        """More regularisation → fewer groups."""
        d = small_poisson_dataset

        fc_high = FactorClusterer(family="poisson", lambda_=1.0)
        fc_high.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])

        fc_low = FactorClusterer(family="poisson", lambda_=0.001)
        fc_low.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])

        lm_high = fc_high.level_map("vehicle_age")
        lm_low = fc_low.level_map("vehicle_age")

        assert lm_high.n_groups <= lm_low.n_groups


class TestLevelMapOutput:
    def test_to_df_structure(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        df = lm.to_df()
        assert list(df.columns) == ["original_level", "merged_group", "coefficient", "group_exposure"]
        assert len(df) == 10  # 10 levels (0-9)

    def test_group_summary(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        summary = lm.group_summary()
        assert len(summary) == lm.n_groups
        assert "levels" in summary.columns

    def test_apply_method(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        recoded = lm.apply(d["X"]["vehicle_age"])
        assert len(recoded) == len(d["X"])
        assert recoded.isna().sum() == 0

    def test_repr(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        r = repr(lm)
        assert "vehicle_age" in r
        assert "n_levels=10" in r
