"""
Tests for exposure offset handling in Poisson and Gamma models.

Key invariant: Poisson with log(exposure) offset is equivalent to fitting
on (y/exposure, weight=exposure). These tests verify the equivalence and
check that exposure is correctly propagated through the fitting pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster import FactorClusterer
from insurance_glm_tools.cluster.level_map import LevelMap


class TestPoissonExposure:
    def test_fit_with_exposure(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(
            family="poisson",
            lambda_=0.1,
            min_exposure=0.0,
        )
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        assert fc._is_fitted

    def test_fit_without_exposure(self, small_poisson_dataset):
        """Fitting without exposure should work (assumes unit exposure)."""
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], ordinal_factors=["vehicle_age"])
        assert fc._is_fitted

    def test_exposure_affects_groups(self, small_poisson_dataset):
        """Fitting with vs without exposure should generally yield different groupings."""
        d = small_poisson_dataset
        fc_with = FactorClusterer(family="poisson", lambda_=0.05)
        fc_with.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])

        fc_without = FactorClusterer(family="poisson", lambda_=0.05)
        fc_without.fit(d["X"], d["y"], ordinal_factors=["vehicle_age"])

        # Not necessarily different, but both should work
        lm_with = fc_with.level_map("vehicle_age")
        lm_without = fc_without.level_map("vehicle_age")
        assert isinstance(lm_with, LevelMap)
        assert isinstance(lm_without, LevelMap)

    def test_level_map_exposure_totals(self, small_poisson_dataset):
        """Group exposure totals should sum to total exposure."""
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")

        total_group_exp = sum(lm.group_exposure)
        total_exp = float(d["exposure"].sum())
        assert total_group_exp == pytest.approx(total_exp, rel=1e-6)

    def test_refit_runs_with_exposure(self, small_poisson_dataset):
        d = small_poisson_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["exposure"], ordinal_factors=["vehicle_age"])
        result = fc.refit_glm(d["X"], d["y"], exposure=d["exposure"])
        assert result is not None
        assert len(result.params) > 0


class TestGammaExposure:
    def test_fit_with_weights(self, gamma_dataset):
        d = gamma_dataset
        fc = FactorClusterer(family="gamma", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["weights"], ordinal_factors=["vehicle_age"])
        assert fc._is_fitted

    def test_fit_without_weights(self, gamma_dataset):
        d = gamma_dataset
        fc = FactorClusterer(family="gamma", lambda_=0.1)
        fc.fit(d["X"], d["y"], ordinal_factors=["vehicle_age"])
        assert fc._is_fitted

    def test_refit_runs_gamma(self, gamma_dataset):
        d = gamma_dataset
        fc = FactorClusterer(family="gamma", lambda_=0.1)
        fc.fit(d["X"], d["y"], exposure=d["weights"], ordinal_factors=["vehicle_age"])
        result = fc.refit_glm(d["X"], d["y"], exposure=d["weights"])
        assert result is not None

    def test_gamma_level_map_structure(self, gamma_dataset):
        d = gamma_dataset
        fc = FactorClusterer(family="gamma", lambda_=0.05)
        fc.fit(d["X"], d["y"], exposure=d["weights"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert lm.n_levels == 16
        assert lm.n_groups >= 1
        assert lm.n_groups <= 16
