"""
Tests for cluster/level_map.py.

Covers: LevelMap properties, to_df, group_summary, apply, __repr__,
and the build_level_map factory function.  The existing tests only
exercised LevelMap indirectly through FactorClusterer — these tests
target the class and factory directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.level_map import LevelMap, build_level_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_level_map() -> LevelMap:
    """Three levels merged to two groups: {0,1} → group 0, {2} → group 1."""
    return LevelMap(
        factor="vehicle_age",
        ordered_levels=(0, 1, 2),
        groups=(0, 0, 1),
        coefficients=(0.0, 0.4),
        group_exposure=(200.0, 100.0),
    )


def _four_level_map() -> LevelMap:
    """Four levels merged to three groups."""
    return LevelMap(
        factor="ncd",
        ordered_levels=(0, 1, 2, 3),
        groups=(0, 0, 1, 2),
        coefficients=(0.0, -0.3, -0.6),
        group_exposure=(400.0, 300.0, 150.0),
    )


# ---------------------------------------------------------------------------
# LevelMap properties
# ---------------------------------------------------------------------------


class TestLevelMapProperties:
    def test_n_levels(self):
        lm = _simple_level_map()
        assert lm.n_levels == 3

    def test_n_groups(self):
        lm = _simple_level_map()
        assert lm.n_groups == 2

    def test_n_groups_four_levels(self):
        lm = _four_level_map()
        assert lm.n_groups == 3

    def test_level_to_group_dict(self):
        lm = _simple_level_map()
        d = lm.level_to_group
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1

    def test_level_to_group_all_keys(self):
        lm = _four_level_map()
        d = lm.level_to_group
        assert set(d.keys()) == {0, 1, 2, 3}

    def test_frozen_immutability(self):
        """LevelMap is a frozen dataclass — attributes cannot be reassigned."""
        lm = _simple_level_map()
        with pytest.raises((AttributeError, TypeError)):
            lm.factor = "other"  # type: ignore[misc]

    def test_repr_contains_factor_name(self):
        lm = _simple_level_map()
        r = repr(lm)
        assert "vehicle_age" in r

    def test_repr_contains_n_levels(self):
        lm = _simple_level_map()
        r = repr(lm)
        assert "n_levels=3" in r

    def test_repr_contains_n_groups(self):
        lm = _simple_level_map()
        r = repr(lm)
        assert "n_groups=2" in r


# ---------------------------------------------------------------------------
# to_df
# ---------------------------------------------------------------------------


class TestLevelMapToDf:
    def test_returns_dataframe(self):
        lm = _simple_level_map()
        df = lm.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_equals_n_levels(self):
        lm = _simple_level_map()
        df = lm.to_df()
        assert len(df) == 3

    def test_column_names(self):
        lm = _simple_level_map()
        df = lm.to_df()
        assert set(df.columns) == {"original_level", "merged_group", "coefficient", "group_exposure"}

    def test_merged_group_values(self):
        lm = _simple_level_map()
        df = lm.to_df().set_index("original_level")
        assert df.loc[0, "merged_group"] == 0
        assert df.loc[1, "merged_group"] == 0
        assert df.loc[2, "merged_group"] == 1

    def test_coefficient_for_reference_group_is_zero(self):
        lm = _simple_level_map()
        df = lm.to_df().set_index("original_level")
        assert df.loc[0, "coefficient"] == pytest.approx(0.0)

    def test_coefficient_for_second_group_correct(self):
        lm = _simple_level_map()
        df = lm.to_df().set_index("original_level")
        assert df.loc[2, "coefficient"] == pytest.approx(0.4)

    def test_group_exposure_sums(self):
        lm = _simple_level_map()
        df = lm.to_df()
        # Both levels in group 0 should show the same group_exposure (200.0)
        g0_rows = df[df["merged_group"] == 0]
        assert (g0_rows["group_exposure"] == 200.0).all()

    def test_four_level_map_to_df_length(self):
        lm = _four_level_map()
        df = lm.to_df()
        assert len(df) == 4

    def test_no_nans_in_standard_case(self):
        lm = _simple_level_map()
        df = lm.to_df()
        assert df.notna().all().all()


# ---------------------------------------------------------------------------
# group_summary
# ---------------------------------------------------------------------------


class TestLevelMapGroupSummary:
    def test_row_count_equals_n_groups(self):
        lm = _simple_level_map()
        summary = lm.group_summary()
        assert len(summary) == 2

    def test_column_names(self):
        lm = _simple_level_map()
        summary = lm.group_summary()
        assert set(summary.columns) == {"merged_group", "levels", "coefficient", "group_exposure"}

    def test_levels_column_contains_lists(self):
        lm = _simple_level_map()
        summary = lm.group_summary()
        assert all(isinstance(x, list) for x in summary["levels"])

    def test_group0_levels_correct(self):
        lm = _simple_level_map()
        summary = lm.group_summary().set_index("merged_group")
        assert sorted(summary.loc[0, "levels"]) == [0, 1]

    def test_group1_levels_correct(self):
        lm = _simple_level_map()
        summary = lm.group_summary().set_index("merged_group")
        assert summary.loc[1, "levels"] == [2]

    def test_group_exposure_values(self):
        lm = _simple_level_map()
        summary = lm.group_summary().set_index("merged_group")
        assert summary.loc[0, "group_exposure"] == pytest.approx(200.0)
        assert summary.loc[1, "group_exposure"] == pytest.approx(100.0)

    def test_four_group_map_summary_length(self):
        lm = _four_level_map()
        summary = lm.group_summary()
        assert len(summary) == 3


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestLevelMapApply:
    def test_basic_recoding(self):
        lm = _simple_level_map()
        series = pd.Series([0, 1, 2, 0, 2])
        result = lm.apply(series)
        assert list(result) == [0, 0, 1, 0, 1]

    def test_preserves_length(self):
        lm = _simple_level_map()
        series = pd.Series(list(range(3)) * 10)
        result = lm.apply(series)
        assert len(result) == len(series)

    def test_no_nans_for_known_levels(self):
        lm = _simple_level_map()
        series = pd.Series([0, 1, 2, 0, 1])
        result = lm.apply(series)
        assert result.isna().sum() == 0

    def test_single_level_series(self):
        lm = _simple_level_map()
        result = lm.apply(pd.Series([1]))
        assert result.iloc[0] == 0  # level 1 → group 0

    def test_string_levels(self):
        lm = LevelMap(
            factor="region",
            ordered_levels=("A", "B", "C"),
            groups=(0, 0, 1),
            coefficients=(0.0, 0.2),
            group_exposure=(100.0, 50.0),
        )
        result = lm.apply(pd.Series(["A", "B", "C", "A"]))
        assert list(result) == [0, 0, 1, 0]


# ---------------------------------------------------------------------------
# build_level_map factory
# ---------------------------------------------------------------------------


class TestBuildLevelMap:
    def test_returns_level_map_instance(self):
        groups = np.array([0, 0, 1, 2], dtype=np.int32)
        coef = np.array([0.0, 0.3, 0.6])
        exposure = np.array([100.0, 120.0, 80.0, 60.0])
        lm = build_level_map("va", [0, 1, 2, 3], groups, coef, exposure)
        assert isinstance(lm, LevelMap)

    def test_factor_name_preserved(self):
        groups = np.array([0, 1], dtype=np.int32)
        coef = np.array([0.0, 0.5])
        exposure = np.array([100.0, 200.0])
        lm = build_level_map("ncd", [0, 1], groups, coef, exposure)
        assert lm.factor == "ncd"

    def test_ordered_levels_tuple(self):
        groups = np.array([0, 1], dtype=np.int32)
        coef = np.array([0.0, 0.5])
        exposure = np.array([100.0, 200.0])
        lm = build_level_map("ncd", [0, 1], groups, coef, exposure)
        assert isinstance(lm.ordered_levels, tuple)

    def test_groups_are_tuples_of_ints(self):
        groups = np.array([0, 0, 1], dtype=np.int32)
        coef = np.array([0.0, 0.5])
        exposure = np.array([100.0, 100.0, 200.0])
        lm = build_level_map("va", [0, 1, 2], groups, coef, exposure)
        assert all(isinstance(g, int) for g in lm.groups)

    def test_group_exposure_computed_correctly(self):
        """Group 0 gets levels 0 and 1 with exposure 100 + 120 = 220."""
        groups = np.array([0, 0, 1], dtype=np.int32)
        coef = np.array([0.0, 0.5])
        exposure = np.array([100.0, 120.0, 200.0])
        lm = build_level_map("va", [0, 1, 2], groups, coef, exposure)
        assert lm.group_exposure[0] == pytest.approx(220.0)
        assert lm.group_exposure[1] == pytest.approx(200.0)

    def test_n_levels_and_n_groups(self):
        groups = np.array([0, 0, 1, 2], dtype=np.int32)
        coef = np.array([0.0, 0.3, 0.6])
        exposure = np.array([50.0, 60.0, 70.0, 80.0])
        lm = build_level_map("va", [0, 1, 2, 3], groups, coef, exposure)
        assert lm.n_levels == 4
        assert lm.n_groups == 3

    def test_single_group_all_merged(self):
        groups = np.array([0, 0, 0], dtype=np.int32)
        coef = np.array([0.0])
        exposure = np.array([100.0, 200.0, 300.0])
        lm = build_level_map("va", [0, 1, 2], groups, coef, exposure)
        assert lm.n_groups == 1
        assert lm.group_exposure[0] == pytest.approx(600.0)

    @pytest.mark.parametrize("n_levels", [2, 5, 10])
    def test_varying_level_counts(self, n_levels):
        groups = np.zeros(n_levels, dtype=np.int32)
        coef = np.array([0.0])
        exposure = np.ones(n_levels) * 100.0
        lm = build_level_map("f", list(range(n_levels)), groups, coef, exposure)
        assert lm.n_levels == n_levels
