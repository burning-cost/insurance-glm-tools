"""
Tests for the split-coding reparameterisation.

Core invariants:
  - delta_to_beta(beta_to_delta(β)) == β (round-trip)
  - make_split_coded_matrix produces correct binary structure
  - identify_merged_groups correctly identifies zero deltas
  - build_full_split_matrix concatenates factor blocks correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.penalties import (
    make_split_coded_matrix,
    delta_to_beta,
    beta_to_delta,
    identify_merged_groups,
    build_full_split_matrix,
    split_coded_columns_for_factor,
)


class TestDeltaBetaTransform:
    def test_round_trip_simple(self):
        beta = np.array([1.0, 1.5, 2.0, 2.0, 3.0])
        assert np.allclose(delta_to_beta(beta_to_delta(beta)), beta)

    def test_round_trip_negative(self):
        beta = np.array([-2.0, -1.7, -1.7, -1.3])
        assert np.allclose(delta_to_beta(beta_to_delta(beta)), beta)

    def test_delta_from_equal_betas(self):
        """Equal adjacent betas should give zero deltas."""
        beta = np.array([1.0, 1.0, 2.0, 2.0])
        delta = beta_to_delta(beta)
        assert delta[0] == pytest.approx(1.0)
        assert delta[1] == pytest.approx(0.0)
        assert delta[2] == pytest.approx(1.0)
        assert delta[3] == pytest.approx(0.0)

    def test_delta_to_beta_is_cumsum(self):
        delta = np.array([0.5, 0.3, 0.0, -0.2])
        expected = np.cumsum(delta)
        assert np.allclose(delta_to_beta(delta), expected)

    def test_single_level(self):
        beta = np.array([2.5])
        assert np.allclose(delta_to_beta(beta_to_delta(beta)), beta)


class TestSplitCodedMatrix:
    def test_basic_structure(self):
        """Column j has 1s for observations with level >= j."""
        levels = [0, 1, 2, 3]
        series = pd.Series([0, 1, 2, 3, 1, 0])
        X = make_split_coded_matrix(series, levels)

        assert X.shape == (6, 4)
        # All observations have level >= 0, so column 0 is all ones
        assert np.all(X[:, 0] == 1.0)
        # Level 0 only has column 0 = 1
        assert X[0, :].tolist() == [1.0, 0.0, 0.0, 0.0]
        # Level 3 has all columns = 1
        assert X[3, :].tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_non_integer_levels(self):
        """Should work with string or other hashable levels."""
        levels = ["A", "B", "C"]
        series = pd.Series(["A", "C", "B", "A"])
        X = make_split_coded_matrix(series, levels)
        assert X.shape == (4, 3)
        # "A" is index 0: only col 0 = 1
        assert X[0, :].tolist() == [1.0, 0.0, 0.0]
        # "C" is index 2: all cols = 1
        assert X[1, :].tolist() == [1.0, 1.0, 1.0]

    def test_split_coded_sum(self):
        """For K levels, row for level k should have exactly k+1 ones."""
        levels = list(range(5))
        series = pd.Series(levels)
        X = make_split_coded_matrix(series, levels)
        for i, lv in enumerate(levels):
            assert X[i, :].sum() == i + 1

    def test_split_beta_relationship(self):
        """
        X_split @ delta == X_onehot @ beta.

        This is the core algebraic identity justifying split-coding.
        """
        levels = [0, 1, 2, 3]
        series = pd.Series([0, 1, 1, 2, 3, 0])
        beta = np.array([1.0, 1.5, 2.0, 2.5])
        delta = beta_to_delta(beta)

        X_split = make_split_coded_matrix(series, levels)

        # Build one-hot for comparison
        X_onehot = np.zeros((len(series), len(levels)))
        for i, v in enumerate(series):
            X_onehot[i, v] = 1.0

        split_pred = X_split @ delta
        onehot_pred = X_onehot @ beta
        assert np.allclose(split_pred, onehot_pred)


class TestIdentifyMergedGroups:
    def test_all_different(self):
        """No merging when all deltas are non-zero."""
        delta = np.array([1.0, 0.5, -0.3, 0.2])
        groups = identify_merged_groups(delta)
        assert list(groups) == [0, 1, 2, 3]

    def test_all_same(self):
        """All merged when deltas 2+ are zero."""
        delta = np.array([1.0, 0.0, 0.0, 0.0])
        groups = identify_merged_groups(delta)
        assert list(groups) == [0, 0, 0, 0]

    def test_partial_merge(self):
        """Middle levels merged."""
        delta = np.array([1.0, 0.5, 0.0, 0.0, 0.3])
        groups = identify_merged_groups(delta)
        assert list(groups) == [0, 1, 1, 1, 2]

    def test_tolerance(self):
        """Values below tol are treated as zero."""
        delta = np.array([1.0, 0.5, 1e-10, 0.3])
        groups = identify_merged_groups(delta, tol=1e-8)
        assert list(groups) == [0, 1, 1, 2]

    def test_single_level(self):
        delta = np.array([0.5])
        groups = identify_merged_groups(delta)
        assert list(groups) == [0]


class TestBuildFullSplitMatrix:
    def test_two_factors(self):
        X = pd.DataFrame({
            "va": [0, 1, 2, 3],
            "ncd": [0, 1, 2, 0],
        })
        levels_map = {"va": [0, 1, 2, 3], "ncd": [0, 1, 2]}
        X_split, col_names = build_full_split_matrix(X, ["va", "ncd"], levels_map)

        assert X_split.shape == (4, 7)  # 4 + 3 = 7 columns
        assert len(col_names) == 7
        assert col_names[0].startswith("va__split_")
        assert col_names[4].startswith("ncd__split_")

    def test_column_names(self):
        names = split_coded_columns_for_factor("vehicle_age", [0, 1, 2])
        assert names == ["vehicle_age__split_0", "vehicle_age__split_1", "vehicle_age__split_2"]
