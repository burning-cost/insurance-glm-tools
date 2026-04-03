"""
Extended tests for cluster/split_coding.py and cluster/penalties.py.

split_coding.py: build_split_coding_matrix and apply_split_coding
penalties.py:    build_full_split_matrix with other_cols, edge cases

The existing test_ported_functions.py and test_penalties.py covered the main
paths.  These tests add:
  - apply_split_coding properties (non-decreasing column sums)
  - build_split_coding_matrix for n=1 and n=2
  - build_full_split_matrix with other_cols pass-through
  - make_split_coded_matrix single-level factor
  - identify_merged_groups with exactly-tol values
  - beta_to_delta / delta_to_beta for all-zero inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.split_coding import (
    build_split_coding_matrix,
    apply_split_coding,
)
from insurance_glm_tools.cluster.penalties import (
    make_split_coded_matrix,
    delta_to_beta,
    beta_to_delta,
    identify_merged_groups,
    build_full_split_matrix,
    split_coded_columns_for_factor,
)


# ---------------------------------------------------------------------------
# build_split_coding_matrix — edge cases
# ---------------------------------------------------------------------------


class TestBuildSplitCodingMatrixEdgeCases:
    def test_n1_is_identity(self):
        T = build_split_coding_matrix(1)
        np.testing.assert_array_equal(T, np.array([[1.0]]))

    def test_n2_correct(self):
        T = build_split_coding_matrix(2)
        expected = np.array([[1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(T, expected)

    def test_upper_triangle_all_zeros(self):
        for n in [3, 5, 8]:
            T = build_split_coding_matrix(n)
            assert np.all(np.triu(T, k=1) == 0)

    def test_diagonal_all_ones(self):
        for n in [3, 5, 8]:
            T = build_split_coding_matrix(n)
            np.testing.assert_array_equal(np.diag(T), np.ones(n))

    def test_row_sums_equal_row_index_plus_one(self):
        """Row i should have i+1 ones (lower triangular)."""
        n = 6
        T = build_split_coding_matrix(n)
        for i in range(n):
            assert T[i, :].sum() == i + 1


# ---------------------------------------------------------------------------
# apply_split_coding — properties
# ---------------------------------------------------------------------------


class TestApplySplitCodingExtended:
    def test_column_sums_non_decreasing(self):
        """For a balanced one-hot input, column j sum ≥ column j+1 sum."""
        n = 5
        X = np.eye(n, dtype=np.float64)
        result = apply_split_coding(X)
        col_sums = result.sum(axis=0)
        for j in range(n - 1):
            assert col_sums[j] >= col_sums[j + 1]

    def test_binary_output_for_one_hot_input(self):
        """apply_split_coding on a one-hot matrix should yield 0/1 values."""
        n = 4
        X = np.eye(n, dtype=np.float64)
        result = apply_split_coding(X)
        unique_vals = np.unique(result)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_all_ones_input_first_col_unchanged(self):
        """Column 0 of any input should equal column 0 of the original + reversed cumsum."""
        X = np.ones((3, 3))
        result = apply_split_coding(X)
        # Each row should have column 0 = sum of all columns in that row
        for i in range(3):
            assert result[i, 0] == pytest.approx(X[i, :].sum())

    def test_apply_consistent_with_matrix_multiply(self):
        """apply_split_coding(X) should equal X @ T (where T is the triangular matrix)."""
        rng = np.random.default_rng(0)
        n_obs, n_levels = 8, 5
        # Create a valid one-hot-like matrix
        indices = rng.integers(0, n_levels, n_obs)
        X = np.zeros((n_obs, n_levels))
        for i, idx in enumerate(indices):
            X[i, idx] = 1.0

        T = build_split_coding_matrix(n_levels)
        expected = X @ T
        result = apply_split_coding(X)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# make_split_coded_matrix — edge cases
# ---------------------------------------------------------------------------


class TestMakeSplitCodedMatrixEdgeCases:
    def test_single_level(self):
        """Factor with one level → single column, all ones."""
        series = pd.Series(["A", "A", "A"])
        X = make_split_coded_matrix(series, ["A"])
        assert X.shape == (3, 1)
        np.testing.assert_array_equal(X[:, 0], np.ones(3))

    def test_two_levels(self):
        series = pd.Series([0, 1, 0, 1])
        levels = [0, 1]
        X = make_split_coded_matrix(series, levels)
        assert X.shape == (4, 2)
        # Column 0 all ones (level >= 0 always true)
        np.testing.assert_array_equal(X[:, 0], np.ones(4))
        # Column 1 = 1 only for level >= 1
        np.testing.assert_array_equal(X[:, 1], [0.0, 1.0, 0.0, 1.0])

    def test_dtype_float64(self):
        series = pd.Series([0, 1, 2])
        X = make_split_coded_matrix(series, [0, 1, 2])
        assert X.dtype == np.float64

    def test_repeated_levels(self):
        """Multiple rows with the same level should all get the same row."""
        series = pd.Series([2, 2, 2])
        levels = [0, 1, 2]
        X = make_split_coded_matrix(series, levels)
        # Level 2 → all columns should be 1
        np.testing.assert_array_equal(X, np.ones((3, 3)))


# ---------------------------------------------------------------------------
# delta_to_beta / beta_to_delta — edge cases
# ---------------------------------------------------------------------------


class TestDeltaBetaEdgeCases:
    def test_all_zero_delta_gives_all_zero_beta(self):
        delta = np.zeros(5)
        beta = delta_to_beta(delta)
        np.testing.assert_array_equal(beta, np.zeros(5))

    def test_all_zero_beta_gives_all_zero_delta(self):
        beta = np.zeros(5)
        delta = beta_to_delta(beta)
        np.testing.assert_array_equal(delta, np.zeros(5))

    def test_negative_values_round_trip(self):
        beta = np.array([-3.0, -2.5, -2.0, -1.8])
        assert np.allclose(delta_to_beta(beta_to_delta(beta)), beta)

    def test_delta_first_element_equals_beta_first_element(self):
        """δ₁ = β₁ by definition."""
        beta = np.array([1.5, 2.0, 3.0])
        delta = beta_to_delta(beta)
        assert delta[0] == pytest.approx(beta[0])

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_round_trip_various_lengths(self, n):
        rng = np.random.default_rng(n)
        beta = rng.normal(size=n)
        assert np.allclose(delta_to_beta(beta_to_delta(beta)), beta)


# ---------------------------------------------------------------------------
# identify_merged_groups — edge cases
# ---------------------------------------------------------------------------


class TestIdentifyMergedGroupsEdgeCases:
    def test_exactly_at_tol_treated_as_zero(self):
        """|delta| == tol is NOT > tol, so level is merged (treated as zero)."""
        tol = 1e-6
        delta = np.array([0.5, tol, 0.3])
        groups = identify_merged_groups(delta, tol=tol)
        # delta[1] == tol → not > tol → merged with group 0
        assert groups[1] == groups[0]

    def test_slightly_above_tol_creates_new_group(self):
        tol = 1e-6
        delta = np.array([0.5, tol * 10, 0.3])
        groups = identify_merged_groups(delta, tol=tol)
        assert groups[1] != groups[0]

    def test_two_levels_both_nonzero(self):
        delta = np.array([1.0, 0.5])
        groups = identify_merged_groups(delta)
        assert list(groups) == [0, 1]

    def test_first_delta_never_creates_new_group(self):
        """delta[0] is always the anchor: group 0, regardless of magnitude."""
        for val in [0.0, 1e-20, 1.0, 100.0]:
            delta = np.array([val])
            groups = identify_merged_groups(delta)
            assert groups[0] == 0

    def test_large_tol_merges_everything(self):
        """Huge tol → all non-anchor deltas treated as zero → 1 group."""
        delta = np.array([0.5, 0.3, 0.2, 0.8])
        groups = identify_merged_groups(delta, tol=100.0)
        assert len(np.unique(groups)) == 1


# ---------------------------------------------------------------------------
# build_full_split_matrix — other_cols pass-through
# ---------------------------------------------------------------------------


class TestBuildFullSplitMatrixExtended:
    def test_other_cols_included(self):
        X = pd.DataFrame({
            "va": [0, 1, 2],
            "some_continuous": [1.0, 2.0, 3.0],
        })
        levels_map = {"va": [0, 1, 2]}
        X_split, col_names = build_full_split_matrix(
            X, ["va"], levels_map, other_cols=["some_continuous"]
        )
        # 3 split cols + 1 continuous = 4 columns
        assert X_split.shape == (3, 4)
        assert "some_continuous" in col_names

    def test_other_cols_values_correct(self):
        X = pd.DataFrame({
            "va": [0, 1],
            "rate": [0.5, 1.5],
        })
        levels_map = {"va": [0, 1]}
        X_split, col_names = build_full_split_matrix(
            X, ["va"], levels_map, other_cols=["rate"]
        )
        rate_idx = col_names.index("rate")
        np.testing.assert_allclose(X_split[:, rate_idx], [0.5, 1.5])

    def test_empty_dataframe_no_other_cols(self):
        """No factors, no other_cols → empty matrix."""
        X = pd.DataFrame({"va": [0, 1, 2]})
        X_split, col_names = build_full_split_matrix(X, [], {})
        assert X_split.shape == (3, 0)
        assert col_names == []

    def test_column_order_factors_then_other(self):
        X = pd.DataFrame({
            "va": [0, 1],
            "extra": [10.0, 20.0],
        })
        levels_map = {"va": [0, 1]}
        _, col_names = build_full_split_matrix(
            X, ["va"], levels_map, other_cols=["extra"]
        )
        # Split cols come first, then extra
        assert col_names.index("extra") > col_names.index("va__split_0")


# ---------------------------------------------------------------------------
# split_coded_columns_for_factor
# ---------------------------------------------------------------------------


class TestSplitCodedColumnsForFactor:
    def test_single_level(self):
        names = split_coded_columns_for_factor("va", [0])
        assert names == ["va__split_0"]

    def test_names_indexed_from_zero(self):
        names = split_coded_columns_for_factor("ncd", list(range(5)))
        assert names[0] == "ncd__split_0"
        assert names[4] == "ncd__split_4"

    def test_length_matches_levels(self):
        for n in [1, 3, 10]:
            names = split_coded_columns_for_factor("f", list(range(n)))
            assert len(names) == n

    def test_factor_name_in_each_column(self):
        names = split_coded_columns_for_factor("my_factor", [0, 1, 2])
        assert all("my_factor" in name for name in names)
