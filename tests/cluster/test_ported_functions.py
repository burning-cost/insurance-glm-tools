"""
Tests for functions ported from insurance-glm-cluster:
  - enforce_min_claims
  - enforce_monotonicity
  - check_monotonicity
  - build_split_coding_matrix
  - apply_split_coding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.constraints import (
    enforce_min_claims,
    enforce_monotonicity,
    check_monotonicity,
)
from insurance_glm_tools.cluster.split_coding import (
    build_split_coding_matrix,
    apply_split_coding,
)


# ---------------------------------------------------------------------------
# enforce_min_claims
# ---------------------------------------------------------------------------


class TestEnforceMinClaims:
    def test_delegates_to_min_exposure(self):
        """enforce_min_claims should behave identically to enforce_min_exposure
        when the 'exposure' is claim counts."""
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        claims = np.array([50.0, 50.0, 3.0, 200.0, 200.0])
        coef = np.array([-1.0, -1.0, -0.98, -0.5, -0.5])

        result = enforce_min_claims(groups, claims, coef, min_claims=10)
        # Group 1 (3 claims) is well below 10 — should be absorbed
        assert result[2] != result[3]  # absorbed into 0, not 2
        assert result[2] == result[0]

    def test_no_merging_when_all_meet_threshold(self):
        groups = np.array([0, 0, 1, 1], dtype=np.int32)
        claims = np.array([100.0, 100.0, 200.0, 200.0])
        coef = np.zeros(4)
        result = enforce_min_claims(groups, claims, coef, min_claims=50)
        assert len(np.unique(result)) == 2

    def test_single_group_unchanged(self):
        groups = np.array([0, 0, 0], dtype=np.int32)
        claims = np.array([1.0, 1.0, 1.0])
        coef = np.zeros(3)
        result = enforce_min_claims(groups, claims, coef, min_claims=1000)
        assert np.all(result == result[0])


# ---------------------------------------------------------------------------
# enforce_monotonicity
# ---------------------------------------------------------------------------


class TestEnforceMonotonicity:
    def test_already_increasing(self):
        s = pd.Series([0.1, 0.3, 0.5, 0.9], index=[0, 1, 2, 3])
        result = enforce_monotonicity(s, direction="increasing")
        assert list(result.values) == pytest.approx([0.1, 0.3, 0.5, 0.9], abs=1e-6)

    def test_projects_to_increasing(self):
        # Decreasing input → should be projected to flat (isotonic)
        s = pd.Series([1.0, 0.5, 0.2, 0.1], index=[0, 1, 2, 3])
        result = enforce_monotonicity(s, direction="increasing")
        values = result.values
        # Result must be non-decreasing
        assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))

    def test_projects_to_decreasing(self):
        s = pd.Series([0.1, 0.5, 0.3, 0.9], index=[0, 1, 2, 3])
        result = enforce_monotonicity(s, direction="decreasing")
        values = result.values
        # Result must be non-increasing
        assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))

    def test_preserves_index(self):
        s = pd.Series([0.2, 0.1, 0.4], index=[10, 20, 30])
        result = enforce_monotonicity(s, direction="increasing")
        assert list(result.index) == [10, 20, 30]

    def test_invalid_direction_raises(self):
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="direction"):
            enforce_monotonicity(s, direction="sideways")

    def test_single_element(self):
        s = pd.Series([0.5], index=[0])
        result = enforce_monotonicity(s, direction="increasing")
        assert result.values[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# check_monotonicity
# ---------------------------------------------------------------------------


class TestCheckMonotonicity:
    def test_increasing_passes(self):
        s = pd.Series([0.1, 0.3, 0.5], index=[0, 1, 2])
        ok, violations = check_monotonicity(s, direction="increasing")
        assert ok
        assert violations == []

    def test_increasing_fails(self):
        s = pd.Series([0.1, 0.5, 0.3], index=[0, 1, 2])
        ok, violations = check_monotonicity(s, direction="increasing")
        assert not ok
        assert 1 in violations  # group 1 violates

    def test_decreasing_passes(self):
        s = pd.Series([0.9, 0.5, 0.2], index=[0, 1, 2])
        ok, violations = check_monotonicity(s, direction="decreasing")
        assert ok
        assert violations == []

    def test_decreasing_fails(self):
        s = pd.Series([0.9, 0.2, 0.5], index=[0, 1, 2])
        ok, violations = check_monotonicity(s, direction="decreasing")
        assert not ok
        assert 1 in violations

    def test_tolerance_respected(self):
        # Tiny downward blip within tol — should still pass
        s = pd.Series([0.1, 0.1 + 1e-8, 0.3], index=[0, 1, 2])
        ok, _ = check_monotonicity(s, direction="increasing", tol=1e-6)
        assert ok

    def test_invalid_direction_raises(self):
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="direction"):
            check_monotonicity(s, direction="flat")

    def test_roundtrip_enforce_then_check(self):
        """After enforce_monotonicity, check_monotonicity should always pass."""
        s = pd.Series([0.5, 0.1, 0.8, 0.2, 0.9], index=range(5))
        projected = enforce_monotonicity(s, direction="increasing")
        ok, violations = check_monotonicity(projected, direction="increasing")
        assert ok, f"Expected monotone after projection, got violations: {violations}"


# ---------------------------------------------------------------------------
# build_split_coding_matrix
# ---------------------------------------------------------------------------


class TestBuildSplitCodingMatrix:
    def test_shape(self):
        T = build_split_coding_matrix(4)
        assert T.shape == (4, 4)

    def test_lower_triangular_ones(self):
        T = build_split_coding_matrix(3)
        expected = np.array(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        np.testing.assert_array_equal(T, expected)

    def test_dtype(self):
        T = build_split_coding_matrix(5)
        assert T.dtype == np.float64

    def test_beta_equals_T_at_delta(self):
        """β = T @ δ should recover cumsum(δ)."""
        delta = np.array([0.5, 0.2, -0.1, 0.3])
        T = build_split_coding_matrix(4)
        beta = T @ delta
        np.testing.assert_allclose(beta, np.cumsum(delta), atol=1e-12)


# ---------------------------------------------------------------------------
# apply_split_coding
# ---------------------------------------------------------------------------


class TestApplySplitCoding:
    def test_identity_input(self):
        """apply_split_coding on the identity matrix should give T (lower-tri ones).

        Row i of the identity = one-hot for level i.
        Level i satisfies thresholds ≥0, ≥1, ..., ≥i but not ≥(i+1), ...
        So row i of output has 1s in positions 0..i, i.e. the lower-triangular T.
        """
        X = np.eye(3, dtype=np.float64)
        result = apply_split_coding(X)
        T = build_split_coding_matrix(3)
        np.testing.assert_allclose(result, T, atol=1e-12)

    def test_shape_preserved(self):
        X = np.random.default_rng(0).random((10, 5))
        result = apply_split_coding(X)
        assert result.shape == (10, 5)

    def test_dtype(self):
        X = np.ones((3, 3))
        result = apply_split_coding(X)
        assert result.dtype == np.float64

    def test_one_hot_last_level_satisfies_all_thresholds(self):
        """Level 2 (last) should satisfy all three split thresholds: [1, 1, 1]."""
        X = np.array([[0.0, 0.0, 1.0]])
        result = apply_split_coding(X)
        np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])

    def test_one_hot_first_level_satisfies_only_threshold_zero(self):
        """Level 0 satisfies only threshold ≥0: [1, 0, 0]."""
        X = np.array([[1.0, 0.0, 0.0]])
        result = apply_split_coding(X)
        np.testing.assert_array_equal(result[0], [1.0, 0.0, 0.0])

    def test_one_hot_middle_level(self):
        """Level 1 satisfies thresholds ≥0 and ≥1 but not ≥2: [1, 1, 0]."""
        X = np.array([[0.0, 1.0, 0.0]])
        result = apply_split_coding(X)
        np.testing.assert_array_equal(result[0], [1.0, 1.0, 0.0])
