"""
Extended tests for cluster/constraints.py.

Covers the private _pav_increasing function directly,
and additional edge cases for enforce_min_exposure and enforce_monotonicity
not covered by test_constraints.py and test_ported_functions.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.cluster.constraints import (
    _pav_increasing,
    enforce_min_exposure,
    enforce_monotonicity,
    check_monotonicity,
    relabel_groups_contiguous,
    compute_group_exposure,
)


# ---------------------------------------------------------------------------
# _pav_increasing — pool adjacent violators (pure NumPy implementation)
# ---------------------------------------------------------------------------


class TestPavIncreasing:
    def test_already_increasing(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = _pav_increasing(values)
        np.testing.assert_allclose(result, values)

    def test_decreasing_flattens_to_mean(self):
        """Fully decreasing → all values pool to a single block with the mean."""
        values = np.array([4.0, 3.0, 2.0, 1.0])
        result = _pav_increasing(values)
        assert all(v == pytest.approx(2.5) for v in result)

    def test_single_violation(self):
        """One downward step: [1, 3, 2, 4] → [1, 2.5, 2.5, 4]."""
        values = np.array([1.0, 3.0, 2.0, 4.0])
        result = _pav_increasing(values)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.5)
        assert result[2] == pytest.approx(2.5)
        assert result[3] == pytest.approx(4.0)

    def test_flat_input_unchanged(self):
        values = np.array([3.0, 3.0, 3.0])
        result = _pav_increasing(values)
        np.testing.assert_allclose(result, values)

    def test_single_element(self):
        values = np.array([5.0])
        result = _pav_increasing(values)
        assert result[0] == pytest.approx(5.0)

    def test_length_preserved(self):
        for n in [2, 5, 10, 20]:
            values = np.random.default_rng(n).normal(size=n)
            result = _pav_increasing(values)
            assert len(result) == n

    def test_result_is_non_decreasing(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            values = rng.normal(size=15)
            result = _pav_increasing(values)
            diffs = np.diff(result)
            assert np.all(diffs >= -1e-10), f"Not non-decreasing: {result}"

    def test_result_is_within_range(self):
        """PAV cannot produce values outside [min(input), max(input)]."""
        values = np.array([2.0, 5.0, 1.0, 4.0, 3.0])
        result = _pav_increasing(values)
        assert result.min() >= values.min() - 1e-10
        assert result.max() <= values.max() + 1e-10

    def test_two_elements_in_order(self):
        values = np.array([1.0, 2.0])
        result = _pav_increasing(values)
        np.testing.assert_allclose(result, values)

    def test_two_elements_reversed(self):
        values = np.array([2.0, 1.0])
        result = _pav_increasing(values)
        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# enforce_min_exposure — additional edge cases
# ---------------------------------------------------------------------------


class TestEnforceMinExposureExtended:
    def test_two_groups_both_below_threshold_merge_to_one(self):
        """When both groups are below threshold, the smaller absorbs into the larger."""
        groups = np.array([0, 1], dtype=np.int32)
        exposure = np.array([30.0, 50.0])
        coef = np.array([0.0, 0.1])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        assert len(np.unique(result)) == 1

    def test_groups_preserved_when_equal_to_threshold(self):
        """Exactly at threshold should NOT trigger absorption."""
        groups = np.array([0, 1], dtype=np.int32)
        exposure = np.array([100.0, 100.0])
        coef = np.array([0.0, 0.5])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        assert len(np.unique(result)) == 2

    def test_negative_coefficients_use_nearest(self):
        """Test with negative coefficients — nearest should still be correct."""
        # Groups: 0 (coef -2.0), 1 (coef -1.9, tiny), 2 (coef 0.0)
        # Nearest to group 1 is group 0 (distance 0.1 vs 1.9)
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([200.0, 200.0, 5.0, 200.0, 200.0])
        coef = np.array([-2.0, -2.0, -1.9, 0.0, 0.0])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        assert result[2] == result[0]  # absorbed into group 0

    def test_returns_numpy_array(self):
        groups = np.array([0, 1], dtype=np.int32)
        exposure = np.array([200.0, 200.0])
        coef = np.array([0.0, 0.1])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=0.0)
        assert isinstance(result, np.ndarray)

    def test_does_not_modify_input(self):
        """The original groups array should not be mutated."""
        groups = np.array([0, 1, 2], dtype=np.int32)
        original = groups.copy()
        exposure = np.array([5.0, 5.0, 5.0])
        coef = np.array([0.0, 0.1, 0.2])
        enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        np.testing.assert_array_equal(groups, original)


# ---------------------------------------------------------------------------
# enforce_monotonicity — additional edge cases
# ---------------------------------------------------------------------------


class TestEnforceMonotonicityExtended:
    def test_two_elements_increasing(self):
        s = pd.Series([1.0, 2.0], index=[0, 1])
        result = enforce_monotonicity(s, direction="increasing")
        assert result.values[0] <= result.values[1] + 1e-9

    def test_two_elements_decreasing_projection(self):
        """Input [2.0, 1.0] projected to increasing → [1.5, 1.5]."""
        s = pd.Series([2.0, 1.0], index=[0, 1])
        result = enforce_monotonicity(s, direction="increasing")
        assert result.values[0] == pytest.approx(1.5)
        assert result.values[1] == pytest.approx(1.5)

    def test_empty_series(self):
        """Empty series should not crash."""
        s = pd.Series([], dtype=float)
        result = enforce_monotonicity(s, direction="increasing")
        assert len(result) == 0

    def test_all_equal_stays_equal(self):
        s = pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2])
        result = enforce_monotonicity(s, direction="increasing")
        np.testing.assert_allclose(result.values, [0.5, 0.5, 0.5])

    def test_decreasing_mirror_of_increasing(self):
        """enforce_monotonicity(s, decreasing) should give -enforce(-s, increasing)."""
        rng = np.random.default_rng(10)
        values = rng.normal(size=8)
        s = pd.Series(values, index=range(8))
        result_dec = enforce_monotonicity(s, direction="decreasing")
        result_inc_neg = enforce_monotonicity(-s, direction="increasing")
        np.testing.assert_allclose(result_dec.values, -result_inc_neg.values, atol=1e-10)

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_result_is_monotone_after_projection(self, n):
        rng = np.random.default_rng(n)
        values = rng.normal(size=n)
        s = pd.Series(values, index=range(n))
        result = enforce_monotonicity(s, direction="increasing")
        diffs = np.diff(result.values)
        assert np.all(diffs >= -1e-9)


# ---------------------------------------------------------------------------
# check_monotonicity — additional edge cases
# ---------------------------------------------------------------------------


class TestCheckMonotonicityExtended:
    def test_empty_series_is_monotone(self):
        s = pd.Series([], dtype=float)
        ok, violations = check_monotonicity(s, direction="increasing")
        assert ok
        assert violations == []

    def test_single_element_is_monotone(self):
        s = pd.Series([1.0], index=[0])
        ok, violations = check_monotonicity(s, direction="increasing")
        assert ok

    def test_two_equal_elements_pass_increasing(self):
        s = pd.Series([1.0, 1.0], index=[0, 1])
        ok, violations = check_monotonicity(s, direction="increasing")
        assert ok

    def test_two_equal_elements_pass_decreasing(self):
        s = pd.Series([1.0, 1.0], index=[0, 1])
        ok, violations = check_monotonicity(s, direction="decreasing")
        assert ok

    def test_multiple_violations_reported(self):
        """[1, 3, 2, 4, 3] → violations at indices 1 and 3."""
        s = pd.Series([1.0, 3.0, 2.0, 4.0, 3.0], index=[0, 1, 2, 3, 4])
        ok, violations = check_monotonicity(s, direction="increasing")
        assert not ok
        assert 1 in violations
        assert 3 in violations

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_roundtrip_after_projection_always_passes(self, direction):
        rng = np.random.default_rng(77)
        values = rng.normal(size=15)
        s = pd.Series(values, index=range(15))
        projected = enforce_monotonicity(s, direction=direction)
        ok, violations = check_monotonicity(projected, direction=direction)
        assert ok, f"Violations after projection: {violations}"


# ---------------------------------------------------------------------------
# relabel_groups_contiguous — additional checks
# ---------------------------------------------------------------------------


class TestRelabelGroupsContiguousExtended:
    def test_all_same_label_to_zero(self):
        groups = np.array([5, 5, 5, 5], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert np.all(result == 0)

    def test_non_monotone_input(self):
        """Labels [2, 0, 0, 2] → should map in order of first appearance: 2→0, 0→1."""
        groups = np.array([2, 0, 0, 2], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert result[0] == result[3]  # 2 → same label
        assert result[1] == result[2]  # 0 → same label
        assert result[0] != result[1]  # different from each other

    def test_preserves_first_occurrence_order(self):
        """First-seen label maps to 0, second to 1, etc."""
        groups = np.array([7, 3, 7, 5, 3], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert result[0] == 0  # 7 seen first → 0
        assert result[1] == 1  # 3 seen second → 1
        assert result[3] == 2  # 5 seen third → 2
        assert result[2] == 0  # 7 again → 0
        assert result[4] == 1  # 3 again → 1

    def test_output_dtype_preserves_int(self):
        groups = np.array([0, 2, 2], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert np.issubdtype(result.dtype, np.integer)


# ---------------------------------------------------------------------------
# compute_group_exposure — additional checks
# ---------------------------------------------------------------------------


class TestComputeGroupExposureExtended:
    def test_single_group(self):
        groups = np.array([0, 0, 0], dtype=np.int32)
        exposure = np.array([100.0, 200.0, 300.0])
        result = compute_group_exposure(groups, exposure)
        assert result == {0: pytest.approx(600.0)}

    def test_returns_dict(self):
        groups = np.array([0, 1], dtype=np.int32)
        exposure = np.array([50.0, 75.0])
        result = compute_group_exposure(groups, exposure)
        assert isinstance(result, dict)

    def test_non_contiguous_groups(self):
        """Non-contiguous group labels (0, 5, 10) should still work."""
        groups = np.array([0, 5, 10, 0], dtype=np.int32)
        exposure = np.array([100.0, 200.0, 300.0, 50.0])
        result = compute_group_exposure(groups, exposure)
        assert result[0] == pytest.approx(150.0)
        assert result[5] == pytest.approx(200.0)
        assert result[10] == pytest.approx(300.0)

    def test_total_exposure_preserved(self):
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([100.0, 50.0, 200.0, 80.0, 120.0])
        result = compute_group_exposure(groups, exposure)
        assert sum(result.values()) == pytest.approx(exposure.sum())
