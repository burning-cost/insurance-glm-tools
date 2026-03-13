"""
Tests for min-exposure constraint enforcement.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_glm_tools.cluster.constraints import (
    enforce_min_exposure,
    compute_group_exposure,
    relabel_groups_contiguous,
)


class TestEnforceMinExposure:
    def test_no_enforcement_when_zero(self):
        """min_exposure=0 disables constraint."""
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([10.0, 10.0, 5.0, 100.0, 100.0])
        coef = np.array([0.0, 0.0, 0.1, 0.5, 0.5])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=0.0)
        assert np.array_equal(result, groups)

    def test_under_exposure_absorbed(self):
        """A group with 5.0 exposure is absorbed when min_exposure=100."""
        # Three groups: group 1 has tiny exposure
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([200.0, 200.0, 5.0, 300.0, 300.0])
        coef = np.array([-1.0, -1.0, -0.95, -0.5, -0.5])

        result = enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        # Group 1 (coef -0.95) is nearest to group 0 (coef -1.0)
        # Should be absorbed into group 0
        assert result[2] == result[0]  # level 2 merged into group 0

    def test_under_exposure_goes_to_nearest(self):
        """Small group absorbs into nearest-coefficient neighbour, not nearest level."""
        # Groups: 0 (coef 0.0), 1 (coef 1.9, tiny exposure), 2 (coef 2.0)
        # Nearest coefficient to group 1 is group 2
        groups = np.array([0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([500.0, 10.0, 600.0, 600.0])
        coef = np.array([0.0, 1.9, 2.0, 2.0])

        result = enforce_min_exposure(groups, exposure, coef, min_exposure=500.0)
        # Group 1 (level 1) should merge to group 2 (coefficient distance 0.1)
        # not group 0 (coefficient distance 1.9)
        assert result[1] == result[2]
        assert result[1] != result[0]

    def test_all_adequate(self):
        """No merging when all groups meet threshold."""
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        exposure = np.array([200.0, 200.0, 300.0, 300.0, 400.0, 400.0])
        coef = np.zeros(6)
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=100.0)
        # Groups preserved (though labels may differ)
        unique_in = len(np.unique(groups))
        unique_out = len(np.unique(result))
        assert unique_out == unique_in

    def test_iterative_absorption(self):
        """Multiple rounds: first merge creates a still-undersized group."""
        # Group 0: 50, Group 1: 30, Group 2: 200
        # min_exposure=60 → group 1 absorbs into 0 (nearest) → group 0+1 has 80 ≥ 60
        groups = np.array([0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([50.0, 30.0, 200.0, 200.0])
        coef = np.array([-1.0, -1.05, -0.5, -0.5])
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=60.0)
        # Final should have 2 groups
        assert len(np.unique(result)) == 2

    def test_single_group_no_change(self):
        """Single group cannot be absorbed further."""
        groups = np.array([0, 0, 0], dtype=np.int32)
        exposure = np.array([10.0, 10.0, 10.0])
        coef = np.zeros(3)
        result = enforce_min_exposure(groups, exposure, coef, min_exposure=1000.0)
        assert np.all(result == 0)


class TestComputeGroupExposure:
    def test_basic(self):
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        exposure = np.array([100.0, 50.0, 200.0, 80.0, 120.0])
        result = compute_group_exposure(groups, exposure)
        assert result[0] == pytest.approx(150.0)
        assert result[1] == pytest.approx(200.0)
        assert result[2] == pytest.approx(200.0)


class TestRelabelGroupsContiguous:
    def test_already_contiguous(self):
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert np.array_equal(result, groups)

    def test_with_gaps(self):
        groups = np.array([0, 3, 3, 5, 5], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert list(result) == [0, 1, 1, 2, 2]

    def test_single_group(self):
        groups = np.array([7, 7, 7], dtype=np.int32)
        result = relabel_groups_contiguous(groups)
        assert np.all(result == 0)
