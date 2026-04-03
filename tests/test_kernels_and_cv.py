"""Unit tests for insurance_glm_tools.robust._kernels and _cv.

The kernel functions are pure numerical utilities — they're easy to test
with exact expectations. The CV utilities (select_lambda, make_lambda_grid)
are tested at the unit level using pre-built DataFrames and mocked families
rather than running full ADMM fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.robust._kernels import (
    gaussian_kernel,
    gaussian_kernel_matrix,
    median_bandwidth_y,
    median_bandwidth_x,
    mmd_loss_per_obs,
    total_mmd_loss,
)
from insurance_glm_tools.robust._cv import select_lambda


# ---------------------------------------------------------------------------
# gaussian_kernel
# ---------------------------------------------------------------------------


class TestGaussianKernel:

    def test_self_similarity_is_one(self):
        """K(a, a) = 1 for any a and h."""
        a = np.array([0.0, 1.0, -2.5, 100.0])
        result = gaussian_kernel(a, a, h=1.0)
        np.testing.assert_allclose(result, np.ones(4))

    def test_symmetry(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 1.0, 4.0])
        np.testing.assert_allclose(gaussian_kernel(a, b, h=1.5), gaussian_kernel(b, a, h=1.5))

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        k = gaussian_kernel(a, b, h=1.0)
        assert np.all(k >= 0) and np.all(k <= 1.0)

    def test_decreases_with_distance(self):
        """Larger |a-b| => smaller kernel value."""
        h = 1.0
        k_near = gaussian_kernel(np.array([0.0]), np.array([0.5]), h)
        k_far = gaussian_kernel(np.array([0.0]), np.array([5.0]), h)
        assert k_near > k_far

    def test_bandwidth_h_one(self):
        a = np.array([0.0])
        b = np.array([1.0])
        expected = np.exp(-0.5)
        np.testing.assert_allclose(gaussian_kernel(a, b, h=1.0), [expected])

    @pytest.mark.parametrize("h", [0.1, 1.0, 5.0, 100.0])
    def test_various_bandwidths(self, h):
        """Kernel always in (0, 1] for non-identical points."""
        a = np.array([0.0])
        b = np.array([1.0])
        k = gaussian_kernel(a, b, h)
        assert 0 < k <= 1.0

    def test_wider_bandwidth_gives_higher_value(self):
        """Wider h => less sensitive to distance => larger kernel."""
        a = np.array([0.0])
        b = np.array([2.0])
        k_narrow = gaussian_kernel(a, b, h=0.5)
        k_wide = gaussian_kernel(a, b, h=5.0)
        assert k_wide > k_narrow


# ---------------------------------------------------------------------------
# gaussian_kernel_matrix
# ---------------------------------------------------------------------------


class TestGaussianKernelMatrix:

    def test_shape(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        K = gaussian_kernel_matrix(X, h=1.0)
        assert K.shape == (4, 4)

    def test_diagonal_is_one(self):
        X = np.random.default_rng(0).standard_normal(10)
        K = gaussian_kernel_matrix(X, h=1.0)
        np.testing.assert_allclose(np.diag(K), np.ones(10))

    def test_symmetric(self):
        X = np.array([1.0, 2.0, 5.0])
        K = gaussian_kernel_matrix(X, h=1.0)
        np.testing.assert_allclose(K, K.T)

    def test_positive_semidefinite(self):
        """Gram matrix of RBF kernel is PSD."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal(8)
        K = gaussian_kernel_matrix(X, h=1.0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_all_entries_in_range(self):
        X = np.array([0.0, 1.0, 10.0, -5.0])
        K = gaussian_kernel_matrix(X, h=2.0)
        assert np.all(K >= 0) and np.all(K <= 1.0 + 1e-10)

    def test_single_element(self):
        K = gaussian_kernel_matrix(np.array([3.0]), h=1.0)
        assert K.shape == (1, 1)
        assert K[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# median_bandwidth_y
# ---------------------------------------------------------------------------


class TestMedianBandwidthY:

    def test_positive_result(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        h = median_bandwidth_y(y)
        assert h > 0

    def test_single_element_returns_one(self):
        h = median_bandwidth_y(np.array([5.0]))
        assert h == pytest.approx(1.0)

    def test_empty_triggers_fallback(self):
        """n=0 edge: returns 1.0."""
        h = median_bandwidth_y(np.array([]))
        assert h == pytest.approx(1.0)

    def test_all_identical_returns_positive(self):
        """Identical values => median distance = 0 => fallback to std/unit."""
        y = np.ones(20) * 3.0
        h = median_bandwidth_y(y)
        assert h > 0

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        y = np.random.default_rng(7).standard_normal(200)
        h1 = median_bandwidth_y(y, rng=rng1)
        h2 = median_bandwidth_y(y, rng=rng2)
        assert h1 == pytest.approx(h2)

    def test_scale_invariant_up_to_scale(self):
        """Bandwidth should scale proportionally with the data scale."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        h1 = median_bandwidth_y(y, rng=np.random.default_rng(0))
        h2 = median_bandwidth_y(10.0 * y, rng=np.random.default_rng(0))
        assert h2 == pytest.approx(10.0 * h1, rel=0.1)

    def test_large_n_uses_subsampling(self):
        """n*(n-1)/2 > max_pairs should trigger subsampling — result still positive."""
        rng = np.random.default_rng(5)
        y = rng.standard_normal(500)
        h = median_bandwidth_y(y, max_pairs=100, rng=rng)
        assert h > 0

    def test_small_n_uses_all_pairs(self):
        """n*(n-1)/2 <= max_pairs should use exact computation."""
        y = np.array([1.0, 3.0, 6.0])  # 3 pairs: |1-3|=2, |1-6|=5, |3-6|=3 → median=3
        h = median_bandwidth_y(y, max_pairs=10)
        assert h == pytest.approx(3.0)

    def test_two_element_array(self):
        y = np.array([2.0, 5.0])
        h = median_bandwidth_y(y)
        assert h == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# median_bandwidth_x
# ---------------------------------------------------------------------------


class TestMedianBandwidthX:

    def test_positive_result(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        h = median_bandwidth_x(X)
        assert h > 0

    def test_single_row(self):
        X = np.array([[1.0, 2.0, 3.0]])
        h = median_bandwidth_x(X)
        assert h == pytest.approx(1.0)

    def test_identical_rows_fallback(self):
        """All identical rows => median distance = 0 => fallback to 1.0."""
        X = np.ones((20, 3)) * 2.0
        h = median_bandwidth_x(X)
        assert h > 0

    def test_subsampling_for_large_n(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 4))
        h = median_bandwidth_x(X, max_pairs=100, rng=rng)
        assert h > 0

    def test_reproducible_with_rng(self):
        rng = np.random.default_rng(0)
        X = np.random.default_rng(1).standard_normal((100, 3))
        h1 = median_bandwidth_x(X, rng=np.random.default_rng(42))
        h2 = median_bandwidth_x(X, rng=np.random.default_rng(42))
        assert h1 == pytest.approx(h2)

    def test_1d_feature(self):
        """Single-feature X should work (Euclidean distance = absolute value)."""
        X = np.array([[1.0], [3.0], [6.0]])  # dists: 2, 5, 3 → median=3
        h = median_bandwidth_x(X, max_pairs=10)
        assert h == pytest.approx(3.0)

    def test_exact_pairs_computation(self):
        """3 rows: n*(n-1)/2 = 3 pairs, all used."""
        X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        # Distances: 5, 0, 5 → median = 5
        h = median_bandwidth_x(X, max_pairs=100)
        assert h == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# mmd_loss_per_obs and total_mmd_loss
# ---------------------------------------------------------------------------


class TestMmdLoss:

    def test_mmd_loss_per_obs_formula(self):
        """l_tilde = 1 - 2 * ek_y."""
        ek_y = np.array([0.3, 0.5, 0.8])
        expected = np.array([1 - 0.6, 1 - 1.0, 1 - 1.6])
        np.testing.assert_allclose(mmd_loss_per_obs(ek_y), expected)

    def test_mmd_loss_per_obs_zero_ek(self):
        """ek_y=0 (perfect mismatch) => loss=1."""
        ek_y = np.zeros(5)
        loss = mmd_loss_per_obs(ek_y)
        np.testing.assert_allclose(loss, np.ones(5))

    def test_mmd_loss_per_obs_perfect_fit(self):
        """ek_y=1 (kernel = 1) => loss = -1."""
        ek_y = np.ones(5)
        loss = mmd_loss_per_obs(ek_y)
        np.testing.assert_allclose(loss, -np.ones(5))

    def test_total_mmd_loss_is_mean(self):
        ek_y = np.array([0.2, 0.4, 0.6])
        per_obs = mmd_loss_per_obs(ek_y)
        expected_total = float(np.mean(per_obs))
        assert total_mmd_loss(ek_y) == pytest.approx(expected_total)

    def test_total_mmd_loss_scalar(self):
        ek_y = np.ones(10) * 0.5
        result = total_mmd_loss(ek_y)
        assert isinstance(result, float)

    def test_total_mmd_loss_range(self):
        """MMD loss per obs is in [-1, 1]; mean should also be in [-1, 1]."""
        rng = np.random.default_rng(0)
        ek_y = rng.uniform(0.0, 1.0, 100)
        loss = total_mmd_loss(ek_y)
        assert -1.0 <= loss <= 1.0

    def test_better_fit_lower_loss(self):
        """Higher ek_y (better fit) should give lower total loss."""
        ek_good = np.array([0.8, 0.9, 0.85])
        ek_bad = np.array([0.1, 0.2, 0.15])
        assert total_mmd_loss(ek_good) < total_mmd_loss(ek_bad)

    def test_single_obs(self):
        ek_y = np.array([0.6])
        assert total_mmd_loss(ek_y) == pytest.approx(1 - 2 * 0.6)


# ---------------------------------------------------------------------------
# select_lambda (from _cv.py)
# ---------------------------------------------------------------------------


class TestSelectLambda:

    def test_selects_minimum(self):
        """select_lambda picks the lambda with lowest mean_mmd_loss."""
        df = pd.DataFrame({
            "lambda": [1.0, 0.5, 0.1, 0.01],
            "mean_mmd_loss": [0.5, 0.3, 0.1, 0.4],
            "std_mmd_loss": [0.0, 0.0, 0.0, 0.0],
        })
        lam = select_lambda(df)
        assert lam == pytest.approx(0.1)

    def test_single_row(self):
        df = pd.DataFrame({
            "lambda": [0.5],
            "mean_mmd_loss": [0.2],
            "std_mmd_loss": [0.0],
        })
        lam = select_lambda(df)
        assert lam == pytest.approx(0.5)

    def test_empty_dataframe_fallback(self):
        """Empty CV results => fallback to 0.01."""
        df = pd.DataFrame({"lambda": [], "mean_mmd_loss": [], "std_mmd_loss": []})
        lam = select_lambda(df)
        assert lam == pytest.approx(0.01)

    def test_all_equal_losses(self):
        """When all losses equal, first minimum is selected."""
        df = pd.DataFrame({
            "lambda": [2.0, 1.0, 0.5],
            "mean_mmd_loss": [0.3, 0.3, 0.3],
        })
        lam = select_lambda(df)
        assert lam in df["lambda"].values

    def test_returns_float(self):
        df = pd.DataFrame({
            "lambda": [0.1, 0.5],
            "mean_mmd_loss": [0.2, 0.8],
        })
        result = select_lambda(df)
        assert isinstance(result, float)

    def test_minimum_at_extremes(self):
        """Best lambda could be at either end of the grid."""
        df_left = pd.DataFrame({
            "lambda": [10.0, 1.0, 0.1],
            "mean_mmd_loss": [0.05, 0.3, 0.5],
        })
        assert select_lambda(df_left) == pytest.approx(10.0)

        df_right = pd.DataFrame({
            "lambda": [10.0, 1.0, 0.1],
            "mean_mmd_loss": [0.5, 0.3, 0.05],
        })
        assert select_lambda(df_right) == pytest.approx(0.1)

    def test_negative_losses_handled(self):
        """MMD loss can be negative; still selects global minimum."""
        df = pd.DataFrame({
            "lambda": [1.0, 0.5, 0.1],
            "mean_mmd_loss": [-0.2, -0.5, -0.3],
        })
        lam = select_lambda(df)
        assert lam == pytest.approx(0.5)
