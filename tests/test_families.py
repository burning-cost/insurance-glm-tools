"""Unit tests for insurance_glm_tools.robust._families.

Tests cover each family class's core methods (mean, eta, ek_y, dek_dmu,
init_params) plus the factory function. The families are the numerical heart
of RobustMMDGLM — they compute E[K_y] which drives the ADMM theta-step. We
test correctness, boundary conditions, and that the gradient is consistent
with finite-difference checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_glm_tools.robust._families import (
    GaussianFamily,
    LogisticFamily,
    PoissonFamily,
    GammaFamily,
    get_family,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_gradient(family, mu, y_obs, h_y, delta=1e-5):
    """Finite-difference approximation of d(ek_y)/d(mu)."""
    mu_plus = mu + delta
    mu_minus = np.maximum(mu - delta, 1e-12)
    ek_plus = family.ek_y(mu_plus, y_obs, h_y)
    ek_minus = family.ek_y(mu_minus, y_obs, h_y)
    return (ek_plus - ek_minus) / (2.0 * delta)


# ---------------------------------------------------------------------------
# GaussianFamily
# ---------------------------------------------------------------------------


class TestGaussianFamily:

    def test_mean_is_identity(self):
        fam = GaussianFamily(sigma=1.0)
        eta = np.array([-2.0, 0.0, 3.5])
        np.testing.assert_array_equal(fam.mean(eta), eta)

    def test_eta_is_identity(self):
        fam = GaussianFamily(sigma=1.0)
        mu = np.array([1.0, -1.0, 0.5])
        np.testing.assert_array_equal(fam.eta(mu), mu)

    def test_mean_eta_roundtrip(self):
        fam = GaussianFamily(sigma=1.5)
        mu = np.array([2.0, -1.0, 0.0])
        np.testing.assert_allclose(fam.mean(fam.eta(mu)), mu)

    def test_ek_y_at_perfect_fit(self):
        """When mu == y_obs, ek_y = h / sqrt(h^2 + sigma^2) (maximum value)."""
        fam = GaussianFamily(sigma=1.0)
        h = 1.0
        mu = np.array([2.0])
        y = np.array([2.0])
        ek = fam.ek_y(mu, y, h)
        expected = h / np.sqrt(h**2 + fam.sigma**2)
        np.testing.assert_allclose(ek, [expected], rtol=1e-10)

    def test_ek_y_decreases_with_distance(self):
        """ek_y should decrease as |y - mu| increases."""
        fam = GaussianFamily(sigma=1.0)
        y = np.array([0.0, 0.0, 0.0])
        mu_near = np.array([0.5, 1.0, 2.0])
        ek = fam.ek_y(mu_near, y, h_y=1.0)
        assert ek[0] > ek[1] > ek[2]

    def test_ek_y_shape(self):
        fam = GaussianFamily()
        n = 50
        rng = np.random.default_rng(0)
        mu = rng.standard_normal(n)
        y = rng.standard_normal(n)
        ek = fam.ek_y(mu, y, h_y=1.0)
        assert ek.shape == (n,)
        assert np.all(ek > 0)

    def test_ek_y_bounded_above_by_one(self):
        """Gaussian kernel is at most 1, so ek_y <= h/sqrt(h^2 + sigma^2) < 1."""
        fam = GaussianFamily(sigma=0.5)
        rng = np.random.default_rng(1)
        mu = rng.standard_normal(100)
        y = rng.standard_normal(100)
        ek = fam.ek_y(mu, y, h_y=1.0)
        assert np.all(ek <= 1.0 + 1e-10)

    def test_dek_dmu_sign(self):
        """When y > mu, dek/dmu > 0 (moving mu toward y increases ek_y)."""
        fam = GaussianFamily(sigma=1.0)
        # y=2, mu=0: moving mu up toward y increases ek_y
        mu = np.array([0.0])
        y = np.array([2.0])
        grad = fam.dek_dmu(mu, y, h_y=1.0)
        assert grad[0] > 0

    def test_dek_dmu_shape(self):
        fam = GaussianFamily()
        mu = np.ones(30)
        y = np.zeros(30)
        grad = fam.dek_dmu(mu, y, h_y=1.0)
        assert grad.shape == (30,)

    def test_dek_dmu_consistent_with_fd(self):
        """dek_dmu should match finite-difference gradient."""
        fam = GaussianFamily(sigma=1.0)
        rng = np.random.default_rng(42)
        mu = rng.uniform(0.5, 2.0, 20)
        y = rng.standard_normal(20)
        analytic = fam.dek_dmu(mu, y, h_y=1.0)
        fd = _fd_gradient(fam, mu, y, h_y=1.0)
        np.testing.assert_allclose(analytic, fd, rtol=1e-4)

    def test_init_params_returns_constant(self):
        fam = GaussianFamily()
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        init = fam.init_params(y, exposure=None)
        assert init.shape == (100,)
        assert np.all(init == pytest.approx(np.mean(y)))

    def test_init_params_ignores_exposure(self):
        """Gaussian init_params ignores exposure (no link needed)."""
        fam = GaussianFamily()
        y = np.array([1.0, 2.0, 3.0])
        exp = np.array([2.0, 1.0, 0.5])
        init_with = fam.init_params(y, exposure=exp)
        init_without = fam.init_params(y, exposure=None)
        np.testing.assert_array_equal(init_with, init_without)

    def test_update_sigma_changes_value(self):
        fam = GaussianFamily(sigma=1.0)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200) * 3.0  # std ~3
        mu = np.zeros(200)
        fam.update_sigma(y, mu)
        assert fam.sigma > 1.0  # should be closer to 3

    def test_update_sigma_constant_y(self):
        """Constant y => residuals all zero => sigma clamped to 1e-8."""
        fam = GaussianFamily(sigma=1.0)
        y = np.ones(50) * 5.0
        mu = np.ones(50) * 5.0
        fam.update_sigma(y, mu)
        assert fam.sigma == pytest.approx(1e-8)

    def test_update_sigma_positive(self):
        fam = GaussianFamily(sigma=0.0)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.zeros(4)
        fam.update_sigma(y, mu)
        assert fam.sigma > 0


# ---------------------------------------------------------------------------
# LogisticFamily
# ---------------------------------------------------------------------------


class TestLogisticFamily:

    def test_mean_sigmoid(self):
        fam = LogisticFamily()
        eta = np.array([0.0, 1.0, -1.0, 10.0, -10.0])
        mu = fam.mean(eta)
        assert np.all(mu > 0) and np.all(mu < 1)
        assert mu[0] == pytest.approx(0.5)
        assert mu[3] > 0.99
        assert mu[4] < 0.01

    def test_eta_logit(self):
        fam = LogisticFamily()
        mu = np.array([0.5, 0.9, 0.1])
        eta = fam.eta(mu)
        assert eta[0] == pytest.approx(0.0, abs=1e-8)
        assert eta[1] > 0
        assert eta[2] < 0

    def test_mean_eta_roundtrip(self):
        fam = LogisticFamily()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(fam.mean(fam.eta(mu)), mu, rtol=1e-6)

    def test_eta_clips_extreme_mu(self):
        """eta should handle mu=0 and mu=1 without -inf/inf."""
        fam = LogisticFamily()
        mu = np.array([0.0, 1.0])
        eta = fam.eta(mu)
        assert np.all(np.isfinite(eta))

    def test_ek_y_binary_responses(self):
        """For binary y_obs in {0, 1}, ek_y is a convex combination of two kernels."""
        fam = LogisticFamily()
        mu = np.array([0.5, 0.5])
        y = np.array([0.0, 1.0])
        ek = fam.ek_y(mu, y, h_y=1.0)
        # Both values should be the same since mu=0.5 and y={0,1}
        # ek_y = mu*K(1,y) + (1-mu)*K(0,y) with K = Gaussian
        assert np.all(ek > 0)
        assert np.all(ek <= 1.0)

    def test_ek_y_shape(self):
        fam = LogisticFamily()
        n = 40
        rng = np.random.default_rng(1)
        mu = rng.uniform(0.1, 0.9, n)
        y = rng.integers(0, 2, n).astype(float)
        ek = fam.ek_y(mu, y, h_y=1.0)
        assert ek.shape == (n,)

    def test_dek_dmu_consistent_with_fd(self):
        fam = LogisticFamily()
        rng = np.random.default_rng(7)
        mu = rng.uniform(0.2, 0.8, 20)
        y = rng.integers(0, 2, 20).astype(float)
        analytic = fam.dek_dmu(mu, y, h_y=1.0)
        fd = _fd_gradient(fam, mu, y, h_y=1.0)
        np.testing.assert_allclose(analytic, fd, rtol=1e-4)

    def test_dek_dmu_when_y_equals_1(self):
        """When y=1, kernel(1,y)=1 (max) and kernel(0,y) < 1, so gradient > 0."""
        fam = LogisticFamily()
        mu = np.array([0.4])
        y = np.array([1.0])
        grad = fam.dek_dmu(mu, y, h_y=0.5)
        # K(1,1)=1, K(0,1)=exp(-1/(2*0.25)) < 1, so grad = 1 - K(0,1) > 0
        assert grad[0] > 0

    def test_init_params_in_range(self):
        fam = LogisticFamily()
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.3, 200).astype(float)
        init = fam.init_params(y, exposure=None)
        assert init.shape == (200,)
        assert np.all(init > 0) and np.all(init < 1)

    def test_init_params_clips_extreme(self):
        """With all-zero y, mean=0 but should be clipped to 0.01."""
        fam = LogisticFamily()
        y = np.zeros(50)
        init = fam.init_params(y, exposure=None)
        assert np.all(init >= 0.01)


# ---------------------------------------------------------------------------
# PoissonFamily
# ---------------------------------------------------------------------------


class TestPoissonFamily:

    def test_mean_exp_link(self):
        fam = PoissonFamily()
        eta = np.array([0.0, 1.0, -1.0])
        mu = fam.mean(eta)
        np.testing.assert_allclose(mu, np.exp(eta), rtol=1e-10)

    def test_mean_clips_extreme_eta(self):
        """Very large/small eta should not produce inf/nan."""
        fam = PoissonFamily()
        eta = np.array([-100.0, 0.0, 100.0])
        mu = fam.mean(eta)
        assert np.all(np.isfinite(mu))
        assert np.all(mu > 0)

    def test_eta_log_link(self):
        fam = PoissonFamily()
        mu = np.array([1.0, 2.0, 0.5])
        eta = fam.eta(mu)
        np.testing.assert_allclose(eta, np.log(mu), rtol=1e-10)

    def test_eta_clips_zero_mu(self):
        fam = PoissonFamily()
        mu = np.array([0.0])
        eta = fam.eta(mu)
        assert np.isfinite(eta[0])

    def test_ek_y_non_negative(self):
        fam = PoissonFamily()
        rng = np.random.default_rng(0)
        mu = rng.uniform(0.5, 5.0, 20)
        y = rng.poisson(2.0, 20).astype(float)
        ek = fam.ek_y(mu, y, h_y=1.0)
        assert np.all(ek >= 0)

    def test_ek_y_shape(self):
        fam = PoissonFamily()
        n = 15
        mu = np.ones(n) * 2.0
        y = np.arange(n, dtype=float)
        ek = fam.ek_y(mu, y, h_y=1.5)
        assert ek.shape == (n,)

    def test_ek_y_near_zero_mu(self):
        """Very small mu — all mass near 0. ek_y ≈ exp(-y_obs^2 / 2h^2)."""
        fam = PoissonFamily()
        mu = np.array([1e-12])
        y = np.array([0.0])
        h = 1.0
        ek = fam.ek_y(mu, y, h)
        # When mu~0, Y~0 almost surely, so E[K_y(Y, y=0)] ≈ K_y(0, 0) = 1
        assert ek[0] == pytest.approx(1.0, abs=0.01)

    def test_ek_y_finite_for_large_mu(self):
        """Large mu should not produce inf or nan."""
        fam = PoissonFamily()
        mu = np.array([100.0, 200.0])
        y = np.array([95.0, 190.0])
        ek = fam.ek_y(mu, y, h_y=5.0)
        assert np.all(np.isfinite(ek))

    def test_dek_dmu_consistent_with_fd(self):
        """Finite-difference check on PoissonFamily.dek_dmu (which itself uses FD)."""
        fam = PoissonFamily()
        rng = np.random.default_rng(11)
        mu = rng.uniform(1.0, 5.0, 10)
        y = rng.poisson(3.0, 10).astype(float)
        analytic = fam.dek_dmu(mu, y, h_y=1.5)
        fd = _fd_gradient(fam, mu, y, h_y=1.5, delta=1e-4)
        np.testing.assert_allclose(analytic, fd, rtol=0.05)  # FD methods: wider tol

    def test_init_params_no_exposure(self):
        fam = PoissonFamily()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        init = fam.init_params(y, exposure=None)
        assert init.shape == (4,)
        assert np.all(init == pytest.approx(np.mean(y)))

    def test_init_params_with_exposure(self):
        """With exposure, init = sum(y)/sum(exposure)."""
        fam = PoissonFamily()
        y = np.array([2.0, 4.0])
        exposure = np.array([2.0, 4.0])  # rate = 1.0
        init = fam.init_params(y, exposure)
        expected_rate = y.sum() / exposure.sum()
        assert np.all(init == pytest.approx(expected_rate))

    def test_init_params_positive(self):
        """Init params must always be positive."""
        fam = PoissonFamily()
        y = np.zeros(30)
        init = fam.init_params(y, exposure=None)
        assert np.all(init > 0)

    def test_max_k_cap(self):
        """Very large mu should not cause memory errors (MAX_K=500 cap)."""
        fam = PoissonFamily()
        mu = np.array([1000.0])
        y = np.array([1000.0])
        ek = fam.ek_y(mu, y, h_y=20.0)
        assert np.isfinite(ek[0])


# ---------------------------------------------------------------------------
# GammaFamily
# ---------------------------------------------------------------------------


class TestGammaFamily:

    def test_mean_exp_link(self):
        fam = GammaFamily(alpha=2.0)
        eta = np.array([0.0, 1.0, -0.5])
        np.testing.assert_allclose(fam.mean(eta), np.exp(eta), rtol=1e-10)

    def test_mean_clips_extreme(self):
        fam = GammaFamily(alpha=2.0)
        eta = np.array([-50.0, 0.0, 50.0])
        mu = fam.mean(eta)
        assert np.all(np.isfinite(mu))
        assert np.all(mu > 0)

    def test_eta_log_link(self):
        fam = GammaFamily(alpha=2.0)
        mu = np.array([1.0, 100.0, 0.01])
        np.testing.assert_allclose(fam.eta(mu), np.log(mu), rtol=1e-10)

    def test_ek_y_non_negative(self):
        fam = GammaFamily(alpha=3.0, n_quadrature=16)
        rng = np.random.default_rng(5)
        mu = rng.uniform(1.0, 10.0, 15)
        y = rng.gamma(3.0, 2.0, 15)
        ek = fam.ek_y(mu, y, h_y=2.0)
        assert np.all(ek >= 0)

    def test_ek_y_shape(self):
        fam = GammaFamily(alpha=2.0, n_quadrature=8)
        n = 10
        mu = np.ones(n) * 5.0
        y = np.linspace(1.0, 10.0, n)
        ek = fam.ek_y(mu, y, h_y=2.0)
        assert ek.shape == (n,)

    def test_ek_y_finite(self):
        fam = GammaFamily(alpha=2.0, n_quadrature=16)
        rng = np.random.default_rng(0)
        mu = rng.uniform(0.1, 100.0, 20)
        y = rng.gamma(2.0, 1.0, 20)
        ek = fam.ek_y(mu, y, h_y=5.0)
        assert np.all(np.isfinite(ek))

    def test_dek_dmu_consistent_with_fd(self):
        fam = GammaFamily(alpha=3.0, n_quadrature=16)
        rng = np.random.default_rng(99)
        mu = rng.uniform(2.0, 8.0, 8)
        y = rng.gamma(3.0, 2.0, 8)
        analytic = fam.dek_dmu(mu, y, h_y=2.0)
        fd = _fd_gradient(fam, mu, y, h_y=2.0, delta=1e-4)
        np.testing.assert_allclose(analytic, fd, rtol=0.05)

    def test_init_params_positive(self):
        fam = GammaFamily(alpha=2.0)
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 100.0, 50)
        init = fam.init_params(y, exposure=None)
        assert init.shape == (50,)
        assert np.all(init > 0)

    def test_init_params_equals_mean_y(self):
        fam = GammaFamily(alpha=2.0)
        y = np.array([100.0, 200.0, 300.0])
        init = fam.init_params(y, exposure=None)
        expected = np.mean(y)
        assert np.all(init == pytest.approx(expected))

    def test_update_alpha_from_residuals(self):
        """update_alpha should produce a positive alpha from noisy data."""
        fam = GammaFamily(alpha=1.0, n_quadrature=16)
        rng = np.random.default_rng(0)
        true_alpha = 5.0
        mu = np.ones(500) * 100.0
        y = rng.gamma(shape=true_alpha, scale=100.0 / true_alpha, size=500)
        fam.update_alpha(y, mu)
        assert fam.alpha > 0
        assert np.isfinite(fam.alpha)

    def test_update_alpha_clips_to_bounds(self):
        """alpha is clipped to [0.1, 100]."""
        fam = GammaFamily(alpha=1.0)
        # CV^2 very small → alpha would be huge → clip to 100
        y = np.ones(50) * 10.0 + np.random.default_rng(0).normal(0, 1e-6, 50)
        mu = np.ones(50) * 10.0
        fam.update_alpha(y, mu)
        assert fam.alpha <= 100.0 + 1e-10

    def test_different_n_quadrature(self):
        """n_quadrature affects precision but not validity."""
        y_obs = np.array([5.0, 10.0])
        mu = np.array([5.0, 10.0])
        ek_low = GammaFamily(alpha=2.0, n_quadrature=8).ek_y(mu, y_obs, h_y=2.0)
        ek_high = GammaFamily(alpha=2.0, n_quadrature=32).ek_y(mu, y_obs, h_y=2.0)
        # Both should be finite and positive
        assert np.all(np.isfinite(ek_low)) and np.all(np.isfinite(ek_high))
        # 8-point GL is coarse; use rtol=0.35 to avoid spurious failures while
        # still catching gross errors. 32-point GL is the reference.
        np.testing.assert_allclose(ek_low, ek_high, rtol=0.35)


# ---------------------------------------------------------------------------
# get_family factory
# ---------------------------------------------------------------------------


class TestGetFamilyFactory:

    @pytest.mark.parametrize("name", ["gaussian", "logistic", "binomial", "poisson", "gamma"])
    def test_valid_families(self, name):
        fam = get_family(name)
        assert hasattr(fam, "ek_y")
        assert hasattr(fam, "mean")

    def test_gaussian_alias_lowercase(self):
        fam = get_family("gaussian")
        assert isinstance(fam, GaussianFamily)

    def test_logistic_alias(self):
        fam = get_family("logistic")
        assert isinstance(fam, LogisticFamily)

    def test_binomial_alias(self):
        """binomial is an alias for logistic."""
        fam = get_family("binomial")
        assert isinstance(fam, LogisticFamily)

    def test_poisson(self):
        fam = get_family("poisson")
        assert isinstance(fam, PoissonFamily)

    def test_gamma(self):
        fam = get_family("gamma")
        assert isinstance(fam, GammaFamily)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            get_family("tweedie")

    def test_unknown_family_message_contains_name(self):
        with pytest.raises(ValueError, match="negbinom"):
            get_family("negbinom")

    def test_case_insensitive(self):
        """Family names should be case-insensitive."""
        fam = get_family("Gaussian")
        assert isinstance(fam, GaussianFamily)
        fam2 = get_family("POISSON")
        assert isinstance(fam2, PoissonFamily)

    def test_gamma_kwargs_passed(self):
        """kwargs like n_quadrature should be forwarded to GammaFamily."""
        fam = get_family("gamma", n_quadrature=8, alpha=5.0)
        assert isinstance(fam, GammaFamily)
        assert fam.n_quadrature == 8
        assert fam.alpha == pytest.approx(5.0)

    def test_gaussian_kwargs_passed(self):
        fam = get_family("gaussian", sigma=2.5)
        assert isinstance(fam, GaussianFamily)
        assert fam.sigma == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Cross-family sanity checks
# ---------------------------------------------------------------------------


class TestFamilyCrossSanity:
    """Property-based invariants that should hold for all families."""

    @pytest.fixture(params=["gaussian", "logistic", "poisson", "gamma"])
    def family(self, request):
        return get_family(request.param)

    def test_ek_y_positive_everywhere(self, family):
        """ek_y should always be positive (it's an expectation of a positive kernel)."""
        rng = np.random.default_rng(0)
        n = 30

        if isinstance(family, LogisticFamily):
            mu = rng.uniform(0.05, 0.95, n)
            y = rng.integers(0, 2, n).astype(float)
        elif isinstance(family, GaussianFamily):
            mu = rng.standard_normal(n)
            y = rng.standard_normal(n)
        else:
            mu = rng.uniform(0.5, 5.0, n)
            y = rng.gamma(2.0, 1.0, n)

        ek = family.ek_y(mu, y, h_y=1.0)
        assert np.all(ek > 0), f"{type(family).__name__}: ek_y contains non-positive values"

    def test_ek_y_finite_everywhere(self, family):
        """ek_y should always be finite."""
        rng = np.random.default_rng(1)
        n = 30

        if isinstance(family, LogisticFamily):
            mu = rng.uniform(0.05, 0.95, n)
            y = rng.integers(0, 2, n).astype(float)
        elif isinstance(family, GaussianFamily):
            mu = rng.standard_normal(n)
            y = rng.standard_normal(n)
        else:
            mu = rng.uniform(0.5, 5.0, n)
            y = rng.gamma(2.0, 1.0, n)

        ek = family.ek_y(mu, y, h_y=1.0)
        assert np.all(np.isfinite(ek)), f"{type(family).__name__}: ek_y contains non-finite values"
