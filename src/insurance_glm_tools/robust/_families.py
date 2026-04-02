"""GLM family objects for RobustMMDGLM.

Each family encapsulates:
- The link function and its inverse (the mean function)
- E[K_y(Y, y_obs)] — the expected kernel value under the predicted distribution
- Gradient of E[K_y] with respect to mu (used in the MMD theta-step)

Design note: we keep families as plain dataclasses rather than class hierarchies
to make it easy to inspect exactly what each one does. The Gaussian and Logistic
families have closed-form E[K_y]. Poisson uses truncated summation. Gamma uses
Gauss-Laguerre quadrature — this is the main complexity.

Reference: Kang & Kang (2026), arXiv:2602.21132.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, roots_laguerre
from scipy.stats import poisson as poisson_dist
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------

class Family:
    """Abstract base. Subclasses must implement mean(), ek_y(), dek_dmu()."""

    def mean(self, eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eta(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ek_y(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        """E_{Y~F(mu)}[K_y(Y, y_obs)]. Shape: (n,)."""
        raise NotImplementedError

    def dek_dmu(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        """d/d_mu E[K_y]. Shape: (n,)."""
        raise NotImplementedError

    def init_params(self, y: np.ndarray, exposure: Optional[np.ndarray]) -> np.ndarray:
        """Return sensible initial mu values."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

class GaussianFamily(Family):
    """Identity link, additive noise N(0, sigma^2).

    E[K_y(Y, y_i)] = h_y / sqrt(h_y^2 + sigma^2)
                     * exp(-(y_i - mu_i)^2 / (2*(h_y^2 + sigma^2)))

    sigma^2 is estimated as the residual variance from the initialisation step
    and held fixed during ADMM. This is a deliberate simplification: estimating
    sigma jointly with theta via MMD is possible but adds substantial complexity.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def mean(self, eta: np.ndarray) -> np.ndarray:
        return eta  # identity link

    def eta(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def ek_y(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        sigma2 = self.sigma ** 2
        h2 = h_y ** 2
        denom = h2 + sigma2
        scale = h_y / np.sqrt(denom)
        exponent = -(y_obs - mu) ** 2 / (2.0 * denom)
        return scale * np.exp(exponent)

    def dek_dmu(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        sigma2 = self.sigma ** 2
        h2 = h_y ** 2
        denom = h2 + sigma2
        scale = h_y / np.sqrt(denom)
        diff = y_obs - mu
        ek = self.ek_y(mu, y_obs, h_y)
        # d/dmu of scale * exp(-(y - mu)^2 / 2*denom)
        # = ek * (y - mu) / denom  (note: derivative of -(y-mu)^2 wrt mu = 2(y-mu))
        return ek * diff / denom

    def init_params(self, y: np.ndarray, exposure: Optional[np.ndarray]) -> np.ndarray:
        return np.full(len(y), np.mean(y))

    def update_sigma(self, y: np.ndarray, mu: np.ndarray) -> None:
        """Update sigma from current residuals."""
        residuals = y - mu
        self.sigma = float(np.std(residuals))
        if self.sigma < 1e-8:
            self.sigma = 1e-8


# ---------------------------------------------------------------------------
# Logistic (Bernoulli)
# ---------------------------------------------------------------------------

class LogisticFamily(Family):
    """Logit link, Bernoulli response.

    E[K_y(Y, y_i)] = pi_i * K_y(1, y_i) + (1 - pi_i) * K_y(0, y_i)
    where pi_i = sigmoid(eta_i).
    """

    def mean(self, eta: np.ndarray) -> np.ndarray:
        return expit(eta)

    def eta(self, mu: np.ndarray) -> np.ndarray:
        mu_clipped = np.clip(mu, 1e-8, 1 - 1e-8)
        return np.log(mu_clipped / (1.0 - mu_clipped))

    def _kernel(self, a: float, b: np.ndarray, h_y: float) -> np.ndarray:
        return np.exp(-((a - b) ** 2) / (2.0 * h_y ** 2))

    def ek_y(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        k1 = self._kernel(1.0, y_obs, h_y)
        k0 = self._kernel(0.0, y_obs, h_y)
        return mu * k1 + (1.0 - mu) * k0

    def dek_dmu(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        k1 = self._kernel(1.0, y_obs, h_y)
        k0 = self._kernel(0.0, y_obs, h_y)
        return k1 - k0

    def init_params(self, y: np.ndarray, exposure: Optional[np.ndarray]) -> np.ndarray:
        p = float(np.mean(y))
        p = np.clip(p, 0.01, 0.99)
        return np.full(len(y), p)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

class PoissonFamily(Family):
    """Log link, Poisson response (with optional exposure offset).

    E[K_y(Y, y_i)] = sum_{k=0}^{K_max} K_y(k, y_i) * P(Y=k; mu_i)

    K_max is chosen per observation as the 1-1e-10 quantile of Poisson(mu_i).
    We cap at 500 to avoid memory blow-up for large mu.
    """

    MAX_K = 500

    def mean(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(eta, -30, 30))

    def eta(self, mu: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(mu, 1e-12))

    def ek_y(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        n = len(mu)
        result = np.zeros(n)
        h2_2 = 2.0 * h_y ** 2

        for i in range(n):
            lam = float(mu[i])
            if lam < 1e-10:
                result[i] = np.exp(-(y_obs[i]) ** 2 / h2_2)  # all mass at 0
                continue
            # Upper truncation at 1-1e-10 quantile, capped at MAX_K
            k_max = int(min(self.MAX_K, poisson_dist.ppf(1 - 1e-10, lam)))
            k_max = max(k_max, int(y_obs[i]) + 5)  # include the obs value
            k_max = min(k_max, self.MAX_K)
            ks = np.arange(0, k_max + 1, dtype=float)
            kernel_vals = np.exp(-((ks - y_obs[i]) ** 2) / h2_2)
            pmf_vals = poisson_dist.pmf(ks, lam)
            result[i] = float(np.dot(kernel_vals, pmf_vals))

        return result

    def dek_dmu(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        """d/d_mu E[K_y] using finite differences (numerically stable for Poisson)."""
        eps = np.maximum(mu * 1e-4, 1e-6)
        ek_plus = self.ek_y(mu + eps, y_obs, h_y)
        ek_minus = self.ek_y(np.maximum(mu - eps, 1e-12), y_obs, h_y)
        return (ek_plus - ek_minus) / (2.0 * eps)

    def init_params(self, y: np.ndarray, exposure: Optional[np.ndarray]) -> np.ndarray:
        rate = np.mean(y) if exposure is None else np.sum(y) / np.sum(exposure)
        rate = max(rate, 1e-6)
        return np.full(len(y), rate)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class GammaFamily(Family):
    """Log link, Gamma response.

    E[K_y(Y, y_i)] via Gauss-Laguerre quadrature.

    If Y ~ Gamma(alpha, mu) — i.e., mean=mu, shape=alpha — then the pdf is:
        f(y) = (alpha/mu)^alpha * y^(alpha-1) * exp(-alpha*y/mu) / Gamma(alpha)

    Substituting u = alpha*y/mu (so y = mu*u/alpha, dy = mu/alpha * du):
        E[K_y(Y, y_i)] = integral_0^inf K_y(mu*u/alpha, y_i) * u^(alpha-1) * exp(-u) / Gamma(alpha) du

    This is in Gauss-Laguerre form: integral_0^inf g(u) * exp(-u) du
    with g(u) = K_y(mu*u/alpha, y_i) * u^(alpha-1) / Gamma(alpha)

    alpha (dispersion shape parameter) is estimated from initialisation residuals
    and held fixed. Typically alpha in [1, 10] for insurance severity.
    """

    def __init__(self, alpha: float = 2.0, n_quadrature: int = 32):
        self.alpha = alpha
        self.n_quadrature = n_quadrature
        # Pre-compute quadrature nodes and weights
        self._nodes, self._weights = roots_laguerre(n_quadrature)

    def mean(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(eta, -30, 30))

    def eta(self, mu: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(mu, 1e-12))

    def ek_y(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        """Gauss-Laguerre quadrature for E[K_y]."""
        from scipy.special import gammaln
        alpha = self.alpha
        nodes = self._nodes  # shape (Q,)
        weights = self._weights  # shape (Q,)
        h2_2 = 2.0 * h_y ** 2
        log_gamma_alpha = float(gammaln(alpha))
        n = len(mu)
        result = np.zeros(n)

        for i in range(n):
            # y_q = mu_i * u / alpha for each quadrature node u
            y_q = mu[i] * nodes / alpha  # shape (Q,)
            kernel_q = np.exp(-((y_q - y_obs[i]) ** 2) / h2_2)
            # integrand without exp(-u): u^(alpha-1) / Gamma(alpha)
            # nodes are >0 for GL quadrature
            log_integrand = (alpha - 1.0) * np.log(np.maximum(nodes, 1e-300)) - log_gamma_alpha
            integrand = np.exp(log_integrand)
            # GL: integral = sum w_j * f(u_j) * exp(-u_j) but weights already absorb exp(-u)
            # scipy roots_laguerre: integral_0^inf f(x)*exp(-x)dx ~ sum w_i f(x_i)
            # so we need: sum w_i * kernel(y_q_i) * u_i^(alpha-1) / Gamma(alpha)
            # note: mu/alpha factor from dy = mu/alpha * du is NOT needed here because
            # we've already changed variables — the K_y argument changed, not the measure
            result[i] = float(np.dot(weights, kernel_q * integrand))

        return result

    def dek_dmu(
        self, mu: np.ndarray, y_obs: np.ndarray, h_y: float, **kw
    ) -> np.ndarray:
        """Finite differences for Gamma — quadrature gradient is messy and FD is stable."""
        eps = np.maximum(mu * 1e-4, 1e-6)
        ek_plus = self.ek_y(mu + eps, y_obs, h_y)
        ek_minus = self.ek_y(np.maximum(mu - eps, 1e-12), y_obs, h_y)
        return (ek_plus - ek_minus) / (2.0 * eps)

    def init_params(self, y: np.ndarray, exposure: Optional[np.ndarray]) -> np.ndarray:
        mean_y = np.mean(y)
        return np.full(len(y), max(mean_y, 1e-6))

    def update_alpha(self, y: np.ndarray, mu: np.ndarray) -> None:
        """Method-of-moments estimate of alpha from residuals."""
        # Gamma: Var(Y) = mu^2 / alpha => alpha = mu^2 / Var
        cv2 = np.var(y / np.maximum(mu, 1e-8))
        if cv2 > 1e-8:
            self.alpha = 1.0 / cv2
        self.alpha = np.clip(self.alpha, 0.1, 100.0)
        # Recompute quadrature nodes with same count (nodes don't depend on alpha)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FAMILY_MAP = {
    "gaussian": GaussianFamily,
    "logistic": LogisticFamily,
    "binomial": LogisticFamily,  # alias
    "poisson": PoissonFamily,
    "gamma": GammaFamily,
}


def get_family(name: str, **kwargs) -> Family:
    """Instantiate a family by name."""
    name_lower = name.lower()
    if name_lower not in _FAMILY_MAP:
        raise ValueError(
            f"Unknown family '{name}'. Choose from: {list(_FAMILY_MAP.keys())}"
        )
    return _FAMILY_MAP[name_lower](**kwargs)
