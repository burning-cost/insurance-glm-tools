"""
Synthetic insurance datasets with known factor structure for testing.

Poisson frequency dataset
--------------------------
10,000 policies, two factors:
  vehicle_age (0-15): true grouping [0-2, 3-7, 8-15]
    group 0 (ages 0-2):  log-rate = -2.0
    group 1 (ages 3-7):  log-rate = -1.7
    group 2 (ages 8-15): log-rate = -1.3

  ncd_years (0-9): true grouping [0-1, 2-4, 5-9]
    group 0 (0-1):  log-rate = 0.0   (reference)
    group 1 (2-4):  log-rate = -0.3
    group 2 (5-9):  log-rate = -0.6

Gamma severity dataset
-----------------------
5,000 claims, vehicle_age same grouping, different coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# True group structure for tests
VEHICLE_AGE_GROUPS = {
    0: 0, 1: 0, 2: 0,               # group 0
    3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # group 1
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,  # group 2
}
VEHICLE_AGE_LOG_RATES = [-2.0, -1.7, -1.3]

NCD_GROUPS = {
    0: 0, 1: 0,       # group 0
    2: 1, 3: 1, 4: 1, # group 1
    5: 2, 6: 2, 7: 2, 8: 2, 9: 2,  # group 2
}
NCD_LOG_RATES = [0.0, -0.3, -0.6]


@pytest.fixture(scope="session")
def poisson_dataset():
    """
    10,000-policy synthetic Poisson frequency dataset.

    Returns dict with keys: X, y, exposure, true_va_groups, true_ncd_groups.
    """
    rng = np.random.default_rng(42)
    n = 10_000

    vehicle_age = rng.integers(0, 16, n)   # 0-15 inclusive
    ncd_years = rng.integers(0, 10, n)     # 0-9 inclusive
    exposure = rng.uniform(0.1, 1.0, n)    # years at risk

    # True log-rates
    va_group = np.array([VEHICLE_AGE_GROUPS[v] for v in vehicle_age])
    ncd_group = np.array([NCD_GROUPS[v] for v in ncd_years])

    log_rate = (
        np.array(VEHICLE_AGE_LOG_RATES)[va_group]
        + np.array(NCD_LOG_RATES)[ncd_group]
    )
    mu = exposure * np.exp(log_rate)
    y = rng.poisson(mu)

    X = pd.DataFrame({
        "vehicle_age": vehicle_age,
        "ncd_years": ncd_years,
    })

    return {
        "X": X,
        "y": y.astype(np.float64),
        "exposure": exposure,
        "true_va_groups": VEHICLE_AGE_GROUPS,
        "true_ncd_groups": NCD_GROUPS,
    }


@pytest.fixture(scope="session")
def gamma_dataset():
    """
    5,000-claim synthetic Gamma severity dataset.

    Returns dict with keys: X, y, weights (claim counts).
    """
    rng = np.random.default_rng(123)
    n = 5_000

    # Gamma severity: different coefficients to Poisson
    SEVERITY_VA_LOG = [-6.0, -5.7, -5.4]   # log average cost
    SEVERITY_VA_GROUPS = VEHICLE_AGE_GROUPS  # same grouping, different values

    vehicle_age = rng.integers(0, 16, n)
    claim_counts = rng.integers(1, 20, n).astype(np.float64)  # weights

    va_group = np.array([SEVERITY_VA_GROUPS[v] for v in vehicle_age])
    log_mu = np.array(SEVERITY_VA_LOG)[va_group]
    mu = np.exp(log_mu)

    # Gamma draw with shape=5 (moderate dispersion)
    y = rng.gamma(shape=5, scale=mu / 5, size=n)

    X = pd.DataFrame({"vehicle_age": vehicle_age})

    return {
        "X": X,
        "y": y,
        "weights": claim_counts,
        "true_va_groups": SEVERITY_VA_GROUPS,
    }


@pytest.fixture(scope="session")
def small_poisson_dataset():
    """
    Small (1,000 policy) Poisson dataset for fast unit tests.
    vehicle_age only, true grouping: [0-4, 5-9].
    """
    rng = np.random.default_rng(7)
    n = 1_000

    vehicle_age = rng.integers(0, 10, n)   # 0-9
    exposure = rng.uniform(0.5, 1.0, n)

    # Two-group structure: ages 0-4 vs 5-9
    true_groups = {v: (0 if v < 5 else 1) for v in range(10)}
    log_rates = [-2.0, -1.5]

    va_group = np.array([true_groups[v] for v in vehicle_age])
    mu = exposure * np.exp(np.array(log_rates)[va_group])
    y = rng.poisson(mu).astype(np.float64)

    X = pd.DataFrame({"vehicle_age": vehicle_age})

    return {
        "X": X,
        "y": y,
        "exposure": exposure,
        "true_groups": true_groups,
    }
