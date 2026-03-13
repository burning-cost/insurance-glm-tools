"""
insurance_glm_tools.cluster: Automated GLM factor-level clustering for insurance pricing.

Implements the R2VF algorithm (Ben Dror 2025, arXiv:2503.01521) for ordinal
factor clustering via split-coded fused lasso. Designed for UK motor pricing
actuaries who currently do this manually in Excel.

Public API
----------
FactorClusterer : Main class for fitting and transforming.
LevelMap        : Output object mapping original levels to merged groups.
DiagnosticPath  : BIC / deviance path for lambda selection inspection.

Example
-------
>>> from insurance_glm_tools.cluster import FactorClusterer
>>> fc = FactorClusterer(family='poisson', lambda_='bic', min_exposure=500)
>>> fc.fit(X, y, exposure=exposure, ordinal_factors=['vehicle_age', 'ncd_years'])
>>> X_merged = fc.transform(X)
>>> result = fc.refit_glm(X_merged, y, exposure=exposure)
>>> lm = fc.level_map('vehicle_age')
>>> print(lm.to_df())
"""

from .clusterer import FactorClusterer
from .level_map import LevelMap, build_level_map
from .diagnostics import DiagnosticPath
from .penalties import (
    make_split_coded_matrix,
    delta_to_beta,
    beta_to_delta,
    identify_merged_groups,
)
from .constraints import enforce_min_exposure, relabel_groups_contiguous

__all__ = [
    "FactorClusterer",
    "LevelMap",
    "build_level_map",
    "DiagnosticPath",
    "make_split_coded_matrix",
    "delta_to_beta",
    "beta_to_delta",
    "identify_merged_groups",
    "enforce_min_exposure",
    "relabel_groups_contiguous",
]
