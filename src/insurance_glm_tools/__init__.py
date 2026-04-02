"""insurance-glm-tools: GLM tools for UK insurance pricing.

Three subpackages:

    insurance_glm_tools.nested
        Nested GLM with neural network entity embeddings and spatially
        constrained territory clustering (Wang, Shi, Cao NAAJ 2025).

    insurance_glm_tools.cluster
        Automated GLM factor-level clustering via the R2VF algorithm
        (Ben Dror 2025, arXiv:2503.01521).

    insurance_glm_tools.robust
        MMD-penalised GLM with L1 regularisation for robust insurance pricing.
        Supports Gaussian, Logistic, Poisson, and Gamma families.
        (Kang & Kang 2026, arXiv:2602.21132).

Quick imports::

    from insurance_glm_tools.nested import NestedGLM, EmbeddingNet, NestedGLMPipeline
    from insurance_glm_tools.cluster import FactorClusterer, LevelMap
    from insurance_glm_tools.robust import RobustMMDGLM

Top-level convenience imports::

    from insurance_glm_tools import NestedGLMPipeline, FactorClusterer, RobustMMDGLM
"""

from __future__ import annotations

# Top-level convenience imports — avoids requiring users to remember subpackage paths
# for the most commonly used entry points.
from insurance_glm_tools.nested import NestedGLMPipeline
from insurance_glm_tools.cluster import FactorClusterer, LevelMap
from insurance_glm_tools.robust import RobustMMDGLM

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-glm-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    "nested",
    "cluster",
    "robust",
    "NestedGLMPipeline",
    "FactorClusterer",
    "LevelMap",
    "RobustMMDGLM",
    "__version__",
]
