"""insurance-glm-tools: GLM tools for UK insurance pricing.

Two subpackages:

    insurance_glm_tools.nested
        Nested GLM with neural network entity embeddings and spatially
        constrained territory clustering (Wang, Shi, Cao NAAJ 2025).

    insurance_glm_tools.cluster
        Automated GLM factor-level clustering via the R2VF algorithm
        (Ben Dror 2025, arXiv:2503.01521).

Quick imports::

    from insurance_glm_tools.nested import NestedGLM, EmbeddingNet, NestedGLMPipeline
    from insurance_glm_tools.cluster import FactorClusterer, LevelMap

Top-level convenience imports::

    from insurance_glm_tools import NestedGLMPipeline, FactorClusterer
"""

from __future__ import annotations

# Top-level convenience imports — avoids requiring users to remember subpackage paths
# for the two most commonly used entry points.
from insurance_glm_tools.nested import NestedGLMPipeline
from insurance_glm_tools.cluster import FactorClusterer, LevelMap

__version__ = "0.1.5"

__all__ = [
    "nested",
    "cluster",
    "NestedGLMPipeline",
    "FactorClusterer",
    "LevelMap",
    "__version__",
]
