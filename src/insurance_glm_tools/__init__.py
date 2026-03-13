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
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["nested", "cluster", "__version__"]
