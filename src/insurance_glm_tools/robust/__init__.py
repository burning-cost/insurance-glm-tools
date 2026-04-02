"""insurance_glm_tools.robust — MMD-penalised GLMs with L1 regularisation.

Implements the RobustMMDGLM estimator from Kang & Kang (2026), arXiv:2602.21132.

Four families are supported:
- Gaussian  (identity link)
- Logistic  (logit link, Bernoulli response)
- Poisson   (log link, count response with optional exposure)
- Gamma     (log link, positive continuous response with optional exposure)

Quick start::

    from insurance_glm_tools.robust import RobustMMDGLM

    # Gamma severity model
    model = RobustMMDGLM(family="gamma", lambda_="cv")
    model.fit(X, y, exposure=vehicle_years)
    print(model.relativities())

    # Poisson frequency model
    model = RobustMMDGLM(family="poisson", lambda_=0.01)
    model.fit(X, claim_counts, exposure=vehicle_years)
    print(model.selected_features())
"""

from insurance_glm_tools.robust._glm import RobustMMDGLM

__all__ = ["RobustMMDGLM"]
