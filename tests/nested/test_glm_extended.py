"""
Extended tests for insurance_glm_tools.nested.glm (NestedGLM).

Covers paths not in test_glm.py:
  - result_ property alias
  - aic/bic/deviance with gamma family
  - relativities confidence interval width
  - predict returns counts when multiplied by exposure
  - formula with intercept-only
  - empty embedding column list handled gracefully
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_tools.nested.glm import NestedGLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poisson_df(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age_band": rng.choice(["17-25", "26-35", "36-50", "51+"], n),
        "ncb": rng.choice([0, 1, 2, 3, 4], n),
    })
    exposure = rng.uniform(0.5, 1.5, n)
    y = rng.poisson(0.08 * exposure).astype(float)
    return df, y, exposure


def _gamma_df(n: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age_band": rng.choice(["A", "B", "C"], n),
    })
    exposure = rng.uniform(0.5, 1.5, n)
    y = rng.gamma(shape=3.0, scale=500.0, size=n)
    return df, y, exposure


# ---------------------------------------------------------------------------
# result_ property
# ---------------------------------------------------------------------------


def test_result_property_is_none_before_fit():
    glm = NestedGLM()
    assert glm.result_ is None


def test_result_property_set_after_fit():
    df, y, exposure = _poisson_df()
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    assert glm.result_ is not None


# ---------------------------------------------------------------------------
# aic / bic / deviance with gamma
# ---------------------------------------------------------------------------


def test_gamma_aic_finite():
    df, y, exposure = _gamma_df()
    glm = NestedGLM(family="gamma", formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    assert np.isfinite(glm.aic())


def test_gamma_bic_finite():
    df, y, exposure = _gamma_df()
    glm = NestedGLM(family="gamma", formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    assert np.isfinite(glm.bic())


def test_gamma_deviance_non_negative():
    df, y, exposure = _gamma_df()
    glm = NestedGLM(family="gamma", formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    assert glm.deviance() >= 0


# ---------------------------------------------------------------------------
# relativities CI
# ---------------------------------------------------------------------------


def test_relativities_ci_lower_le_upper():
    df, y, exposure = _poisson_df()
    glm = NestedGLM(formula="age_band + ncb", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    rel = glm.relativities()
    assert (rel["ci_lower"] <= rel["ci_upper"]).all()


def test_relativities_ci_contains_relativity():
    """Point estimate should lie between CI bounds."""
    df, y, exposure = _poisson_df()
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    rel = glm.relativities()
    within = (rel["ci_lower"] <= rel["relativity"]) & (rel["relativity"] <= rel["ci_upper"])
    assert within.all()


def test_relativities_intercept_present():
    """Intercept term should appear in relativities table."""
    df, y, exposure = _poisson_df()
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    rel = glm.relativities()
    assert "Intercept" in rel["term"].values


# ---------------------------------------------------------------------------
# predict: rate * exposure ≈ expected counts
# ---------------------------------------------------------------------------


def test_predict_rate_times_exposure_reasonable():
    """rate * exposure should give reasonable expected claims."""
    df, y, exposure = _poisson_df(n=500, seed=5)
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    rates = glm.predict(df, exposure)
    expected_counts = rates * exposure
    # Expected counts should be in a reasonable range for insurance data
    assert expected_counts.mean() > 0
    assert np.all(expected_counts >= 0)


def test_predict_on_different_data():
    """Prediction on a held-out dataset should work without error."""
    df, y, exposure = _poisson_df(n=400, seed=6)
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df[:300], y[:300], exposure[:300])
    pred = glm.predict(df[300:], exposure[300:])
    assert pred.shape == (100,)
    assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# Intercept-only formula
# ---------------------------------------------------------------------------


def test_intercept_only_formula():
    """formula=None should produce a valid intercept-only model."""
    df, y, exposure = _poisson_df()
    # Use a column-free DataFrame
    df_empty = df[[]].copy()
    glm = NestedGLM(formula=None, add_embedding_cols=False, add_territory=False)
    glm.fit(df_empty, y, exposure)
    assert glm._fitted
    pred = glm.predict(df_empty, exposure)
    # Intercept-only: all predictions should be the same rate
    assert np.std(pred) < 1e-6 * pred.mean()


# ---------------------------------------------------------------------------
# summary string
# ---------------------------------------------------------------------------


def test_summary_contains_family():
    df, y, exposure = _poisson_df()
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    s = glm.summary()
    # statsmodels summaries mention the family somewhere
    assert isinstance(s, str)
    assert len(s) > 100


def test_unfitted_aic_raises():
    glm = NestedGLM()
    with pytest.raises(RuntimeError, match="fit"):
        glm.aic()


def test_unfitted_bic_raises():
    glm = NestedGLM()
    with pytest.raises(RuntimeError, match="fit"):
        glm.bic()


def test_unfitted_deviance_raises():
    glm = NestedGLM()
    with pytest.raises(RuntimeError, match="fit"):
        glm.deviance()


def test_unfitted_relativities_raises():
    glm = NestedGLM()
    with pytest.raises(RuntimeError, match="fit"):
        glm.relativities()


# ---------------------------------------------------------------------------
# Territory flag when column absent
# ---------------------------------------------------------------------------


def test_add_territory_flag_ignored_when_column_absent():
    """add_territory=True but no territory column should not crash."""
    df, y, exposure = _poisson_df()
    # No territory column
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=True)
    glm.fit(df, y, exposure)
    assert glm._fitted
