"""
Extended tests for insurance_glm_tools.nested.embedding.

Covers gaps not addressed by test_embedding.py:
  - EmbeddingNet with dropout
  - EmbeddingNet multiple forward passes are deterministic given same inputs
  - EmbeddingTrainer get_embedding_frame column naming convention
  - EmbeddingTrainer with custom embedding_dims
  - EmbeddingTrainer fit with all-zero y
  - _poisson_deviance_loss properties
  - _default_embedding_dim boundary values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from insurance_glm_tools.nested.embedding import (
    EmbeddingNet,
    EmbeddingTrainer,
    _default_embedding_dim,
    _poisson_deviance_loss,
)


# ---------------------------------------------------------------------------
# _default_embedding_dim — boundary values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_levels,expected", [
    (1, 1),     # ceil(1/2) = 1
    (2, 1),     # ceil(2/2) = 1
    (3, 2),     # ceil(3/2) = 2
    (100, 50),  # capped at 50
    (99, 50),   # ceil(99/2) = 50
    (101, 50),  # > 100 → capped at 50
])
def test_default_embedding_dim_values(n_levels, expected):
    assert _default_embedding_dim(n_levels) == expected


# ---------------------------------------------------------------------------
# EmbeddingNet — dropout
# ---------------------------------------------------------------------------


def test_embedding_net_with_dropout_construction():
    net = EmbeddingNet(
        vocab_sizes={"make": 10},
        embedding_dims={"make": 4},
        hidden_sizes=(16, 8),
        dropout=0.3,
    )
    assert net is not None


def test_embedding_net_dropout_forward_shape():
    torch.manual_seed(0)
    net = EmbeddingNet(
        vocab_sizes={"make": 8},
        embedding_dims={"make": 3},
        hidden_sizes=(16,),
        dropout=0.1,
    )
    net.eval()  # Disable dropout for deterministic test
    cat_inputs = {"make": torch.randint(0, 8, (20,))}
    out = net(cat_inputs)
    assert out.shape == (20,)


def test_embedding_net_eval_mode_deterministic():
    """In eval mode, two forward passes should produce identical output."""
    torch.manual_seed(1)
    net = EmbeddingNet(
        vocab_sizes={"make": 5},
        embedding_dims={"make": 2},
        hidden_sizes=(8,),
        dropout=0.5,
    )
    net.eval()
    cat_inputs = {"make": torch.tensor([0, 1, 2, 3, 4])}
    out1 = net(cat_inputs)
    out2 = net(cat_inputs)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# EmbeddingNet — get_embedding_weights shape
# ---------------------------------------------------------------------------


def test_embedding_weights_shape_large_vocab():
    net = EmbeddingNet(
        vocab_sizes={"model": 100},
        embedding_dims={"model": 10},
        hidden_sizes=(32,),
    )
    weights = net.get_embedding_weights()
    assert weights["model"].shape == (100, 10)


def test_embedding_weights_two_columns():
    net = EmbeddingNet(
        vocab_sizes={"make": 8, "colour": 12},
        embedding_dims={"make": 3, "colour": 5},
        hidden_sizes=(16,),
    )
    weights = net.get_embedding_weights()
    assert weights["make"].shape == (8, 3)
    assert weights["colour"].shape == (12, 5)


# ---------------------------------------------------------------------------
# _poisson_deviance_loss — properties
# ---------------------------------------------------------------------------


def test_poisson_deviance_loss_non_negative():
    """Deviance should always be non-negative."""
    rng = np.random.default_rng(0)
    log_pred = torch.tensor(rng.normal(size=20), dtype=torch.float32)
    y = torch.tensor(rng.poisson(1.0, 20), dtype=torch.float32)
    exp = torch.ones(20)
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert loss.item() >= 0


def test_poisson_deviance_loss_perfect_fit_near_zero():
    """When log_pred gives mu == y (perfect fit), loss should be ~0."""
    y = torch.tensor([2.0, 3.0, 1.0, 4.0])
    exp = torch.ones(4)
    log_pred = torch.log(y.clamp(min=1e-10))
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_poisson_deviance_loss_with_exposure():
    """With varying exposure, loss should still be finite."""
    rng = np.random.default_rng(5)
    log_pred = torch.zeros(10)
    y = torch.tensor(rng.poisson(0.5, 10), dtype=torch.float32)
    exp = torch.tensor(rng.uniform(0.5, 2.0, 10), dtype=torch.float32)
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("n", [1, 5, 100])
def test_poisson_deviance_loss_batch_sizes(n):
    log_pred = torch.zeros(n)
    y = torch.ones(n)
    exp = torch.ones(n)
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# EmbeddingTrainer — custom embedding dims
# ---------------------------------------------------------------------------


def _make_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    makes = ["Ford", "Vauxhall", "Toyota", "BMW", "Audi"]
    df = pd.DataFrame({"make": rng.choice(makes, n)})
    exposure = rng.uniform(0.5, 1.5, n).astype(np.float32)
    y = rng.poisson(0.05 * exposure).astype(np.float32)
    return df, y, exposure


def test_embedding_trainer_custom_dims():
    df, y, exposure = _make_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        embedding_dims={"make": 7},
        hidden_sizes=(8,),
        epochs=2,
        random_state=0,
    )
    trainer.fit(df, y, exposure=exposure)
    emb = trainer.transform(df)
    assert emb.shape == (len(df), 7)


def test_embedding_trainer_embedding_frame_column_names():
    """Embedding frame columns should be 'category' + 'emb_0', 'emb_1', ..."""
    df, y, exposure = _make_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        embedding_dims={"make": 3},
        hidden_sizes=(8,),
        epochs=2,
        random_state=1,
    )
    trainer.fit(df, y, exposure=exposure)
    frames = trainer.get_embedding_frame()
    frame = frames["make"]
    assert "category" in frame.columns
    assert "emb_0" in frame.columns
    assert "emb_1" in frame.columns
    assert "emb_2" in frame.columns


def test_embedding_trainer_transform_shape():
    df, y, exposure = _make_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        embedding_dims={"make": 4},
        hidden_sizes=(8,),
        epochs=2,
        random_state=2,
    )
    trainer.fit(df, y, exposure=exposure)
    emb = trainer.transform(df)
    assert emb.shape == (len(df), 4)


def test_embedding_trainer_all_finite_embeddings():
    df, y, exposure = _make_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(8,),
        epochs=3,
        random_state=3,
    )
    trainer.fit(df, y, exposure=exposure)
    emb = trainer.transform(df)
    assert np.all(np.isfinite(emb))


def test_embedding_trainer_fit_with_zero_y():
    """Fitting on all-zero y should not crash (sparse claims data scenario)."""
    df = pd.DataFrame({"make": ["Ford"] * 50 + ["Vauxhall"] * 50})
    y = np.zeros(100, dtype=np.float32)
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(4,),
        epochs=2,
        random_state=4,
    )
    trainer.fit(df, y)
    emb = trainer.transform(df)
    assert emb.shape[0] == 100
    assert np.all(np.isfinite(emb))


def test_embedding_trainer_get_embedding_frame_n_rows():
    """Frame should have one row per unique category value."""
    df, y, exposure = _make_data(n=300)
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(8,),
        epochs=2,
        random_state=5,
    )
    trainer.fit(df, y, exposure=exposure)
    frames = trainer.get_embedding_frame()
    n_unique_makes = df["make"].nunique()
    assert len(frames["make"]) == n_unique_makes
