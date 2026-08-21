import jax
import jax.numpy as jnp
import pytest

from src.datasets import (
    FEYNMAN_EQUATIONS,
    FeynmanDatasetGenerator,
    make_dataset,
    train_val_test_split,
)


def test_all_equations_generate(key):
    for eq_id in FEYNMAN_EQUATIONS:
        gen = FeynmanDatasetGenerator(eq_id)
        X, y, meta = gen.generate(key, 32, input_scaling="minmax_11")
        assert X.shape[0] == 32
        assert y.shape == (32,)
        assert jnp.all(jnp.isfinite(X))
        assert jnp.all(jnp.isfinite(y))


def test_time_dilation_analytic():
    # t' = t / sqrt(1 - (v/c)^2) with clip
    X = jnp.array([[2.0, 0.6, 2.0]])  # t=2, v=0.6, c=2 → ratio=0.3
    y = FEYNMAN_EQUATIONS["I.15.3t"].func(X)
    expected = 2.0 / jnp.sqrt(1.0 - 0.3**2)
    assert jnp.allclose(y[0], expected, rtol=1e-5)


def test_train_val_test_split_sizes(key):
    X = jax.random.normal(key, (100, 3))
    y = jax.random.normal(key, (100,))
    data = train_val_test_split(X, y, key, train_frac=0.7, val_frac=0.15)
    assert data["x_train"].shape[0] == 70
    assert data["x_val"].shape[0] == 15
    assert data["x_test"].shape[0] == 15


def test_make_dataset_feynman(key):
    data = make_dataset("I.12.1", key, n_samples=200)
    assert data["n_features"] == 2
    assert "x_train" in data


def test_generate_splits(key):
    gen = FeynmanDatasetGenerator("I.12.1")
    data = gen.generate_splits(key, 100)
    assert data["x_train"].shape[0] + data["x_val"].shape[0] + data["x_test"].shape[
        0
    ] == 100
