import jax
import jax.numpy as jnp
import pytest

from src.layers import MLP
from src.utils import count_params


def test_mlp_shapes_and_gradients(key, batch_key):
    model = MLP([2, 4, 1])
    params = model.init(key)
    x = jax.random.normal(batch_key, (8, 2))
    y = model.apply(params, x)
    assert y.shape == (8,)
    assert count_params(params) > 0

    def loss(p):
        return jnp.mean(model.apply(p, x) ** 2)

    grads = jax.grad(loss)(params)
    flat = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)


def test_mlp_jit(key, batch_key):
    model = MLP([2, 4, 1])
    params = model.init(key)
    x = jax.random.normal(batch_key, (8, 2))
    fn = jax.jit(lambda p, xb: model.apply(p, xb))
    y = fn(params, x)
    assert y.shape == (8,)


def test_mlp_no_squeeze(key, batch_key):
    model = MLP([2, 4, 1], squeeze=False)
    params = model.init(key)
    x = jax.random.normal(batch_key, (8, 2))
    y = model.apply(params, x)
    assert y.shape == (8, 1)
