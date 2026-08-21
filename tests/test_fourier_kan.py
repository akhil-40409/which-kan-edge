import jax
import jax.numpy as jnp

from src.layers import FourierKAN
from src.utils import count_params, estimate_flops


def test_fourier_kan_shapes_and_gradients(key, batch_key):
    model = FourierKAN([2, 4, 1], n_modes=3)
    state = model.init(key)
    x = jax.random.uniform(batch_key, (8, 2), minval=-1.0, maxval=1.0)
    y = model.apply(state, x)
    assert y.shape == (8,)
    assert count_params(state) > 0
    assert estimate_flops(model) > 0

    def loss(st):
        return jnp.mean(model.apply(st, x) ** 2)

    grads = jax.grad(loss)(state)
    flat = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)


def test_fourier_kan_jit(key, batch_key):
    model = FourierKAN([2, 4, 1], n_modes=3)
    state = model.init(key)
    x = jax.random.uniform(batch_key, (8, 2), minval=-1.0, maxval=1.0)
    y = jax.jit(lambda st, xb: model.apply(st, xb))(state, x)
    assert y.shape == (8,)
