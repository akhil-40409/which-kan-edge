import jax
import jax.numpy as jnp

from src.layers import SplineKAN, compute_b_splines
from src.utils import count_params


def test_spline_kan_shapes_and_gradients(key, batch_key):
    model = SplineKAN([2, 4, 1], grid_size=3, spline_order=2)
    state = model.init(key)
    x = jax.random.uniform(batch_key, (8, 2), minval=-1.0, maxval=1.0)
    y = model.apply(state, x)
    assert y.shape == (8,)
    assert count_params(state) > 0

    def loss(st):
        return jnp.mean(model.apply(st, x) ** 2)

    grads = jax.grad(loss)(state)
    # Grids should have zero grad (stop_gradient)
    _, grid_grads = grads
    for g in grid_grads:
        assert jnp.all(g == 0.0)
    w_grads, _ = grads
    flat = jax.tree_util.tree_leaves(w_grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)


def test_b_spline_bases(key):
    x = jnp.array([[0.0, 0.5]])
    grid = jnp.linspace(-2, 2, 9)
    grid = jnp.stack([grid, grid], axis=0)
    bases = compute_b_splines(x, grid, spline_order=2)
    assert bases.shape[0] == 1
    assert jnp.all(jnp.isfinite(bases))


def test_spline_kan_jit(key, batch_key):
    model = SplineKAN([2, 4, 1], grid_size=3, spline_order=2)
    state = model.init(key)
    x = jax.random.uniform(batch_key, (8, 2), minval=-1.0, maxval=1.0)
    y = jax.jit(lambda st, xb: model.apply(st, xb))(state, x)
    assert y.shape == (8,)
