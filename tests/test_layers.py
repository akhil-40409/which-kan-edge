# Unit tests for JAX-accelerated MLP, KAN, and QKAN layers

import jax
import jax.numpy as jnp
import pytest
from eigenflow.layers import (
    MLP,
    init_mlp_params,
    forward_mlp,
    KAN,
    init_network_params as init_kan_params,
    kan_network,
    QKAN,
    init_qkan_network_params,
    qkan_network,
)


def test_mlp_shapes_and_gradients():
    """Verify shape handling and backpropagation through MLP."""
    key = jax.random.PRNGKey(42)
    layer_sizes = [3, 8, 4, 1]
    mlp = MLP(layer_sizes, squeeze=True)
    
    # Initialize parameters
    params = mlp.init(key)
    assert len(params) == len(layer_sizes) - 1
    for i, (w, b) in enumerate(params):
        assert w.shape == (layer_sizes[i], layer_sizes[i + 1])
        assert b.shape == (layer_sizes[i + 1],)

    # Batched forward pass
    X_batch = jnp.ones((10, 3))
    y_pred_batch = mlp(params, X_batch)
    assert y_pred_batch.shape == (10,)

    # Unbatched forward pass
    X_single = jnp.ones((3,))
    y_pred_single = mlp(params, X_single)
    assert y_pred_single.shape == ()

    # Squeeze = False
    mlp_nosqueeze = MLP(layer_sizes, squeeze=False)
    y_nosqueeze = mlp_nosqueeze(params, X_batch)
    assert y_nosqueeze.shape == (10, 1)

    # Check JIT compilation
    @jax.jit
    def run_mlp(p, x):
        return mlp(p, x)

    y_jit = run_mlp(params, X_batch)
    assert jnp.allclose(y_pred_batch, y_jit)

    # Check backpropagation/gradients
    def loss_fn(p, x):
        return jnp.sum(mlp(p, x) ** 2)

    val, grads = jax.value_and_grad(loss_fn)(params, X_batch)
    assert val > 0.0
    assert len(grads) == len(params)
    for (w_grad, b_grad) in grads:
        assert not jnp.any(jnp.isnan(w_grad))
        assert not jnp.any(jnp.isnan(b_grad))


def test_kan_shapes_and_gradients():
    """Verify shape handling and backpropagation through KAN."""
    key = jax.random.PRNGKey(123)
    layer_sizes = [2, 6, 1]
    kan = KAN(layer_sizes, grid_size=4, spline_order=3)
    
    params, grids = kan.init(key)
    assert len(params) == len(layer_sizes) - 1
    assert len(grids) == len(layer_sizes) - 1

    # Check shapes
    for i in range(len(layer_sizes) - 1):
        w_base, w_spline = params[i]
        grid = grids[i]
        assert w_base.shape == (layer_sizes[i], layer_sizes[i + 1])
        assert w_spline.shape == (layer_sizes[i], layer_sizes[i + 1], 4 + 3)
        assert grid.shape == (layer_sizes[i], 4 + 2 * 3 + 1)

    # Batched forward
    X_batch = jax.random.uniform(key, shape=(5, 2), minval=-0.9, maxval=0.9)
    y_pred = kan(params, grids, X_batch, squeeze=True)
    assert y_pred.shape == (5,)

    # Unbatched forward
    X_single = jnp.array([0.5, -0.5])
    y_pred_single = kan(params, grids, X_single, squeeze=True)
    assert y_pred_single.shape == ()

    # Squeeze = False
    y_nosqueeze = kan(params, grids, X_batch, squeeze=False)
    assert y_nosqueeze.shape == (5, 1)

    # Check JIT
    @jax.jit
    def run_kan(p, g, x):
        return kan(p, g, x)

    y_jit = run_kan(params, grids, X_batch)
    assert jnp.allclose(y_pred, y_jit)

    # Check backpropagation/gradients
    def loss_fn(p, g, x):
        return jnp.sum(kan(p, g, x) ** 2)

    val, grads = jax.value_and_grad(loss_fn)(params, grids, X_batch)
    assert val > 0.0
    assert len(grads) == len(params)
    for (w_base_grad, w_spline_grad) in grads:
        assert not jnp.any(jnp.isnan(w_base_grad))
        assert not jnp.any(jnp.isnan(w_spline_grad))


def test_qkan_shapes_and_gradients():
    """Verify shape handling and backpropagation through QKAN."""
    key = jax.random.PRNGKey(99)
    layer_sizes = [2, 4, 1]
    qkan = QKAN(layer_sizes, num_layers=2)

    params = qkan.init(key)
    assert len(params) == len(layer_sizes) - 1

    # Check parameter shapes
    for i in range(len(layer_sizes) - 1):
        w_base, w_quantum = params[i]
        assert w_base.shape == (layer_sizes[i], layer_sizes[i + 1])
        assert w_quantum.shape == (layer_sizes[i], layer_sizes[i + 1], 2, 3)

    # Batched forward
    X_batch = jnp.ones((3, 2)) * 0.5
    y_pred = qkan(params, X_batch, squeeze=True)
    assert y_pred.shape == (3,)

    # Unbatched forward
    X_single = jnp.array([0.2, -0.2])
    y_pred_single = qkan(params, X_single, squeeze=True)
    assert y_pred_single.shape == ()

    # Squeeze = False
    y_nosqueeze = qkan(params, X_batch, squeeze=False)
    assert y_nosqueeze.shape == (3, 1)

    # Check JIT compilation
    @jax.jit
    def run_qkan(p, x):
        return qkan(p, x)

    y_jit = run_qkan(params, X_batch)
    assert jnp.allclose(y_pred, y_jit)

    # Check backpropagation/gradients
    def loss_fn(p, x):
        return jnp.sum(qkan(p, x) ** 2)

    val, grads = jax.value_and_grad(loss_fn)(params, X_batch)
    assert val > 0.0
    assert len(grads) == len(params)
    for (w_base_grad, w_quantum_grad) in grads:
        assert not jnp.any(jnp.isnan(w_base_grad))
        assert not jnp.any(jnp.isnan(w_quantum_grad))
