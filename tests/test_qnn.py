import jax
import jax.numpy as jnp

from src.layers import QNN
from src.utils import count_params


def test_qnn_shapes_and_gradients(key, batch_key):
    model = QNN(n_features=2, n_qubits=2, n_layers=1, device="default.qubit")
    params = model.init(key)
    x = jax.random.normal(batch_key, (4, 2)) * 0.5
    y = model.apply(params, x)
    assert y.shape == (4,)
    assert count_params(params) > 0

    def loss(p):
        return jnp.mean(model.apply(p, x) ** 2)

    grads = jax.grad(loss)(params)
    flat = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)


def test_qnn_device_wiring(key):
    model = QNN(n_features=2, n_qubits=2, n_layers=1, device="default.qubit")
    assert model.device == "default.qubit"
    params = model.init(key)
    x = jnp.zeros((2, 2))
    y = model.apply(params, x)
    assert y.shape == (2,)
