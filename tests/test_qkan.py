import jax
import jax.numpy as jnp

from eigenflow.layers import QKAN
from eigenflow.utils import count_params


def test_qvaf_matches_pennylane():
    """Edge ⟨Z⟩ agrees with PennyLane for the documented QVAF circuit."""
    import pennylane as qml
    from eigenflow.layers.qkan import qvaf_expval

    weights = jnp.array([[0.2, 0.1, -0.3], [0.05, -0.2, 0.4]], dtype=jnp.float32)
    x = jnp.float32(0.7)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="jax")
    def circuit(xv, w):
        for layer in range(w.shape[0]):
            qml.RY(w[layer, 0] * xv + w[layer, 1], wires=0)
            qml.RZ(w[layer, 2], wires=0)
        return qml.expval(qml.PauliZ(0))

    assert jnp.allclose(qvaf_expval(x, weights), circuit(x, weights), atol=1e-5)


def test_qkan_shapes_and_gradients(key, batch_key):
    model = QKAN([2, 2, 1], n_reps=1, device="default.qubit")
    params = model.init(key)
    x = jax.random.normal(batch_key, (4, 2)) * 0.3
    y = model.apply(params, x)
    assert y.shape == (4,)
    assert count_params(params) > 0

    def loss(p):
        return jnp.mean(model.apply(p, x) ** 2)

    grads = jax.grad(loss)(params)
    flat = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)
