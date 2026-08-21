import jax
import jax.numpy as jnp

from src.layers import QuIRK
from src.utils import count_params


def test_quirk_shapes_and_gradients(key, batch_key):
    model = QuIRK([2, 2, 1], n_reps=1, device="default.qubit")
    params = model.init(key)
    x = jax.random.uniform(batch_key, (4, 2), minval=0.0, maxval=jnp.pi)
    y = model.apply(params, x)
    assert y.shape == (4,)
    assert count_params(params) > 0
    assert "w_head" in params

    def loss(p):
        return jnp.mean(model.apply(p, x) ** 2)

    grads = jax.grad(loss)(params)
    flat = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)


def test_quirk_no_dense_head(key, batch_key):
    model = QuIRK([2, 1], n_reps=1, use_dense_head=False)
    params = model.init(key)
    assert "w_head" not in params
    x = jax.random.uniform(batch_key, (4, 2), minval=0.0, maxval=jnp.pi)
    y = model.apply(params, x)
    assert y.shape == (4,)
