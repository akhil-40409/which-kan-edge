import jax
import jax.numpy as jnp

from src.datasets import make_dataset
from src.layers import MLP
from src.training import train_model


def test_train_mlp_decreases_and_returns_keys(key):
    k1, k2 = jax.random.split(key)
    data = make_dataset("I.12.1", k1, n_samples=128)
    model = MLP([data["n_features"], 8, 1])
    # Evaluate initial loss
    params0 = model.init(k2)
    pred0 = model.apply(params0, data["x_train"])
    loss0 = float(jnp.mean((pred0 - data["y_train"]) ** 2))

    out = train_model(model, data, k2, steps=80, batch_size=32, lr=3e-3)
    assert "test_rmse" in out
    assert "val_rmse" in out
    assert "n_params" in out
    assert "flops" in out
    assert "state" in out
    assert out["n_params"] > 0
    assert out["flops"] > 0
    pred1 = model.apply(out["state"], data["x_train"])
    loss1 = float(jnp.mean((pred1 - data["y_train"]) ** 2))
    assert loss1 < loss0
