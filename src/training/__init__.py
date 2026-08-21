"""Shared Adam / MSE training loop."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import optax

from src.utils.flops import estimate_flops, trainable_param_count
from src.utils.metrics import rmse


def train_model(
    model: Any,
    data: dict,
    key: jax.Array,
    *,
    steps: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    log_every: int = 0,
) -> dict:
    """Train with Adam; keep best checkpoint by validation RMSE.

    Expects ``model.init(key)`` / ``model.apply(state, x)`` and a data dict
    with ``x_train,y_train,x_val,y_val,x_test,y_test``.
    """
    state = model.init(key)
    n_params = trainable_param_count(model, state)
    try:
        flops = estimate_flops(model)
    except TypeError:
        flops = -1

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(state)

    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    n = int(x_train.shape[0])
    bs = min(batch_size, n)

    def loss_fn(st, xb, yb):
        pred = model.apply(st, xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(st, opt_st, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(st, xb, yb)
        updates, opt_st = optimizer.update(grads, opt_st, st)
        st = optax.apply_updates(st, updates)
        return st, opt_st, loss

    best_state = state
    best_val = float("inf")
    losses = []

    t0 = time.perf_counter()
    for i in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (bs,), 0, n)
        xb, yb = x_train[idx], y_train[idx]
        state, opt_state, loss = step(state, opt_state, xb, yb)
        losses.append(float(loss))

        if (i + 1) % 50 == 0 or i == steps - 1:
            val = float(rmse(model.apply(state, x_val), y_val))
            if val < best_val:
                best_val = val
                best_state = state
            if log_every and (i + 1) % log_every == 0:
                print(
                    f"  step {i + 1}: train_mse={float(loss):.4e} val_rmse={val:.4e}"
                )

    train_time = time.perf_counter() - t0

    _ = model.apply(best_state, x_test)
    t1 = time.perf_counter()
    for _ in range(5):
        _ = model.apply(best_state, x_test).block_until_ready()
    infer_time = (time.perf_counter() - t1) / 5.0

    test = float(rmse(model.apply(best_state, x_test), y_test))
    val = float(rmse(model.apply(best_state, x_val), y_val))

    return {
        "state": best_state,
        "n_params": n_params,
        "flops": flops,
        "val_rmse": val,
        "test_rmse": test,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "losses": losses,
    }
