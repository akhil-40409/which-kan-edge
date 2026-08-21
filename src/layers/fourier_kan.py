"""Fourier Kolmogorov–Arnold Network (pure JAX).

Each edge is a SiLU residual plus a truncated Fourier series (cos/sin modes),
matching the SplineKAN / QKAN residual layout used in this repo.
"""

from __future__ import annotations

from functools import partial
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

# State: list of (w_base, w_fourier) with w_fourier shape (in, out, n_modes, 2)
FourierKANState = List[Tuple[jax.Array, jax.Array]]


def init_fourier_kan_params(
    key: jax.Array,
    in_features: int,
    out_features: int,
    n_modes: int,
) -> Tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    w_base = jax.random.uniform(
        k1, shape=(in_features, out_features), minval=-limit, maxval=limit
    )
    w_fourier = (
        jax.random.normal(k2, shape=(in_features, out_features, n_modes, 2)) * 0.1
    )
    return w_base, w_fourier


def init_fourier_kan_state(
    key: jax.Array,
    layer_sizes: Union[Sequence[int], Tuple[int, ...]],
    n_modes: int,
) -> FourierKANState:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [
        init_fourier_kan_params(keys[i], layer_sizes[i], layer_sizes[i + 1], n_modes)
        for i in range(len(layer_sizes) - 1)
    ]


@partial(jax.jit, static_argnums=(2,))
def fourier_kan_layer(
    x: jax.Array,
    params: Tuple[jax.Array, jax.Array],
    n_modes: int,
) -> jax.Array:
    """SiLU residual + sum_m a_m cos(m π x) + b_m sin(m π x) on each edge."""
    w_base, w_fourier = params
    has_batch = x.ndim > 1
    x_b = x if has_batch else jnp.expand_dims(x, axis=0)

    base_out = jnp.matmul(jax.nn.silu(x_b), w_base)

    # angles: (batch, in, M)
    ms = jnp.arange(1, n_modes + 1, dtype=x_b.dtype)
    angles = x_b[..., None] * (ms * jnp.pi)
    cos_t = jnp.cos(angles)
    sin_t = jnp.sin(angles)

    # w_fourier: (in, out, M, 2) → cos/sin coeffs
    a = w_fourier[..., 0]  # (in, out, M)
    b = w_fourier[..., 1]
    # einsum: batch,in,M × in,out,M → batch,out
    fourier_out = jnp.einsum("bim,iom->bo", cos_t, a) + jnp.einsum(
        "bim,iom->bo", sin_t, b
    )
    out = base_out + fourier_out
    if not has_batch:
        out = jnp.squeeze(out, axis=0)
    return out


@partial(jax.jit, static_argnums=(2,))
def fourier_kan_network(
    x: jax.Array,
    state: FourierKANState,
    n_modes: int,
) -> jax.Array:
    h = x
    for params in state:
        h = fourier_kan_layer(h, params, n_modes)
    return h


class FourierKAN:
    """Fourier-edge KAN. ``params = model.init(key); y = model.apply(params, x)``."""

    def __init__(
        self,
        layer_sizes: List[int],
        n_modes: int = 5,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.n_modes = int(n_modes)
        self.squeeze = squeeze

    def init(self, rng: jax.Array) -> FourierKANState:
        return init_fourier_kan_state(rng, self.layer_sizes, self.n_modes)

    def apply(
        self,
        state: FourierKANState,
        X: jax.Array,
        *,
        squeeze: bool | None = None,
    ) -> jax.Array:
        preds = fourier_kan_network(X, state, self.n_modes)
        sq = self.squeeze if squeeze is None else squeeze
        if sq:
            return jnp.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else jnp.squeeze(preds)
        return preds

    def __call__(self, state, X, *, squeeze: bool | None = None):
        return self.apply(state, X, squeeze=squeeze)

    def count_trainable_params(self, state: FourierKANState) -> int:
        from src.utils.metrics import count_params

        return count_params(state)
