"""JAX multi-layer perceptron (classical baseline)."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


def init_mlp_params(
    rng: jax.Array,
    layer_sizes: Union[Sequence[int], Tuple[int, ...]],
) -> List[Tuple[jax.Array, jax.Array]]:
    """Xavier/Glorot-uniform weights and zero biases."""
    keys = jax.random.split(rng, len(layer_sizes) - 1)
    params: List[Tuple[jax.Array, jax.Array]] = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        lim = jnp.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(
            keys[i], shape=(in_dim, out_dim), minval=-lim, maxval=lim
        )
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


@partial(jax.jit, static_argnames=("activation_fn", "squeeze"))
def forward_mlp(
    params: List[Tuple[jax.Array, jax.Array]],
    X: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    squeeze: bool = True,
) -> jax.Array:
    """Evaluate an MLP on ``X``."""
    h = X
    for w, b in params[:-1]:
        h = activation_fn(jnp.dot(h, w) + b)
    w_last, b_last = params[-1]
    out = jnp.dot(h, w_last) + b_last
    if squeeze:
        return jnp.squeeze(out, axis=-1) if out.shape[-1] == 1 else jnp.squeeze(out)
    return out


class MLP:
    """Drop-in JAX MLP: ``params = model.init(key); y = model.apply(params, x)``."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.silu,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.activation_fn = activation_fn
        self.squeeze = squeeze

    def init(self, rng: jax.Array) -> List[Tuple[jax.Array, jax.Array]]:
        return init_mlp_params(rng, self.layer_sizes)

    def apply(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array,
        *,
        squeeze: bool | None = None,
    ) -> jax.Array:
        sq = self.squeeze if squeeze is None else squeeze
        return forward_mlp(params, X, activation_fn=self.activation_fn, squeeze=sq)

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)
