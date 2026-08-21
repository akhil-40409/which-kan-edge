"""B-spline Kolmogorov–Arnold Network (pure JAX).

Grids are part of the state but are frozen during training by default
(gradients are taken w.r.t. weights only in the shared train loop when
using a custom loss — here the full state is trainable unless you stop-grad
grids yourself).
"""

from __future__ import annotations

from functools import partial
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

# State: (list of (w_base, w_spline), list of grids)
SplineKANState = Tuple[List[Tuple[jax.Array, jax.Array]], List[jax.Array]]


@partial(jax.jit, static_argnums=(2,))
def compute_b_splines(x: jax.Array, grid: jax.Array, spline_order: int) -> jax.Array:
    """Cox–de Boor B-spline bases for ``x`` on ``grid``."""
    has_batch = x.ndim > 1
    x_batched = x if has_batch else jnp.expand_dims(x, axis=0)

    x_expanded = jnp.expand_dims(x_batched, axis=-1)
    grid_expanded = jnp.expand_dims(grid, axis=0)

    bases = (
        (x_expanded >= grid_expanded[:, :, :-1])
        & (x_expanded < grid_expanded[:, :, 1:])
    ).astype(x.dtype)

    for k in range(1, spline_order + 1):
        denom1 = grid_expanded[:, :, k:-1] - grid_expanded[:, :, : -(k + 1)]
        denom1 = jnp.where(denom1 == 0.0, 1.0, denom1)
        denom2 = grid_expanded[:, :, k + 1 :] - grid_expanded[:, :, 1:-k]
        denom2 = jnp.where(denom2 == 0.0, 1.0, denom2)
        term1 = (x_expanded - grid_expanded[:, :, : -(k + 1)]) / denom1 * bases[:, :, :-1]
        term2 = (grid_expanded[:, :, k + 1 :] - x_expanded) / denom2 * bases[:, :, 1:]
        bases = term1 + term2

    if not has_batch:
        bases = jnp.squeeze(bases, axis=0)
    return bases


def init_kan_params(
    key: jax.Array,
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    grid_min: float = -1.0,
    grid_max: float = 1.0,
) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    w_base = jax.random.uniform(
        k1, shape=(in_features, out_features), minval=-limit, maxval=limit
    )
    w_spline = (
        jax.random.normal(k2, shape=(in_features, out_features, grid_size + spline_order))
        * 0.1
    )
    h = (grid_max - grid_min) / grid_size
    grid = jnp.linspace(
        grid_min - spline_order * h,
        grid_max + spline_order * h,
        grid_size + 2 * spline_order + 1,
    )
    grid = jnp.tile(grid, (in_features, 1))
    return (w_base, w_spline), grid


def init_spline_kan_state(
    key: jax.Array,
    layer_sizes: Union[Sequence[int], Tuple[int, ...]],
    grid_size: int,
    spline_order: int,
    grid_min: float = -1.0,
    grid_max: float = 1.0,
) -> SplineKANState:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    network_params = []
    network_grids = []
    for i in range(len(layer_sizes) - 1):
        params, grid = init_kan_params(
            keys[i],
            layer_sizes[i],
            layer_sizes[i + 1],
            grid_size,
            spline_order,
            grid_min,
            grid_max,
        )
        network_params.append(params)
        network_grids.append(grid)
    return network_params, network_grids


@partial(jax.jit, static_argnums=(3,))
def kan_layer(
    x: jax.Array,
    params: Tuple[jax.Array, jax.Array],
    grid: jax.Array,
    spline_order: int,
) -> jax.Array:
    w_base, w_spline = params
    base_out = jnp.matmul(jax.nn.silu(x), w_base)
    bases = compute_b_splines(x, grid, spline_order)
    spline_out = jnp.einsum("...ij,ioj->...o", bases, w_spline)
    return base_out + spline_out


@partial(jax.jit, static_argnums=(2,))
def kan_network(
    x: jax.Array,
    state: SplineKANState,
    spline_order: int,
) -> jax.Array:
    network_params, network_grids = state
    h = x
    for params, grid in zip(network_params, network_grids):
        h = kan_layer(h, params, grid, spline_order)
    return h


class SplineKAN:
    """B-spline KAN. ``state = model.init(key); y = model.apply(state, x)``."""

    def __init__(
        self,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.squeeze = squeeze

    def init(self, rng: jax.Array) -> SplineKANState:
        return init_spline_kan_state(
            rng,
            self.layer_sizes,
            self.grid_size,
            self.spline_order,
            self.grid_min,
            self.grid_max,
        )

    def apply(
        self,
        state: SplineKANState,
        X: jax.Array,
        *,
        squeeze: bool | None = None,
    ) -> jax.Array:
        # Grids are part of state for convenience but frozen (non-trainable).
        params, grids = state
        state = (params, [jax.lax.stop_gradient(g) for g in grids])
        preds = kan_network(X, state, self.spline_order)
        sq = self.squeeze if squeeze is None else squeeze
        if sq:
            return jnp.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else jnp.squeeze(preds)
        return preds

    def __call__(self, state, X, *, squeeze: bool | None = None):
        return self.apply(state, X, squeeze=squeeze)

    def count_trainable_params(self, state: SplineKANState) -> int:
        """Exclude frozen knot grids from the parameter count."""
        from src.utils.metrics import count_params

        params, _grids = state
        return count_params(params)


# Back-compat alias
KAN = SplineKAN
