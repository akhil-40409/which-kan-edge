# JAX-accelerated KAN (Kolmogorov-Arnold Network) Layer

import jax
import jax.numpy as jnp
from typing import List, Tuple, Union
from functools import partial


@partial(jax.jit, static_argnums=(2,))
def compute_b_splines(x: jax.Array, grid: jax.Array, spline_order: int) -> jax.Array:
    """Computes B-spline basis functions for input x over a specified knot grid.

    Args:
        x: Input array of shape (..., in_features).
        grid: Knot grid of shape (in_features, grid_size + 2 * spline_order + 1).
        spline_order: The degree of the spline (e.g., 3 for cubic B-spline).

    Returns:
        B-spline bases of shape (..., in_features, grid_size + spline_order).
    """
    has_batch = x.ndim > 1
    if not has_batch:
        # Add a batch dimension to simplify operations
        x_batched = jnp.expand_dims(x, axis=0)
    else:
        x_batched = x

    # Reshape input to (batch_size, in_features, 1) to broadcast with grid
    x_expanded = jnp.expand_dims(x_batched, axis=-1)
    grid_expanded = jnp.expand_dims(grid, axis=0)  # (1, in_features, knots)

    # Base case: 0-degree basis (piecewise constant indicators)
    bases = ((x_expanded >= grid_expanded[:, :, :-1]) & (x_expanded < grid_expanded[:, :, 1:])).astype(x.dtype)

    # Higher-degree basis via recurrence relation
    for k in range(1, spline_order + 1):
        denom1 = grid_expanded[:, :, k:-1] - grid_expanded[:, :, :-(k + 1)]
        denom1 = jnp.where(denom1 == 0.0, 1.0, denom1)  # Avoid division by zero

        denom2 = grid_expanded[:, :, k + 1:] - grid_expanded[:, :, 1:-k]
        denom2 = jnp.where(denom2 == 0.0, 1.0, denom2)

        term1 = (x_expanded - grid_expanded[:, :, :-(k + 1)]) / denom1 * bases[:, :, :-1]
        term2 = (grid_expanded[:, :, k + 1:] - x_expanded) / denom2 * bases[:, :, 1:]
        bases = term1 + term2

    if not has_batch:
        # Squeeze batch dimension back
        bases = jnp.squeeze(bases, axis=0)
    return bases


def init_kan_params(
    key: jax.random.PRNGKey,
    in_features: int,
    out_features: int,
    grid_size: int,
    spline_order: int,
    grid_min: float = -1.0,
    grid_max: float = 1.0
) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
    """Initializes weights and spline grids for a single KAN layer.

    Args:
        key: JAX random key.
        in_features: Number of input dimensions.
        out_features: Number of output dimensions.
        grid_size: Number of intervals in the grid.
        spline_order: Order of the B-splines.
        grid_min: Minimum grid value.
        grid_max: Maximum grid value.

    Returns:
        params: Tuple of (w_base, w_spline)
        grid: Knot grid of shape (in_features, grid_size + 2 * spline_order + 1)
    """
    k1, k2 = jax.random.split(key)

    # Base weights: shape (in_features, out_features) - residual connection
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    w_base = jax.random.uniform(k1, shape=(in_features, out_features), minval=-limit, maxval=limit)

    # Spline weights: shape (in_features, out_features, grid_size + spline_order)
    w_spline = jax.random.normal(k2, shape=(in_features, out_features, grid_size + spline_order)) * 0.1

    # Grid: shape (in_features, grid_size + 2 * spline_order + 1)
    h = (grid_max - grid_min) / grid_size
    grid = jnp.linspace(
        grid_min - spline_order * h,
        grid_max + spline_order * h,
        grid_size + 2 * spline_order + 1
    )
    grid = jnp.tile(grid, (in_features, 1))

    return (w_base, w_spline), grid


def init_network_params(
    key: jax.random.PRNGKey,
    layer_sizes: Union[List[int], Tuple[int, ...]],
    grid_size: int,
    spline_order: int,
    grid_min: float = -1.0,
    grid_max: float = 1.0
) -> Tuple[List[Tuple[jax.Array, jax.Array]], List[jax.Array]]:
    """Initializes parameters for a multi-layer KAN.

    Args:
        key: JAX random key.
        layer_sizes: List of layer sizes (including inputs and outputs).
        grid_size: Number of intervals in spline grids.
        spline_order: B-spline order.
        grid_min: Minimum grid value.
        grid_max: Maximum grid value.

    Returns:
        network_params: List of (w_base, w_spline) tuples for each layer.
        network_grids: List of grid arrays for each layer.
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)
    network_params = []
    network_grids = []
    for i in range(len(layer_sizes) - 1):
        in_feat = layer_sizes[i]
        out_feat = layer_sizes[i + 1]
        params, grid = init_kan_params(
            keys[i], in_feat, out_feat, grid_size, spline_order, grid_min, grid_max
        )
        network_params.append(params)
        network_grids.append(grid)
    return network_params, network_grids


@partial(jax.jit, static_argnums=(3,))
def kan_layer(
    x: jax.Array,
    params: Tuple[jax.Array, jax.Array],
    grid: jax.Array,
    spline_order: int
) -> jax.Array:
    """Evaluates a single KAN layer on input x.

    Args:
        x: Input array of shape (..., in_features).
        params: Tuple (w_base, w_spline).
        grid: Knot grid array.
        spline_order: Order of the spline.

    Returns:
        Layer output of shape (..., out_features).
    """
    w_base, w_spline = params
    # 1. Base residual connection: SiLU + linear projection
    base_out = jnp.matmul(jax.nn.silu(x), w_base)

    # 2. Spline evaluation
    # Compute basis values (shape: (..., in_features, grid_size + spline_order))
    bases = compute_b_splines(x, grid, spline_order)

    # Contract B-splines with spline weights
    # w_spline shape: (in_features, out_features, grid_size + spline_order)
    # Resulting shape: (..., out_features)
    spline_out = jnp.einsum('...ij,ioj->...o', bases, w_spline)

    return base_out + spline_out


@partial(jax.jit, static_argnums=(3,))
def kan_network(
    x: jax.Array,
    network_params: List[Tuple[jax.Array, jax.Array]],
    network_grids: List[jax.Array],
    spline_order: int
) -> jax.Array:
    """Evaluates a multi-layer KAN.

    Args:
        x: Input array of shape (..., in_features).
        network_params: List of layer weights.
        network_grids: List of layer grids.
        spline_order: Spline order.

    Returns:
        Network predictions.
    """
    h = x
    for params, grid in zip(network_params, network_grids):
        h = kan_layer(h, params, grid, spline_order)
    return h


class KAN:
    """A reusable, JAX-accelerated Kolmogorov-Arnold Network (KAN)."""

    def __init__(
        self,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_min: float = -1.0,
        grid_max: float = 1.0
    ):
        """Initializes the KAN definition.

        Args:
            layer_sizes: List of layer dimensions, e.g. [input_dim, hidden_dim_1, ..., output_dim].
            grid_size: Number of grid intervals for splines.
            spline_order: Order of B-splines (e.g. 3 for cubic splines).
            grid_min: Minimum value of grid boundary.
            grid_max: Maximum value of grid boundary.
        """
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_min = grid_min
        self.grid_max = grid_max

    def init(self, rng: jax.random.PRNGKey) -> Tuple[List[Tuple[jax.Array, jax.Array]], List[jax.Array]]:
        """Initializes network parameters (weights) and grids.

        Args:
            rng: JAX random number generator key.

        Returns:
            network_params: List of weight tuples (w_base, w_spline) per layer.
            network_grids: List of knot grids per layer.
        """
        return init_network_params(
            rng,
            self.layer_sizes,
            self.grid_size,
            self.spline_order,
            self.grid_min,
            self.grid_max
        )

    def apply(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        grids: List[jax.Array],
        X: jax.Array,
        squeeze: bool = True
    ) -> jax.Array:
        """Applies the KAN forward pass on the input array X.

        Args:
            params: List of layer parameter tuples (w_base, w_spline).
            grids: List of layer grids.
            X: Input array of shape (batch_size, input_dim).
            squeeze: If True, squeezes the last dimension of the output (e.g., for 1D outputs).

        Returns:
            Predictions array.
        """
        preds = kan_network(X, params, grids, self.spline_order)
        if squeeze:
            return jnp.squeeze(preds)
        return preds

    def __call__(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        grids: List[jax.Array],
        X: jax.Array,
        squeeze: bool = True
    ) -> jax.Array:
        """Alias for self.apply."""
        return self.apply(params, grids, X, squeeze)
