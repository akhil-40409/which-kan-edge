# JAX-accelerated MLP Layer

import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable, Union
from functools import partial


def init_mlp_params(
    rng: jax.random.PRNGKey,
    layer_sizes: Union[List[int], Tuple[int, ...]]
) -> List[Tuple[jax.Array, jax.Array]]:
    """Initializes Multi-Layer Perceptron parameters with Xavier/Glorot uniform initialization.

    Args:
        rng: JAX random number generator key.
        layer_sizes: List or tuple of integers specifying the sizes of the layers
                     (including inputs and outputs).

    Returns:
        A list of tuples (w, b) for each layer.
    """
    keys = jax.random.split(rng, len(layer_sizes) - 1)
    params = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        lim = jnp.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(keys[i], shape=(in_dim, out_dim), minval=-lim, maxval=lim)
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


@partial(jax.jit, static_argnames=("activation_fn", "squeeze"))
def forward_mlp(
    params: List[Tuple[jax.Array, jax.Array]],
    X: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    squeeze: bool = True
) -> jax.Array:
    """Evaluates the MLP on inputs X.

    Args:
        params: List of parameter tuples (w, b).
        X: Input array of shape (..., in_features).
        activation_fn: The activation function to use on hidden layers.
        squeeze: If True, squeezes the last dimension of the output (useful for 1D targets).

    Returns:
        The evaluated MLP output.
    """
    activation = X
    for w, b in params[:-1]:
        activation = activation_fn(jnp.dot(activation, w) + b)
    w_last, b_last = params[-1]
    out = jnp.dot(activation, w_last) + b_last
    if squeeze:
        return jnp.squeeze(out)
    return out


class MLP:
    """A reusable, JAX-accelerated MLP Layer/Network."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.silu,
        squeeze: bool = True
    ):
        """Initializes the MLP layer definition.

        Args:
            layer_sizes: List of layer dimensions, e.g. [input_dim, hidden_dim_1, ..., output_dim].
            activation_fn: JAX activation function for hidden layers.
            squeeze: Whether to squeeze the output array (e.g. from (batch_size, 1) to (batch_size,)).
        """
        self.layer_sizes = layer_sizes
        self.activation_fn = activation_fn
        self.squeeze = squeeze

    def init(self, rng: jax.random.PRNGKey) -> List[Tuple[jax.Array, jax.Array]]:
        """Initializes parameters for the MLP network.

        Args:
            rng: JAX random number generator key.

        Returns:
            A list of parameter tuples (w, b).
        """
        return init_mlp_params(rng, self.layer_sizes)

    def apply(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array
    ) -> jax.Array:
        """Applies the MLP forward pass on the input array X.

        Args:
            params: Parameters list of tuples (w, b).
            X: Input array of shape (batch_size, input_dim).

        Returns:
            Output array of predictions.
        """
        return forward_mlp(params, X, activation_fn=self.activation_fn, squeeze=self.squeeze)

    def __call__(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array
    ) -> jax.Array:
        """Alias for self.apply."""
        return self.apply(params, X)
