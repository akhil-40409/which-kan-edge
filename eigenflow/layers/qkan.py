# JAX-accelerated QKAN (Quantum Kolmogorov-Arnold Network) Layer

import jax
import jax.numpy as jnp
import pennylane as qml
from typing import List, Tuple, Union
from functools import partial

# Initialize the PennyLane device using JAX-compatible backend
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="jax")
def _qvaf_circuit(x: float, weights: jax.Array) -> jax.Array:
    """A 1-qubit variational circuit acting as an activation function on an edge.

    Args:
        x: A scalar input value.
        weights: Parameters of shape (num_layers, 3).

    Returns:
        The expectation value of PauliZ operator.
    """
    num_layers = weights.shape[0]
    for l in range(num_layers):
        # RY gate: encodes input x scaled/shifted
        qml.RY(weights[l, 0] * x + weights[l, 1], wires=0)
        # RZ gate: learnable rotation
        qml.RZ(weights[l, 2], wires=0)
    return qml.expval(qml.PauliZ(0))


# Vectorize _qvaf_circuit over:
# 1. Batch axis of X (axis 0)
_qvaf_batch = jax.vmap(_qvaf_circuit, in_axes=(0, None))

# 2. Input features axis of weights (axis 0) and columns of X (axis 1)
_qvaf_in = jax.vmap(_qvaf_batch, in_axes=(1, 0))

# 3. Output features axis of weights (axis 1)
_qvaf_out = jax.vmap(_qvaf_in, in_axes=(None, 1))


def init_qkan_params(
    key: jax.random.PRNGKey,
    in_features: int,
    out_features: int,
    num_layers: int
) -> Tuple[jax.Array, jax.Array]:
    """Initializes base weights and quantum weights for a single QKAN layer.

    Args:
        key: JAX random key.
        in_features: Number of input dimensions.
        out_features: Number of output dimensions.
        num_layers: Number of quantum circuit layers.

    Returns:
        w_base: Residual linear weights of shape (in_features, out_features)
        w_quantum: Quantum circuit parameters of shape (in_features, out_features, num_layers, 3)
    """
    k1, k2 = jax.random.split(key)

    # Base residual weights
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    w_base = jax.random.uniform(k1, shape=(in_features, out_features), minval=-limit, maxval=limit)

    # Quantum weights initialized to small random values
    w_quantum = jax.random.normal(k2, shape=(in_features, out_features, num_layers, 3)) * 0.1

    return w_base, w_quantum


def init_qkan_network_params(
    key: jax.random.PRNGKey,
    layer_sizes: Union[List[int], Tuple[int, ...]],
    num_layers: int
) -> List[Tuple[jax.Array, jax.Array]]:
    """Initializes parameters for a multi-layer QKAN.

    Args:
        key: JAX random key.
        layer_sizes: List of layer dimensions.
        num_layers: Number of quantum circuit layers.

    Returns:
        A list of parameter tuples (w_base, w_quantum) per layer.
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)
    network_params = []
    for i in range(len(layer_sizes) - 1):
        params = init_qkan_params(keys[i], layer_sizes[i], layer_sizes[i + 1], num_layers)
        network_params.append(params)
    return network_params


@partial(jax.jit, static_argnums=(2,))
def qkan_layer(
    x: jax.Array,
    params: Tuple[jax.Array, jax.Array],
    num_layers: int
) -> jax.Array:
    """Evaluates a single QKAN layer on input x.

    Args:
        x: Input array of shape (..., in_features).
        params: Tuple (w_base, w_quantum).
        num_layers: Number of quantum circuit layers.

    Returns:
        Layer output of shape (..., out_features).
    """
    w_base, w_quantum = params

    has_batch = x.ndim > 1
    if not has_batch:
        x_batched = jnp.expand_dims(x, axis=0)
    else:
        x_batched = x

    # 1. Base residual: SiLU + linear weights
    base_out = jnp.matmul(jax.nn.silu(x_batched), w_base)

    # 2. Quantum activation functions on the edges
    # Output shape from vmap: (out_features, in_features, batch_size)
    q_out_raw = _qvaf_out(x_batched, w_quantum)

    # Transpose to (batch_size, in_features, out_features)
    q_edges = jnp.transpose(q_out_raw, (2, 1, 0))

    # Sum along in_features dimension to get node activations
    q_nodes = jnp.sum(q_edges, axis=1)

    out = base_out + q_nodes

    if not has_batch:
        out = jnp.squeeze(out, axis=0)
    return out


@partial(jax.jit, static_argnums=(2,))
def qkan_network(
    x: jax.Array,
    network_params: List[Tuple[jax.Array, jax.Array]],
    num_layers: int
) -> jax.Array:
    """Evaluates a multi-layer QKAN.

    Args:
        x: Input array of shape (..., in_features).
        network_params: List of layer parameters.
        num_layers: Number of quantum layers.

    Returns:
        Network predictions.
    """
    h = x
    for params in network_params:
        h = qkan_layer(h, params, num_layers)
    return h


class QKAN:
    """A reusable, JAX-accelerated Quantum Kolmogorov-Arnold Network (QKAN) using PennyLane."""

    def __init__(self, layer_sizes: List[int], num_layers: int = 2):
        """Initializes the QKAN definition.

        Args:
            layer_sizes: List of layer dimensions, e.g. [input_dim, hidden_dim_1, ..., output_dim].
            num_layers: Number of quantum data re-uploading layers.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = num_layers

    def init(self, rng: jax.random.PRNGKey) -> List[Tuple[jax.Array, jax.Array]]:
        """Initializes parameters for the QKAN network.

        Args:
            rng: JAX random number generator key.

        Returns:
            A list of parameter tuples (w_base, w_quantum) for each layer.
        """
        return init_qkan_network_params(rng, self.layer_sizes, self.num_layers)

    def apply(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array,
        squeeze: bool = True
    ) -> jax.Array:
        """Applies the QKAN forward pass on the input array X.

        Args:
            params: Parameters list of tuples (w_base, w_quantum).
            X: Input array of shape (batch_size, input_dim).
            squeeze: If True, squeezes the last dimension of the output.

        Returns:
            Predictions array.
        """
        preds = qkan_network(X, params, self.num_layers)
        if squeeze:
            return jnp.squeeze(preds)
        return preds

    def __call__(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array,
        squeeze: bool = True
    ) -> jax.Array:
        """Alias for self.apply."""
        return self.apply(params, X, squeeze)
