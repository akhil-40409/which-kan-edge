"""Quantum KAN with variational activation functions (Jiang-style QVAF).

Each KAN edge is a 1-qubit data-reuploading unitary + ⟨Z⟩ (SiLU residual).
Edges use an exact 2×2 statevector (same gates as the PennyLane circuit in
``docs/qkan.md``). Multi-qubit QNN keeps live PennyLane; see backends for qjit.
"""

from __future__ import annotations

from functools import partial
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


def _ry(theta: jax.Array) -> jax.Array:
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=theta.dtype)


def _rz(theta: jax.Array) -> jax.Array:
    e0 = jnp.exp(-1j * theta / 2)
    e1 = jnp.exp(1j * theta / 2)
    return jnp.diag(jnp.stack([e0, e1]))


def qvaf_expval(x: jax.Array, weights: jax.Array) -> jax.Array:
    """⟨Z⟩ for RY(w0*x+w1), RZ(w2) re-uploads. weights: (n_reps, 3)."""
    state = jnp.array([1.0 + 0j, 0.0 + 0j], dtype=jnp.complex64)

    def body(i, s):
        w = weights[i]
        theta = w[0] * x + w[1]
        s = _ry(theta).astype(jnp.complex64) @ s
        s = _rz(w[2]).astype(jnp.complex64) @ s
        return s

    state = jax.lax.fori_loop(0, weights.shape[0], body, state)
    return jnp.real(jnp.abs(state[0]) ** 2 - jnp.abs(state[1]) ** 2)


# Vectorize over batch, in-features, out-features
_qvaf_batch = jax.vmap(qvaf_expval, in_axes=(0, None))
_qvaf_in = jax.vmap(_qvaf_batch, in_axes=(1, 0))
_qvaf_out = jax.vmap(_qvaf_in, in_axes=(None, 1))


def init_qkan_params(
    key: jax.Array,
    in_features: int,
    out_features: int,
    num_layers: int,
) -> Tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    w_base = jax.random.uniform(
        k1, shape=(in_features, out_features), minval=-limit, maxval=limit
    )
    w_quantum = (
        jax.random.normal(k2, shape=(in_features, out_features, num_layers, 3)) * 0.1
    )
    return w_base, w_quantum


def init_qkan_network_params(
    key: jax.Array,
    layer_sizes: Union[Sequence[int], Tuple[int, ...]],
    num_layers: int,
) -> List[Tuple[jax.Array, jax.Array]]:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [
        init_qkan_params(keys[i], layer_sizes[i], layer_sizes[i + 1], num_layers)
        for i in range(len(layer_sizes) - 1)
    ]


@partial(jax.jit, static_argnames=())
def qkan_layer(x: jax.Array, params: Tuple[jax.Array, jax.Array]) -> jax.Array:
    w_base, w_quantum = params
    has_batch = x.ndim > 1
    x_batched = x if has_batch else jnp.expand_dims(x, axis=0)
    base_out = jnp.matmul(jax.nn.silu(x_batched), w_base)
    q_out_raw = _qvaf_out(x_batched, w_quantum)  # (out, in, batch)
    q_edges = jnp.transpose(q_out_raw, (2, 1, 0))
    q_nodes = jnp.sum(q_edges, axis=1)
    out = base_out + q_nodes
    if not has_batch:
        out = jnp.squeeze(out, axis=0)
    return out


class QKAN:
    """QVAF-edge KAN. ``params = model.init(key); y = model.apply(params, x)``.

    ``device`` / ``qjit`` are accepted for API parity with QNN; 1-qubit edges
    use an exact JAX statevector (PennyLane-equivalent gates).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        n_reps: int = 2,
        device: str = "default.qubit",
        qjit: bool = False,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.n_reps = n_reps
        self.num_layers = n_reps
        self.device = device
        self.qjit = qjit
        self.squeeze = squeeze
        if qjit:
            # Documented no-op for edge sims; QNN uses real Catalyst.
            pass

    def init(self, rng: jax.Array) -> List[Tuple[jax.Array, jax.Array]]:
        return init_qkan_network_params(rng, self.layer_sizes, self.n_reps)

    def count_trainable_params(self, state: List[Tuple[jax.Array, jax.Array]]) -> int:
        from src.utils.metrics import count_params

        return count_params(state)

    def apply(
        self,
        params: List[Tuple[jax.Array, jax.Array]],
        X: jax.Array,
        *,
        squeeze: bool | None = None,
    ) -> jax.Array:
        h = X
        for layer_params in params:
            h = qkan_layer(h, layer_params)
        sq = self.squeeze if squeeze is None else squeeze
        if sq:
            return jnp.squeeze(h, axis=-1) if h.shape[-1] == 1 else jnp.squeeze(h)
        return h

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)
