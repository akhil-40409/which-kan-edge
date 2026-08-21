"""QuIRK: Quantum-Inspired Re-uploading KAN (Sharma et al., arXiv:2510.08650).

KAN edges are single-qubit DR circuits with φ(x)=RY(x) and U(θ)=RZ·RX.
Exact 2×2 JAX statevector (PennyLane-equivalent). Rescale to [0, π] between
layers; optional dense head.
"""

from __future__ import annotations

from typing import List

import jax
import jax.numpy as jnp


def _ry(theta: jax.Array) -> jax.Array:
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)


def _rz(theta: jax.Array) -> jax.Array:
    e0 = jnp.exp(-1j * theta / 2)
    e1 = jnp.exp(1j * theta / 2)
    return jnp.diag(jnp.stack([e0, e1])).astype(jnp.complex64)


def _rx(theta: jax.Array) -> jax.Array:
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex64)


def quirk_dr_expval(x: jax.Array, weights: jax.Array) -> jax.Array:
    """⟨Z⟩ for RY(x), RZ, RX re-uploads. weights: (n_reps, 2)."""
    state = jnp.array([1.0 + 0j, 0.0 + 0j], dtype=jnp.complex64)

    def body(i, s):
        w = weights[i]
        s = _ry(x) @ s
        s = _rz(w[0]) @ s
        s = _rx(w[1]) @ s
        return s

    state = jax.lax.fori_loop(0, weights.shape[0], body, state)
    return jnp.real(jnp.abs(state[0]) ** 2 - jnp.abs(state[1]) ** 2)


_dr_batch = jax.vmap(quirk_dr_expval, in_axes=(0, None))
_dr_in = jax.vmap(_dr_batch, in_axes=(1, 0))
_dr_out = jax.vmap(_dr_in, in_axes=(None, 1))


def _rescale_to_pi(h: jax.Array) -> jax.Array:
    h_min = jnp.min(h, axis=-1, keepdims=True)
    h_max = jnp.max(h, axis=-1, keepdims=True)
    return (h - h_min) / (h_max - h_min + 1e-8) * jnp.pi


class QuIRK:
    """QuIRK network. ``params = model.init(key); y = model.apply(params, x)``."""

    def __init__(
        self,
        layer_sizes: List[int],
        n_reps: int = 2,
        device: str = "default.qubit",
        qjit: bool = False,
        use_dense_head: bool = True,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.n_reps = n_reps
        self.device = device
        self.qjit = qjit
        self.use_dense_head = use_dense_head
        self.squeeze = squeeze

    def init(self, rng: jax.Array) -> dict:
        keys = jax.random.split(rng, len(self.layer_sizes))
        edge_weights = []
        for i in range(len(self.layer_sizes) - 1):
            in_f, out_f = self.layer_sizes[i], self.layer_sizes[i + 1]
            w = jax.random.normal(keys[i], (in_f, out_f, self.n_reps, 2)) * 0.1
            edge_weights.append(w)
        params: dict = {"edges": edge_weights}
        if self.use_dense_head:
            out_dim = self.layer_sizes[-1]
            limit = jnp.sqrt(6.0 / (out_dim + 1))
            params["w_head"] = jax.random.uniform(
                keys[-1], (out_dim, 1), minval=-limit, maxval=limit
            )
            params["b_head"] = jnp.zeros((1,))
        return params

    def _quirk_layer(self, x: jax.Array, w_edges: jax.Array) -> jax.Array:
        has_batch = x.ndim > 1
        x_batched = x if has_batch else jnp.expand_dims(x, axis=0)
        raw = _dr_out(x_batched, w_edges)
        edges = jnp.transpose(raw, (2, 1, 0))
        nodes = jnp.sum(edges, axis=1)
        if not has_batch:
            nodes = jnp.squeeze(nodes, axis=0)
        return nodes

    def apply(self, params: dict, X: jax.Array, *, squeeze: bool | None = None) -> jax.Array:
        h = X
        edges = params["edges"]
        for i, w in enumerate(edges):
            h = self._quirk_layer(h, w)
            if i < len(edges) - 1:
                h = _rescale_to_pi(h)

        if self.use_dense_head:
            single = h.ndim == 1
            hb = h if not single else h[None, :]
            out = hb @ params["w_head"] + params["b_head"]
            if single:
                out = out[0]
        else:
            out = h

        sq = self.squeeze if squeeze is None else squeeze
        if sq and out.ndim > 0 and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        return out

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)
