"""Data-reuploading quantum neural network (Pérez-Salinas et al.).

Multi-qubit angle-encode + re-upload layers, PennyLane + JAX.
Optional Catalyst ``qjit`` via ``qjit=True`` (Lightning-class device).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pennylane as qml

from src.backends import make_qnode


def _build_qnn_circuit(n_qubits: int, n_layers: int, n_features: int):
    def circuit(x: jax.Array, weights: jax.Array) -> jax.Array:
        # weights: (n_layers, n_qubits, 2) — RZ, RX per qubit per layer
        for layer in range(n_layers):
            for q in range(n_qubits):
                # Re-upload features (cycle if n_features != n_qubits)
                feat = x[q % n_features]
                qml.RY(feat, wires=q)
                qml.RZ(weights[layer, q, 0], wires=q)
                qml.RX(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class QNN:
    """Data-reuploading QNN with a classical linear readout.

    ``params = model.init(key); y = model.apply(params, x)``
    """

    def __init__(
        self,
        n_features: int,
        n_qubits: Optional[int] = None,
        n_layers: int = 2,
        device: str = "default.qubit",
        qjit: bool = False,
        squeeze: bool = True,
    ):
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits is not None else n_features
        self.n_layers = n_layers
        self.device = device
        self.qjit = qjit
        self.squeeze = squeeze

        circuit = _build_qnn_circuit(self.n_qubits, self.n_layers, self.n_features)
        self._qnode = make_qnode(
            circuit,
            wires=self.n_qubits,
            device=device,
            qjit=qjit,
        )
        # vmap over batch: x (B, F), weights shared
        self._batched = jax.vmap(self._qnode, in_axes=(0, None))

    def init(self, rng: jax.Array) -> dict:
        k1, k2 = jax.random.split(rng)
        weights = jax.random.normal(k1, (self.n_layers, self.n_qubits, 2)) * 0.1
        # Classical head: n_qubits -> 1
        limit = jnp.sqrt(6.0 / (self.n_qubits + 1))
        w_out = jax.random.uniform(k2, (self.n_qubits, 1), minval=-limit, maxval=limit)
        b_out = jnp.zeros((1,))
        return {"weights": weights, "w_out": w_out, "b_out": b_out}

    def apply(self, params: dict, X: jax.Array, *, squeeze: bool | None = None) -> jax.Array:
        single = X.ndim == 1
        xb = X if not single else X[None, :]
        # Ensure feature dim
        if xb.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {xb.shape[-1]}"
            )
        expvals = self._batched(xb, params["weights"])  # list or (B, n_qubits)
        expvals = jnp.asarray(expvals)
        if expvals.ndim == 1:
            expvals = expvals[None, :]
        # PennyLane may return (n_qubits, B) when vmapped — normalize to (B, n_qubits)
        if expvals.shape[-1] != self.n_qubits and expvals.shape[0] == self.n_qubits:
            expvals = expvals.T
        out = expvals @ params["w_out"] + params["b_out"]  # (B, 1)
        sq = self.squeeze if squeeze is None else squeeze
        if sq:
            out = jnp.squeeze(out, axis=-1)
        if single:
            out = out[0]
        return out

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)
