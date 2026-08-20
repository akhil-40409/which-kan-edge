# The stack: PennyLane + JAX + Catalyst

```text
Optax training loop
        │
        ▼
   JAX (arrays, grad, vmap, jit)     ← classical autodiff / runtime
        │
        ▼
   PennyLane QNode                   ← circuit DSL + measurements
        │
        ▼
   Device (default.qubit / lightning.qubit / …)
        │
        └── optional: Catalyst @qjit compiles the hybrid hot path
```

## What each piece does


| Layer                | Job                                                        | Always on?                                        |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| **PennyLane**        | Write / specify gates; live QNodes for multi-qubit QNN     | QNN (required); QKAN/QuIRK use equivalent JAX 2×2 |
| **JAX**              | Arrays + autodiff into Optax (`interface="jax"` on QNodes) | Yes                                               |
| **Device**           | Where a live QNode’s statevector lives                     | QNN                                               |
| **Catalyst** `@qjit` | Compile hybrid quantum+classical hot path                  | Optional (QNN)                                    |


In eigenflow:

```python
from eigenflow import QNN, QKAN

# Multi-qubit QNN: live PennyLane QNode + JAX
m = QNN(n_features=2, n_qubits=2, device="default.qubit", qjit=False)

# Optional Catalyst compile (needs Lightning-class device + catalyst extra)
m = QNN(n_features=2, n_qubits=2, device="lightning.qubit", qjit=True)

# QKAN/QuIRK: 1-qubit edges use exact JAX 2×2 (PennyLane-equivalent gates)
k = QKAN([2, 4, 1], n_reps=2)
```

Flipping `device=` / `qjit=` on **QNN** does **not** change the circuit math — only where/how it runs.

## When to use eager vs `@qjit`

- **Eager (default):** debugging, teaching, `default.qubit`, most QNN experiments.
- `qjit=True`**:** hot multi-qubit QNN loops on Lightning when Python overhead shows up in profiles.

## Implementation note

1-qubit QKAN/QuIRK **edges** are evaluated with an exact JAX 2×2 statevector
(same gates as the PennyLane circuit). Multi-qubit **QNN** uses a live PennyLane
QNode (`interface="jax"`, optional Catalyst `qjit`). This keeps edge-heavy KANs
trainable while still matching the documented circuits (verified in tests against
PennyLane).

## Sharp bits

1. Catalyst’s fast path wants **Lightning**, not `default.qubit`.
2. Inside `@qjit`, prefer Catalyst transforms (`catalyst.grad`, …) for quantum-aware grads.
3. Install Catalyst via `pip install 'eigenflow[catalyst]'`.

Code: `[eigenflow/backends/qnode.py](../eigenflow/backends/qnode.py)`.