# Data-reuploading QNN

Pérez-Salinas et al. showed that repeatedly encoding classical data into a quantum circuit (“re-uploading”) yields a universal approximator.

Eigenflow’s `QNN`:

1. For each layer: encode features with `RY`, apply trainable `RZ`/`RX`, optional CNOTs.
2. Measure ⟨Z⟩ on each qubit.
3. Classical linear head → scalar.

Stack: PennyLane QNode, `interface="jax"`, optional `qjit=True` (see [`stack.md`](stack.md)).

## Code

[`src/layers/qnn.py`](../src/layers/qnn.py)

```python
from src import QNN
model = QNN(n_features=3, n_qubits=3, n_layers=2, device="default.qubit")
params = model.init(key)
y = model.apply(params, x)
```

## Cite

Pérez-Salinas et al. — *Data re-uploading for a universal quantum classifier* (2020). [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)
