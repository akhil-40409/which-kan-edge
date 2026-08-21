# QKAN (variational activation / QVAF edges)

Eigenflow QKAN follows the **quantum variational activation** idea: each KAN edge is a **1-qubit data-reuploading circuit** plus a SiLU residual (Jiang et al. / QVAF-style):


\phi(x) = w_b\mathrm{SiLU}(x) + \langle Z\rangle\big(U(x;\theta)\rangle


Circuit per edge (repeat `n_reps` times): `RY(w0·x + w1)`, `RZ(w2)`.

Edges are exact 2×2 JAX statevectors (PennyLane-equivalent; matched in tests).

## Code

`[src/layers/qkan.py](../src/layers/qkan.py)`

```python
from src import QKAN
model = QKAN([3, 8, 1], n_reps=2, device="default.qubit", qjit=False)
params = model.init(key)
y = model.apply(params, x)
```



## Cite

- Jiang et al. — *Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks* (2025). [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)

