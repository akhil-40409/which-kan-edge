# QKAN (variational activation / QVAF edges)

**This is not** Ivashkov-style QSVT/block-encoding QKAN.

Eigenflow QKAN follows the **quantum variational activation** idea: each KAN edge is a **1-qubit data-reuploading circuit** plus a SiLU residual (Jiang et al. / QVAF-style):

\[
\phi(x) = w_b\,\mathrm{SiLU}(x) + \langle Z\rangle\big(U(x;\theta)\rangle
\]

Circuit per edge (repeat `n_reps` times): `RY(w0·x + w1)`, `RZ(w2)`.

Edges are exact 2×2 JAX statevectors (PennyLane-equivalent; matched in tests).

## Code

[`eigenflow/layers/qkan.py`](../eigenflow/layers/qkan.py)

```python
from eigenflow import QKAN
model = QKAN([3, 8, 1], n_reps=2, device="default.qubit", qjit=False)
params = model.init(key)
y = model.apply(params, x)
```

## Cite

- Jiang et al. — *Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks* (2025). [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)
- Contrast (out of scope here): Ivashkov et al. QSVT QKAN [arXiv:2410.04435](https://arxiv.org/abs/2410.04435)
