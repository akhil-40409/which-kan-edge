# QuIRK

**QuIRK** (*Quantum-Inspired Re-uploading KAN*, Sharma et al.) replaces B-spline edges with single-qubit data-reuploading activations and adds a **rescale-to-[0, π]** layer between QuIRK blocks.

Paper circuit choices used here:

- Encoding: \(\phi(x) = R_Y(x)\)
- Trainable block: \(U(\theta) = R_Z(\theta_0)\,R_X(\theta_1)\) → **2 params per DR layer**
- Optional dense head maps network output toward \(\mathbb{R}\)

Unlike QKAN in this repo, QuIRK has **no SiLU residual** and uses **rescale** between layers.

“Quantum-inspired”: circuits are factorizable 1-qubit DR units. We simulate them
with exact JAX 2×2 matrices (PennyLane-equivalent gates) for speed.

## Code

[`eigenflow/layers/quirk.py`](../eigenflow/layers/quirk.py)

```python
from eigenflow import QuIRK
model = QuIRK([3, 4, 1], n_reps=2, use_dense_head=True)
params = model.init(key)
y = model.apply(params, x)
```

## Cite

Sharma et al. — *QuIRK: Quantum-Inspired Re-uploading KAN* (2025). [arXiv:2510.08650](https://arxiv.org/abs/2510.08650)
