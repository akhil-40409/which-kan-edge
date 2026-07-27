# B-spline KAN

Kolmogorov–Arnold Networks place **learnable univariate functions on edges**, not fixed activations on nodes (Liu et al., 2024).

Each edge is a SiLU residual plus a B-spline expansion (Cox–de Boor) on a fixed knot grid:

\[
\phi(x) = w_b\,\mathrm{SiLU}(x) + \sum_j w_j B_j(x)
\]

**Grids are frozen** during training (present in state, `stop_gradient` in `apply`).

## Code

[`eigenflow/layers/spline_kan.py`](../eigenflow/layers/spline_kan.py)

```python
from eigenflow import SplineKAN
model = SplineKAN([3, 16, 1], grid_size=5, spline_order=3)
state = model.init(key)   # (weights, grids)
y = model.apply(state, x)
```

## Cite

Liu et al. — *KAN: Kolmogorov-Arnold Networks* (2024). [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
