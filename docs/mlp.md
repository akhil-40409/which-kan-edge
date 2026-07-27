# MLP (classical baseline)

A standard multi-layer perceptron: affine maps + pointwise nonlinearities.

\[
h_{\ell+1} = \sigma(W_\ell h_\ell + b_\ell)
\]

Hidden activations default to **SiLU**. Weights use Xavier/Glorot uniform init.

## Code

[`eigenflow/layers/mlp.py`](../eigenflow/layers/mlp.py)

```python
from eigenflow import MLP
model = MLP([3, 64, 64, 1])
params = model.init(key)
y = model.apply(params, x)
```

## Cite

Rumelhart, Hinton, Williams — *Learning representations by back-propagating errors* (1986).
