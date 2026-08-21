# FourierKAN

Each KAN edge is a **SiLU residual** plus a truncated **Fourier series** (same residual layout as SplineKAN / QKAN):

\[
\phi(x) = w_b\,\mathrm{SiLU}(x) + \sum_{m=1}^{M}\big(a_m\cos(m\pi x)+b_m\sin(m\pi x)\big)
\]

Modes \(M\) play the role of spline grid resolution \(G\).

## Code

[`src/layers/fourier_kan.py`](../src/layers/fourier_kan.py)

```python
from src import FourierKAN
model = FourierKAN([3, 16, 1], n_modes=5)
params = model.init(key)
y = model.apply(params, x)
```

## Cite

Fourier-edge KANs appear in several variants (e.g. FourierKAN-GCF, FKAN for INRs). See [`paper_claim.md`](paper_claim.md) for the fair-comparison framing used in this repo.
