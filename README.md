# Which KAN Edge?

Code and draft for a short arXiv note: **SplineKAN vs FourierKAN vs QKAN** on scientific formula-fitting.

In the regime where KANs are competitive with MLPs under matched `#params` and FLOPs, which univariate edge is best: B-spline, Fourier, or a one-qubit variational \(\langle Z\rangle\) activation?

QKAN edges are exact \(2\times 2\) JAX statevector simulations, not hardware. We do **not** claim quantum advantage.

Paper skeleton: [`paper/main.tex`](paper/main.tex). Protocol: [`docs/paper_claim.md`](docs/paper_claim.md).

Installable package: `src` (`pip install -e .` → `from src import SplineKAN`).

## Install

```bash
git clone https://github.com/akhil-40409/which-kan-edge.git
cd which-kan-edge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 30-second train

```python
import jax
from src import SplineKAN
from src.datasets import make_dataset
from src.training import train_model

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
data = make_dataset("I.15.3t", k1, n_samples=1000)
model = SplineKAN([data["n_features"], 8, 1], grid_size=5)
out = train_model(model, data, k2, steps=500)
print(out["test_rmse"], out["n_params"], out["flops"])
```

Swap `SplineKAN` for `FourierKAN`, `QKAN`, or `MLP`. Also in-tree (not in the paper comparison): `QNN`, `QuIRK`. See [`docs/`](docs/).

## Experiments

```bash
python experiments/run_benchmark.py --quick
# Yu-style arch sweep + noise (envelope plots)
python experiments/run_benchmark.py --suite fair --list-jobs
# Full-task coarse table — run after picking matched configs from the envelopes
python experiments/run_benchmark.py --suite paper --list-jobs
# ASU Sol overnight on Grace Hopper
GPU=gh200 bash jobs/submit_overnight.sh
```

Sol walkthrough: [`docs/sol.md`](docs/sol.md). Merge shards with [`experiments/merge_results.py`](experiments/merge_results.py).

## Layout

```text
src/           # Python package (mlp, spline_kan, fourier_kan, qkan, …)
docs/          # claim, stack, Sol, per-model notes
experiments/   # run_benchmark.py, merge_results.py
jobs/          # Sol CPU / A100 / GH200 arrays
paper/         # arXiv skeleton (title, abstract, headings, bib)
tests/
```

## References

See [`docs/references.md`](docs/references.md).
