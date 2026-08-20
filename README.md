# Eigenflow

Drop-in **PennyLane + JAX** layers for classical and quantum ML, aimed at scientific function approximation.

Models: **MLP**, **B-spline KAN**, **FourierKAN**, **data-reuploading QNN**, **QKAN** (variational-activation edges), **QuIRK**.

We do **not** claim quantum advantage. Quantum models here are differentiable simulators for research and teaching.

**Preprint target (2026-08-18):** fair SplineKAN vs FourierKAN vs QKAN on AI Feynman + specials — see [`docs/paper_claim.md`](docs/paper_claim.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional compiled hybrid path:
pip install -e ".[catalyst]"
```

## 30-second train

```python
import jax
from eigenflow import MLP
from eigenflow.datasets import make_dataset
from eigenflow.training import train_model

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
data = make_dataset("I.15.3t", k1, n_samples=1000)  # relativistic time dilation
model = MLP([data["n_features"], 32, 32, 1])
out = train_model(model, data, k2, steps=500)
print(out["test_rmse"], out["n_params"], out["flops"])
```

Swap `MLP` for `SplineKAN`, `FourierKAN`, `QNN`, `QKAN`, or `QuIRK` (see [`docs/`](docs/)).

## How to read the code

1. [`docs/paper_claim.md`](docs/paper_claim.md) — preprint claim + fair/paper suites.
2. [`docs/stack.md`](docs/stack.md) — PennyLane vs JAX vs Catalyst (the stack layers).
3. [`docs/sol.md`](docs/sol.md) — ASU Sol overnight CPU/GPU sweeps.
4. One paradigm page under `docs/` — math → circuit → code pointer.
5. [`eigenflow/layers/`](eigenflow/layers/) — the actual implementations.
6. [`experiments/run_benchmark.py`](experiments/run_benchmark.py) — one-command suite.

```bash
python experiments/run_benchmark.py --quick
# Yu-style arch sweep + noise (envelope plots):
python experiments/run_benchmark.py --suite fair --list-jobs
# Full-task coarse table + noise:
python experiments/run_benchmark.py --suite paper --list-jobs
# Sol overnight: Grace Hopper (see docs/sol.md)
GPU=gh200 bash jobs/submit_overnight.sh
```

## Layout

```text
eigenflow/
  layers/      # mlp, spline_kan, fourier_kan, qnn, qkan, quirk
  backends/    # device + QNode factory, optional qjit
  datasets/    # AI Feynman + special functions
  training/    # shared Adam loop
docs/          # claim + Karpathy-style notes + references
experiments/   # run_benchmark.py
tests/
```

## References

See [`docs/references.md`](docs/references.md).
