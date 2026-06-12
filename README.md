# eigenflow

A clean, minimalist, JAX-accelerated library for classical and quantum machine learning models, specifically tailored for symbolic regression benchmarks. It contains pure JAX implementations of Multi-Layer Perceptrons (MLPs), B-spline Kolmogorov-Arnold Networks (KANs), and Quantum KANs (QKANs). Everything is 100% JIT-compilable, differentiable, and vectorized.

## Directory Structure

* `eigenflow/`: Core library package.
  * `layers/`: Drop-in, compilable model layers.
    * `mlp.py`: Multi-Layer Perceptron using Xavier/Glorot uniform initialization and SiLU activations.
    * `kan.py`: Classical Kolmogorov-Arnold Network using B-splines of arbitrary order (Cox-de Boor recurrence relation) and tensor contractions via `jnp.einsum`.
    * `qkan.py`: Quantum Kolmogorov-Arnold Network using PennyLane. Replaces splines with 1-qubit variational data re-uploading circuits parallelized via nested `jax.vmap`.
  * `datasets/`: Vectorized AI Feynman database generators.
  * `utils/`: Plotting and evaluation helpers.
* `experiments/`: Benchmark notebooks and training scripts.
  * `compare_mlp_kan.ipynb`: Side-by-side training comparison of MLP vs KAN.
* `tests/`: Pytest suite verifying shapes, compilation, and gradients.

## Quick Start

### 1. Run Tests
Verify the installation by running the unit test suite:
```bash
.venv/bin/pytest
```
This runs assertions across all three model architectures to ensure shapes, JIT compilation, and parameter gradients are correct.

### 2. Run Experiments
Train the models on the Relativistic Time Dilation equation ($t' = \frac{t}{\sqrt{1 - v^2/c^2}}$):

```bash
# Train MLP
.venv/bin/python experiments/MLPs/mlp_experiment.py

# Train KAN
.venv/bin/python experiments/KANs/kan_experiment.py
```
Or open and execute `experiments/compare_mlp_kan.ipynb` to compare convergence and JIT overhead side-by-side.

## Verification and Ground Truth

Because we benchmark on physical equations, we establish correctness by directly verifying predicted outputs against the analytical closed-form equations (e.g. $t' = \frac{t}{\sqrt{1 - v^2/c^2}}$). We also assert convergence parity against the original prototype notebooks (~0.002 train loss for MLP, ~0.0009 train loss for KAN).

## References

* **KAN**: Liu et al., *KAN: Kolmogorov-Arnold Networks* (2024). [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
* **AI Feynman**: Udrescu & Tegmark, *AI Feynman: A Physics-Inspired Systematic Symbolic Regression Framework* (2019). [arXiv:1905.11481](https://arxiv.org/abs/1905.11481)
* **QKAN (Taiwan)**: Jiang et al., *Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks* (2025). [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)
* **MLP**: Rumelhart et al., *Learning representations by back-propagating errors* (1986). [Nature](https://www.nature.com/articles/323533a0)

