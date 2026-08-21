# Paper claim: fair KAN-edge comparison on scientific targets

**Status:** short arXiv **technical note** (draft 1). Not a conference submission. No CUDA-Q / hardware track.

## Main claim

Under **matched parameter counts** and **matched FLOPs**, compare **SplineKAN vs FourierKAN vs QKAN** on **AI Feynman equations + special functions**, reporting **test RMSE**, **parameter efficiency**, **FLOPs**, and **train time**, with **label noise** \(\in\{0, 0.1\}\). **MLP** is a classical sideline reference, not the primary comparison.

We do **not** claim quantum advantage. QKAN edges are exact 1-qubit JAX statevector simulations (Jiang-style QVAF).

This follows the fairness protocol of Yu et al., [*KAN or MLP: A Fairer Comparison*](https://arxiv.org/abs/2407.16674), applied to the regime where they find KAN strongest (**symbolic / scientific formula representation**), extended across **three edge bases** (B-spline, Fourier, quantum-variational).

## Models

| Role | Model | Edge / activation |
|------|--------|-------------------|
| Main | **SplineKAN** | SiLU residual + B-spline (Liu et al.) |
| Main | **FourierKAN** | SiLU residual + Fourier modes \(\cos(m\pi x),\sin(m\pi x)\) |
| Main | **QKAN** | SiLU residual + 1-qubit data-reuploading \(\langle Z\rangle\) (Jiang et al.) |
| Sideline | **MLP** | Fixed SiLU on nodes |

Code: [`src/layers/`](../src/layers/).

## Experiment suites

### 1. `fair` — architecture sweep (envelope plots)

Yu-style grid over depth / width / basis resolution, on **6 core tasks** (`DEFAULT_TASKS`), **3 seeds**, **noise \(\{0.0, 0.1\}\)**.

| Model | Depth | Width | Basis knobs |
|-------|-------|-------|-------------|
| SplineKAN | 1, 2 | 4, 8, 16 | \(G\in\{3,5,8,16\}\), \(K\in\{2,3\}\) |
| FourierKAN | 1, 2 | 4, 8, 16 | \(M\in\{3,5,8,16\}\) |
| QKAN | 1, 2 | 2, 4, 8 | \(R\in\{1,2,3\}\) |
| MLP | 1, 2 | 16, 32, 64, 128 | — |

Arch keys look like `d1_h8_G5_K3`, `d2_h4_M8`, `d1_h4_R2`, `d1_h64`.

**Plots:** lower envelope of best test RMSE vs `#params` and vs FLOPs (per noise level); train-time vs params on Sol.

```bash
python experiments/run_benchmark.py --suite fair --list-jobs
python experiments/run_benchmark.py --suite fair --platform cpu \
  --shard 0/64 --out results/shards/fair_cpu_0.csv
```

### 2. `paper` — full-task coarse table

All Feynman + specials, budgets `small`/`medium`, noise `{0, 0.1}`, 3 seeds — companion **matched-budget** table after picking configs near equal params/FLOPs from the fair envelopes.

```bash
python experiments/run_benchmark.py --suite paper --list-jobs
```

## Metrics

| Metric | Source |
|--------|--------|
| `test_rmse` / `val_rmse` | Best-by-val checkpoint ([`train_model`](../src/training/__init__.py)) |
| `n_params` | Trainable scalars only (SplineKAN **excludes** frozen grids) |
| `flops` | Analytic per-sample forward ([`src/utils/flops.py`](../src/utils/flops.py)) |
| `train_time_s` | Wall clock on Sol CPU/GPU |

## FLOPs conventions (Yu-style)

- Arithmetic op = 1 FLOP; reported for **one sample**, forward only.
- **MLP:** \(\sum_\ell 2 d_{\mathrm{in}} d_{\mathrm{out}}\) (+ SiLU on hidden).
- **SplineKAN:** SiLU shortcut + De Boor-style per-edge cost (Yu §4).
- **FourierKAN:** SiLU shortcut + base matmul + \(O(M)\) trig/coeff work per edge.
- **QKAN:** SiLU shortcut + base matmul + constant FLOPs per QVAF rep × edges.

Exact constants live in `flops.py` and should be cited in the paper appendix.

## What we are *not* doing in this preprint

- CUDA-Q / CUDA-QX experiments
- Full Yu ML/CV/NLP suite
- Hardware quantum runs
- Claiming symbolic *discovery* (we measure function fit RMSE, not expression recovery)

## Cite

- Liu et al. — KAN (2024). [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Yu et al. — KAN or MLP fair comparison (2024). [arXiv:2407.16674](https://arxiv.org/abs/2407.16674)
- Jiang et al. — QVAF / QKAN (2025). [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)
- Udrescu & Tegmark — AI Feynman (2019). [arXiv:1905.11481](https://arxiv.org/abs/1905.11481)
- Fourier KAN variants — e.g. [arXiv:2406.01034](https://arxiv.org/abs/2406.01034), [arXiv:2409.09323](https://arxiv.org/abs/2409.09323)
