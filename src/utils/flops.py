"""Analytic FLOPs-per-sample estimates for which-kan-edge models.

Conventions follow Yu et al., *KAN or MLP: A Fairer Comparison* (arXiv:2407.16674):
arithmetic ops count as 1 FLOP; we report forward pass cost for one sample.
Formulas are documented in ``docs/paper_claim.md``.
"""

from __future__ import annotations

from typing import Any, List, Sequence


# Approx FLOPs for SiLU / elementwise nonlinearities (Yu treats these separately).
_SILU_FLOPS = 4
# Exact 2×2 statevector QVAF step: RY+RZ matvecs + ⟨Z⟩ (~const per rep).
_QVAF_FLOPS_PER_REP = 48


def _layer_pairs(layer_sizes: Sequence[int]) -> List[tuple[int, int]]:
    return [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]


def flops_mlp(layer_sizes: Sequence[int]) -> int:
    """One MLP forward: sum_ℓ (2 d_in d_out + SILU·d_out), no SILU on last layer."""
    total = 0
    pairs = _layer_pairs(layer_sizes)
    for i, (d_in, d_out) in enumerate(pairs):
        total += 2 * d_in * d_out
        if i < len(pairs) - 1:
            total += _SILU_FLOPS * d_out
    return int(total)


def flops_spline_kan(
    layer_sizes: Sequence[int],
    grid_size: int,
    spline_order: int,
) -> int:
    """B-spline KAN layer FLOPs (Yu §4, simplified De Boor + SiLU shortcut)."""
    g, k = int(grid_size), int(spline_order)
    # Per edge De Boor-ish cost from Yu eq. (approx):
    # 9*K*(G + 1.5*K) + 2*G - 2.5*K + 3  (includes merge with shortcut extras)
    per_edge = int(9 * k * (g + 1.5 * k) + 2 * g - 2.5 * k + 3)
    total = 0
    for d_in, d_out in _layer_pairs(layer_sizes):
        total += _SILU_FLOPS * d_in
        total += d_in * d_out * per_edge
    return int(total)


def flops_fourier_kan(layer_sizes: Sequence[int], n_modes: int) -> int:
    """Fourier KAN: SiLU shortcut + cos/sin + weighted sum per edge."""
    m = int(n_modes)
    # Per edge: M cos, M sin (~8 FLOPs each trig approx), 2M mul-add for coeffs.
    # Use a simple arithmetic model: 10*M (trig-ish) + 4*M (ax+by) = 14*M
    per_edge = 14 * m
    total = 0
    for d_in, d_out in _layer_pairs(layer_sizes):
        total += _SILU_FLOPS * d_in
        total += 2 * d_in * d_out  # base matmul
        total += d_in * d_out * per_edge
    return int(total)


def flops_qkan(layer_sizes: Sequence[int], n_reps: int) -> int:
    """QKAN: SiLU shortcut + QVAF circuit FLOPs × edges × reps."""
    r = int(n_reps)
    total = 0
    for d_in, d_out in _layer_pairs(layer_sizes):
        total += _SILU_FLOPS * d_in
        total += 2 * d_in * d_out  # base matmul
        total += d_in * d_out * (_QVAF_FLOPS_PER_REP * r)
    return int(total)


def estimate_flops(model: Any) -> int:
    """Dispatch FLOPs estimate from a constructed model instance."""
    name = type(model).__name__
    sizes = list(model.layer_sizes)
    if name == "MLP":
        return flops_mlp(sizes)
    if name in ("SplineKAN", "KAN"):
        return flops_spline_kan(sizes, model.grid_size, model.spline_order)
    if name == "FourierKAN":
        return flops_fourier_kan(sizes, model.n_modes)
    if name == "QKAN":
        return flops_qkan(sizes, model.n_reps)
    if name == "QuIRK":
        # Same edge cost family as QKAN (1-qubit reuploading), no SiLU residual
        # in QuIRK — still count circuit FLOPs × edges.
        r = int(getattr(model, "n_reps", 2))
        total = 0
        for d_in, d_out in _layer_pairs(sizes):
            total += d_in * d_out * (_QVAF_FLOPS_PER_REP * r)
        return int(total)
    if name == "QNN":
        # Rough: n_layers * n_qubits * gates — mark as 0 if unknown; caller may skip.
        nq = int(getattr(model, "n_qubits", 0))
        nl = int(getattr(model, "n_layers", 0))
        return int(nq * nl * 64)
    raise TypeError(f"No FLOPs estimate for {name}")


def trainable_param_count(model: Any, state: Any) -> int:
    """Count trainable scalars (excludes frozen SplineKAN grids)."""
    from src.utils.metrics import count_params

    if hasattr(model, "count_trainable_params"):
        return int(model.count_trainable_params(state))
    name = type(model).__name__
    if name in ("SplineKAN", "KAN"):
        params, _grids = state
        return count_params(params)
    return count_params(state)
