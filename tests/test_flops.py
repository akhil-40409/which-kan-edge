from src.layers import FourierKAN, MLP, QKAN, SplineKAN
from src.utils.flops import (
    estimate_flops,
    flops_fourier_kan,
    flops_mlp,
    flops_qkan,
    flops_spline_kan,
    trainable_param_count,
)


def test_flops_positive_and_ordered():
    sizes = [2, 4, 1]
    assert flops_mlp(sizes) > 0
    assert flops_spline_kan(sizes, grid_size=5, spline_order=3) > flops_mlp(sizes)
    assert flops_fourier_kan(sizes, n_modes=5) > 0
    assert flops_qkan(sizes, n_reps=2) > 0


def test_estimate_flops_dispatch(key):
    mlp = MLP([2, 8, 1])
    spline = SplineKAN([2, 8, 1], grid_size=5, spline_order=3)
    fourier = FourierKAN([2, 8, 1], n_modes=5)
    qkan = QKAN([2, 4, 1], n_reps=2)
    for m in (mlp, spline, fourier, qkan):
        assert estimate_flops(m) > 0
        st = m.init(key)
        assert trainable_param_count(m, st) > 0


def test_spline_trainable_excludes_grids(key):
    model = SplineKAN([2, 4, 1], grid_size=5, spline_order=3)
    state = model.init(key)
    params, grids = state
    n_train = trainable_param_count(model, state)
    from src.utils.metrics import count_params

    assert n_train == count_params(params)
    assert n_train < count_params(state)
