"""Drop-in classical and quantum layers."""

from src.layers.fourier_kan import (
    FourierKAN,
    fourier_kan_layer,
    fourier_kan_network,
    init_fourier_kan_params,
    init_fourier_kan_state,
)
from src.layers.mlp import MLP, forward_mlp, init_mlp_params
from src.layers.qkan import QKAN, init_qkan_network_params, init_qkan_params
from src.layers.qnn import QNN
from src.layers.quirk import QuIRK
from src.layers.spline_kan import (
    KAN,
    SplineKAN,
    compute_b_splines,
    init_kan_params,
    init_spline_kan_state,
    kan_layer,
    kan_network,
)


def make_model(name: str, **kwargs):
    """Factory: ``mlp`` | ``spline`` | ``fourier`` | ``qnn`` | ``qkan`` | ``quirk``."""
    key = name.lower()
    if key == "mlp":
        return MLP(**kwargs)
    if key in ("spline", "kan", "spline_kan"):
        return SplineKAN(**kwargs)
    if key in ("fourier", "fourier_kan", "fkan"):
        return FourierKAN(**kwargs)
    if key == "qnn":
        return QNN(**kwargs)
    if key == "qkan":
        return QKAN(**kwargs)
    if key == "quirk":
        return QuIRK(**kwargs)
    raise ValueError(f"Unknown model {name!r}")


__all__ = [
    "MLP",
    "init_mlp_params",
    "forward_mlp",
    "SplineKAN",
    "KAN",
    "compute_b_splines",
    "init_kan_params",
    "init_spline_kan_state",
    "kan_layer",
    "kan_network",
    "FourierKAN",
    "init_fourier_kan_params",
    "init_fourier_kan_state",
    "fourier_kan_layer",
    "fourier_kan_network",
    "QNN",
    "QKAN",
    "init_qkan_params",
    "init_qkan_network_params",
    "QuIRK",
    "make_model",
]
