"""Drop-in classical and quantum layers for eigenflow."""

from eigenflow.layers.mlp import MLP, forward_mlp, init_mlp_params
from eigenflow.layers.qkan import QKAN, init_qkan_network_params, init_qkan_params
from eigenflow.layers.qnn import QNN
from eigenflow.layers.quirk import QuIRK
from eigenflow.layers.spline_kan import (
    KAN,
    SplineKAN,
    compute_b_splines,
    init_kan_params,
    init_spline_kan_state,
    kan_layer,
    kan_network,
)


def make_model(name: str, **kwargs):
    """Factory: ``mlp`` | ``spline`` | ``kan`` | ``qnn`` | ``qkan`` | ``quirk``."""
    key = name.lower()
    if key == "mlp":
        return MLP(**kwargs)
    if key in ("spline", "kan", "spline_kan"):
        return SplineKAN(**kwargs)
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
    "QNN",
    "QKAN",
    "init_qkan_params",
    "init_qkan_network_params",
    "QuIRK",
    "make_model",
]
