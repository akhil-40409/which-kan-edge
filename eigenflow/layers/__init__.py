# JAX-accelerated layers and models for Eigenflow

from eigenflow.layers.mlp import init_mlp_params, forward_mlp, MLP
from eigenflow.layers.kan import (
    compute_b_splines,
    init_kan_params,
    init_network_params,
    kan_layer,
    kan_network,
    KAN,
)
from eigenflow.layers.qkan import (
    init_qkan_params,
    init_qkan_network_params,
    qkan_layer,
    qkan_network,
    QKAN,
)

__all__ = [
    "init_mlp_params",
    "forward_mlp",
    "MLP",
    "compute_b_splines",
    "init_kan_params",
    "init_network_params",
    "kan_layer",
    "kan_network",
    "KAN",
    "init_qkan_params",
    "init_qkan_network_params",
    "qkan_layer",
    "qkan_network",
    "QKAN",
]
