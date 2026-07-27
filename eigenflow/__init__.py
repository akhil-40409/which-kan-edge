"""Eigenflow: PennyLane + JAX drop-in layers for classical and quantum ML."""

from eigenflow import datasets, layers, training, utils
from eigenflow.layers import MLP, QKAN, QNN, QuIRK, SplineKAN, make_model

__version__ = "0.2.0"
__all__ = [
    "datasets",
    "layers",
    "training",
    "utils",
    "MLP",
    "SplineKAN",
    "QNN",
    "QKAN",
    "QuIRK",
    "make_model",
    "__version__",
]
