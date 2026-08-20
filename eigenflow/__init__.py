"""Eigenflow: PennyLane + JAX drop-in layers for classical and quantum ML."""

from eigenflow import datasets, layers, training, utils
from eigenflow.layers import FourierKAN, MLP, QKAN, QNN, QuIRK, SplineKAN, make_model

__version__ = "0.3.0"
__all__ = [
    "datasets",
    "layers",
    "training",
    "utils",
    "MLP",
    "SplineKAN",
    "FourierKAN",
    "QNN",
    "QKAN",
    "QuIRK",
    "make_model",
    "__version__",
]
