"""Which KAN Edge: Spline vs Fourier vs one-qubit variational activations."""

from . import datasets, layers, training, utils
from .layers import FourierKAN, MLP, QKAN, QNN, QuIRK, SplineKAN, make_model

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
