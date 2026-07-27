"""PennyLane device + QNode helpers (JAX interface, optional Catalyst qjit)."""

from eigenflow.backends.qnode import (
    CATALYST_DEVICES,
    make_device,
    make_qnode,
    maybe_qjit,
    require_catalyst_device,
)

__all__ = [
    "CATALYST_DEVICES",
    "make_device",
    "make_qnode",
    "maybe_qjit",
    "require_catalyst_device",
]
