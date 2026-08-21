"""Device and QNode factory for PennyLane + JAX (+ optional Catalyst).

JAX is always the autodiff interface. Catalyst ``qjit`` is an optional
compile step that requires a Lightning-class (or other Catalyst-supported)
device — see ``docs/stack.md``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import pennylane as qml

# Devices known to work with Catalyst (see Catalyst docs). Not exhaustive.
CATALYST_DEVICES = frozenset(
    {
        "lightning.qubit",
        "lightning.kokkos",
        "lightning.gpu",
        "lightning.amdgpu",
        "null.qubit",
    }
)


def make_device(name: str = "default.qubit", wires: int = 1, **kwargs: Any):
    """Create a PennyLane device."""
    return qml.device(name, wires=wires, **kwargs)


def require_catalyst_device(device_name: str) -> None:
    """Raise if ``device_name`` is not suitable for Catalyst qjit."""
    # Strip options like "lightning.qubit" only — name is the short id.
    base = device_name.split(",")[0].strip()
    if base not in CATALYST_DEVICES and not base.startswith("lightning."):
        raise ValueError(
            f"qjit=True requires a Catalyst-supported device "
            f"(e.g. lightning.qubit), got {device_name!r}. "
            f"See docs/stack.md."
        )


def maybe_qjit(fn: Callable, *, qjit: bool) -> Callable:
    """Optionally wrap ``fn`` with Catalyst ``qjit``.

    Raises ImportError-derived ValueError if Catalyst is missing when requested.
    """
    if not qjit:
        return fn
    try:
        from catalyst import qjit as catalyst_qjit
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError(
            "qjit=True requires the Catalyst package. "
            "Install with: pip install -e '.[catalyst]' "
            "(or pip install pennylane-catalyst)."
        ) from exc
    return catalyst_qjit(fn)


def make_qnode(
    circuit_fn: Callable,
    *,
    wires: int = 1,
    device: str = "default.qubit",
    qjit: bool = False,
    interface: str = "jax",
    diff_method: Optional[str] = None,
    device_kwargs: Optional[dict] = None,
) -> Callable:
    """Build a PennyLane QNode with JAX interface; optionally Catalyst-compile it.

    Args:
        circuit_fn: Function that uses ``qml.*`` ops and returns a measurement.
        wires: Number of wires.
        device: PennyLane device name.
        qjit: If True, wrap with Catalyst ``qjit`` (requires Catalyst + supported device).
        interface: Autodiff interface (always ``\"jax\"`` for this library).
        diff_method: Optional PennyLane diff method.
        device_kwargs: Extra kwargs forwarded to ``qml.device``.

    Returns:
        Callable QNode (possibly qjit-compiled).
    """
    if qjit:
        require_catalyst_device(device)

    dev = make_device(device, wires=wires, **(device_kwargs or {}))
    qnode_kwargs: dict[str, Any] = {"interface": interface}
    if diff_method is not None:
        qnode_kwargs["diff_method"] = diff_method

    qnode = qml.QNode(circuit_fn, dev, **qnode_kwargs)
    return maybe_qjit(qnode, qjit=qjit)
