import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from eigenflow.backends import make_qnode, require_catalyst_device


def test_make_qnode_jax_interface():
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = make_qnode(circuit, wires=1, device="default.qubit", qjit=False)
    val = qnode(0.3)
    assert jnp.isfinite(val)


def test_qjit_rejects_default_qubit():
    with pytest.raises(ValueError, match="Catalyst-supported"):
        require_catalyst_device("default.qubit")


def test_qjit_true_without_catalyst_or_bad_device():
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError):
        make_qnode(circuit, wires=1, device="default.qubit", qjit=True)


@pytest.mark.catalyst
def test_qjit_lightning_if_available():
    pytest.importorskip("catalyst")
    try:
        qml.device("lightning.qubit", wires=1)
    except Exception:
        pytest.skip("lightning.qubit not available")

    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = make_qnode(circuit, wires=1, device="lightning.qubit", qjit=True)
    val = qnode(0.2)
    assert jnp.isfinite(jnp.asarray(val))
