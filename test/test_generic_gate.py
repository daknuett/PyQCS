import numpy as np

from pyqcs.gates.gate import GenericGate
from pyqcs import State, X, H, Z

def test_X():
    state = State.new_zero_state(3)
    gate = GenericGate(0, np.array([[0, 1], [1, 0]], dtype=np.cdouble))

    assert state.apply_gate(gate) == X(0) * state


def test_XZ():
    state = State.new_zero_state(3)
    x = np.array([[0, 1], [1, 0]], dtype=np.cdouble)
    z = np.array([[1, 0], [0, -1]], dtype=np.cdouble)
    xz = x.dot(z)
    gate = GenericGate(1, xz)

    assert state.apply_gate(gate) == (Z(1) | X(1)) * state


def test_HX():
    state = State.new_zero_state(3)
    x = np.array([[0, 1], [1, 0]], dtype=np.cdouble)
    h = np.array([[1, 1], [1, -1]], dtype=np.cdouble) / np.sqrt(2)
    hx = h.dot(x)
    gate = GenericGate(1, hx)

    assert state.apply_gate(gate) == (X(1) | H(1)) * state

