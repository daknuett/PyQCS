import pytest

from pyqcs.gates.builtins import X, H
from pyqcs.gates.circuits import AnonymousCompoundGateCircuit
from pyqcs.state.state import BasicState

def test_or_creates_AnonymousCompoundGateCircuit():
    gc1 = X(0)
    gc2 = X(1)

    circuit = gc1 | gc2

    assert isinstance(circuit, AnonymousCompoundGateCircuit)

def test_or_preserves_order():
    gc1 = X(0)
    gc2 = X(1)

    circuit = gc1 | gc2

    assert circuit.to_executor().to_gate_list() == (gc1.to_executor().to_gate_list() + gc2.to_executor().to_gate_list())
    assert gc1.to_executor().to_gate_list() != gc2.to_executor().to_gate_list()

def test_XH():
    M_SQRT1_2 = 0.70710678118654752440
    state = BasicState.new_zero_state(2)

    state = (X(0) | H(0)) * state

    assert state._qm_state == pytest.approx([M_SQRT1_2, -M_SQRT1_2, 0, 0])
