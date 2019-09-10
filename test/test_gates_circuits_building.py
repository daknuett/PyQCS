import pytest

from pyqcs.gates.builtins import X
from pyqcs.gates.circuits import AnonymousCompoundGateCircuit

def test_or_creates_AnonymousCompoundGateCircuit():
    gc1 = X(0)
    gc2 = X(1)

    circuit = gc1 | gc2

    assert isinstance(circuit, AnonymousCompoundGateCircuit)
    
def test_or_preserves_order():
    gc1 = X(0)
    gc2 = X(1)

    circuit = gc1 | gc2

    assert circuit.to_gate_list() == (gc1.to_gate_list() + gc2.to_gate_list())
    assert gc1.to_gate_list() != gc2.to_gate_list()
