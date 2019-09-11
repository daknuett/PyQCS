from pyqcs.gates.circuits import SingleGateCircuit, AnonymousCompoundGateCircuit, NamedCompoundGateCircuit
import pytest

def test_construct_circuit():
    H1 = SingleGateCircuit(1, [], "H", "H1")
    X1 = SingleGateCircuit(1, [], "X", "X1")
    H2 = SingleGateCircuit(2, [], "H", "H2")
    X2 = SingleGateCircuit(2, [], "X", "X2")

    assert H1.to_executor().to_gate_list() == ["H1"]
    assert X1.to_executor().to_gate_list() == ["X1"]
    assert H2.to_executor().to_gate_list() == ["H2"]
    assert X2.to_executor().to_gate_list() == ["X2"]

    assert (H1 | H2 | X1 | X2).to_executor().to_gate_list() == ["H1", "H2", "X1", "X2"]

def test_name_constructed_circuit():
    H1 = SingleGateCircuit(1, [], "H", "H1")
    X1 = SingleGateCircuit(1, [], "X", "X1")
    H2 = SingleGateCircuit(2, [], "H", "H2")
    X2 = SingleGateCircuit(2, [], "X", "X2")

    c = NamedCompoundGateCircuit.from_anonymous(H1 | H2 | X1 | X2, "HX")

    assert c.to_executor().to_gate_list() == ["H1", "H2", "X1", "X2"]
    assert c._name == "HX"

def test_nested_construct():
    H1 = SingleGateCircuit(1, [], "H", "H1")
    X1 = SingleGateCircuit(1, [], "X", "X1")
    H2 = SingleGateCircuit(2, [], "H", "H2")
    X2 = SingleGateCircuit(2, [], "X", "X2")

    assert ((H1 | H2 ) | (X1 | X2)).to_executor().to_gate_list() == ["H1", "H2", "X1", "X2"]
