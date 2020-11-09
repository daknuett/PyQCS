import pytest
from itertools import product

from pyqcs import State, H, X, Z, CZ, list_to_circuit
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import vop_to_circuit
from pyqcs import GenericGate

@pytest.mark.selected
def test_two_qbits_isolated_CZ():
    for c0, c1 ,e in product(range(24), range(24), (True, False)):
        g = GraphState.new_plus_state(2)
        s = (H(0) | H(1)) * State.new_zero_state(2)

        circuit = vop_to_circuit(0, c0) | vop_to_circuit(1, c1)
        if(e):
            circuit = CZ(0, 1) | circuit

        g = circuit * g
        s = circuit * s

        assert g.to_naive_state() @ s == pytest.approx(1)
