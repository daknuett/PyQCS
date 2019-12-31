from pyqcs import X, H, S, CX, CZ, list_to_circuit, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import graph_state_to_circuit

def test_EPR_state():
    g = GraphState.new_zero_state(20)
    g = H(0) * g
    entagle = list_to_circuit([CX(target, target - 1) for target in range(1, 20)])
    g = entagle * g

    applied = graph_state_to_circuit(g) * State.new_zero_state(20)

    assert applied == g.to_naive_state()
