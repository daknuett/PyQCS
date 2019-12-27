from pyqcs import H, X, CX, CZ, M, State, list_to_circuit
from pyqcs.graph.state import GraphState

def test_new_zero_state():
    assert GraphState.new_zero_state(10).to_naive_state() == State.new_zero_state(10)

def test_new_plus_state():
    g_state = GraphState.new_plus_state(10).to_naive_state()
    n_state = list_to_circuit([H(i) for i in range(10)]) * State.new_zero_state(10)

    assert g_state == n_state

def test_bell_state_naive_and_graph():
    circuit = (H(0) | H(1)) | CX(1, 0)

    n_state = State.new_zero_state(2)
    g_state = GraphState.new_zero_state(2)

    n_state = circuit * n_state
    g_state = circuit * g_state


    assert g_state.to_naive_state() == n_state

def test_measurement_result():
    g = GraphState.new_plus_state(2)
    g = (CZ(1, 0) | H(1) | M(1) | M(0)) * g

    assert g._measured[0] == g._measured[1]

    if(g._measured[0] == 0):
        assert g.to_naive_state() == State.new_zero_state(2)
    else:
        assert g.to_naive_state() == (X(0) | X(1)) * State.new_zero_state(2)
