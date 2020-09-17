import pytest
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import H, M, State, CZ, CX, S, X, Z, list_to_circuit
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram

@pytest.mark.deprecated
def test_is_failing_right_now():
    # circuit = (CZ(1, 2) | H(3) | H(2) | S(1) | H(1) | S(0) | H(0))
    plus_state_circuit = list_to_circuit([H(i) for i in range(4)])
    circuit = (S(0) | H(0) | H(1) | X(2) | H(3) | plus_state_circuit)
    g = circuit * GraphState.new_plus_state(4)
    plus_state = plus_state_circuit * State.new_zero_state(4)
    result_state = (plus_state_circuit | circuit) * State.new_zero_state(4)
    print(g._g_state.to_lists())

    res = GraphState.new_plus_state(4) @ g

    assert abs(res) == pytest.approx(abs(plus_state @ result_state))

# ([4, 0, 14, 0], [[], [], [], []])
