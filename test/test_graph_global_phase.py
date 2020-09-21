import pytest
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import H, M, State, CZ, CX, S, X, Z, list_to_circuit
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram


@pytest.mark.skip(reason="debugging")
def test_global_phase_S():
    graph = (H(0) | X(0) | S(0)) * GraphState.new_plus_state(1)

    result = ((H(0) | X(0)) * GraphState.new_plus_state(1)) @ graph

    print(graph._g_state.to_lists())
    print(np.exp(1j * graph._g_state.get_phase()))

    assert result == pytest.approx(1j)

@pytest.mark.skip(reason="debugging")
def test_global_phase_XSX():
    graph = (H(0) | X(0) | S(0) | X(0)) * GraphState.new_plus_state(1)

    result = ((H(0)) * GraphState.new_plus_state(1)) @ graph

    print(graph._g_state.to_lists())
    print(np.exp(1j * graph._g_state.get_phase()))
    print("g-state(+phase):", graph.to_naive_state())
    print("g-state(-phase):", graph.to_naive_state(global_phase=False))

    assert result == pytest.approx(1j)
