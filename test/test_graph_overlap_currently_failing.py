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

#@pytest.mark.skip(reason="WIP")
def test_failing_in_parts():
    #circuit_parts = [
    #        CZ(1, 0)
    #        , S(0)
    #        , CZ(2, 1)
    #        , X(1)
    #        , CZ(0, 2)
    #        , S(1)
    #        , X(1)
    #        , CZ(1, 2)]

    circuit_parts = [
            H(0) | X(0) | S(0) | S(0) | H(2) | S(2)
            | H(3) | S(3) | X(3)
            , CZ(0, 1)
            ]

    graph = GraphState.new_zero_state(4)
    naive = State.new_zero_state(4)

    for i, part in enumerate(circuit_parts):
        graph = part * graph
        naive = part * naive

        #if( GraphState.new_zero_state(4) @ graph
        #        != pytest.approx(State.new_zero_state(4) @ naive)):
        print("after gate", i)
        print("naive        :", naive)
        print("graph(+phase):", graph.to_naive_state())
        print("graph(-phase):", graph.to_naive_state(global_phase=False))
        print("lists:", graph._g_state.to_lists())
        print("phase:", graph._g_state.get_phase())
        print("result:", GraphState.new_zero_state(4) @ graph)
        print("expect:", State.new_zero_state(4) @ naive)
        print()

        assert GraphState.new_zero_state(4) @ graph == pytest.approx(State.new_zero_state(4) @ naive)

