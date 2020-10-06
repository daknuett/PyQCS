import pytest
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import H, M, State, CZ, CX, S, X, Z, list_to_circuit
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram
from pyqcs.util.to_circuit import vop_to_circuit

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

@pytest.mark.skip(reason="WIP")
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

    #circuit_parts = [
    #        H(0) | X(0) | S(0) | S(0) | H(2) | S(2) | H(3) | S(3) | X(3)
    #        , CZ(0, 1)
    #        ]

    #circuit_parts = [
    #        H(3) | S(3) | X(3) | S(3) | X(3) | S(3) | H(3) | X(3) | H(3) | X(3) | H(3) | S(3)
    #        | H(0) | S(0) | S(1) | H(1) | X(1)
    #        , Z(2)
    #        , CZ(2, 1)
    #        , CZ(0, 2)
    #        , H(1) | S(1) | S(0)
    #        , CZ(2, 0) | CZ(1, 0)
    #        , H(0) | X(2) | S(2)
    #        , CZ(0, 1)
    #        , X(1) | S(1)
    #        , CZ(2, 1) | CZ(2, 0)
    #        ]

    #circuit_parts = [
    #        S(1) | H(1) | H(3) | X(3) | CZ(2, 0) | H(0) | CZ(0, 2) | CZ(1, 0) | X(1)
    #        , CZ(2, 1)
    #        ]

    circuit_parts = [
            H(0) | CZ(0, 2) | X(0) | H(0) | X(0) | S(0) | H(1) | Z(1) | X(1) | Z(2) | H(2) | S(2) | X(2) | S(2) | CZ(1, 2) | X(1) | H(1) | H(2) | S(2) | CZ(1, 0) | H(1) | CZ(0, 2) | CZ(0, 1) | X(1) | X(0) | H(0) | S(0) | H(0) | S(0) | CZ(0, 2) | H(0) | S(2) | S(2)
            , CZ(0, 1)
            , S(0) | H(0) | H(1)
            , CZ(0, 1)
            , X(0) | Z(0) | S(0) | H(0)
            , CZ(2, 0)
            , S(2)
            , CZ(1, 2)
            , CZ(1, 0)
            , H(2)
            , CZ(1, 2)
            , X(0) | S(0) | H(1) | X(1)
            , CZ(1, 2)
            , X(1) | X(2) | H(2) | S(2) | S(2) | S(2)
            , CZ(1, 0)
            , S(1) | X(1) | H(1)
            , CZ(1, 0) | CZ(1, 0)
            , Z(0) | X(0) | S(1)

            ]

    graph = GraphState.new_zero_state(4)
    naive = State.new_zero_state(4)

    graphp = GraphState.new_plus_state(4)
    naivep = list_to_circuit([H(i) for i in range(4)]) * State.new_zero_state(4)

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
        print("result |0>:", GraphState.new_zero_state(4) @ graph)
        print("expect |0>:", State.new_zero_state(4) @ naive)
        print("result |+>:", graphp @ graph)
        print("expect |+>:", naivep @ naive)
        print()

        assert GraphState.new_zero_state(4) @ graph == pytest.approx(State.new_zero_state(4) @ naive)
        assert graphp @ graph == pytest.approx(naivep @ naive)

@pytest.mark.deprecated
def test_failing_gstate():
    vops = [21, 1, 22, 6]
    g = GraphState.new_plus_state(4)
    g = CZ(0, 2) * g
    p = list_to_circuit([H(i) for i in range(4)]) * State.new_zero_state(4)
    p = CZ(0, 2) * p
    for i,v in enumerate(vops):
        g = vop_to_circuit(i,v) * g
        p = vop_to_circuit(i,v) * p

    expect = State.new_zero_state(4) @ p
    got = GraphState.new_zero_state(4) @ g

    print("expect:", expect)
    print("got   :", got)
    print("phase :", g._g_state.get_phase())

    assert got == pytest.approx(expect)

