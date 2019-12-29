import pytest
from pyqcs import CZ, H, S, X, State
from pyqcs.graph.state import GraphState
from pyqcs.util.random_circuits import random_circuit

def S_with_extra_arg(act, i):
    return S(act)

def do_test_q4_l10():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    if(naive != graph.to_naive_state()):
        print()
        print("naive", naive)
        print("graph", graph.to_naive_state())
    assert naive == graph.to_naive_state()

def test_random_q4_l10():
    for _ in range(1000):
        do_test_q4_l10()

def do_test_q10_l100():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(10, 100, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    assert naive == graph.to_naive_state()
@pytest.mark.slow
def test_random_q10_l100():
    for _ in range(4000):
        do_test_q10_l100()

