import pytest
import numpy as np
from pyqcs import CZ, H, S, X, State
from pyqcs.graph.state import GraphState
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram

def S_with_extra_arg(act, i):
    return S(act)

def do_test_q4_l10():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    if(naive @ graph.to_naive_state() != pytest.approx(1)):
        print("naive", naive)
        print("graph", graph.to_naive_state())
        print("circuit")
        print(circuit_to_diagram(circuit))
    assert naive == graph.to_naive_state()

@pytest.mark.slow
def test_random_q4_l10():
    np.random.seed(0xdeadbeef)
    for _ in range(1000):
        do_test_q4_l10()

def test_random_q4_l10_redux():
    np.random.seed(2)
    for _ in range(100):
        do_test_q4_l10()

def do_test_q10_l100():
    naive = State.new_zero_state(10)
    graph = GraphState.new_zero_state(10)
    circuit = random_circuit(10, 100, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    if(naive @ graph.to_naive_state() != pytest.approx(1)):
        print("circuit")
        print(circuit_to_diagram(circuit))
        print("naive\n", naive)
        print("graph\n", graph.to_naive_state())
    assert naive == graph.to_naive_state()

@pytest.mark.slow
def test_random_q10_l100():
    np.random.seed(1)
    for _ in range(10000):
        do_test_q10_l100()

