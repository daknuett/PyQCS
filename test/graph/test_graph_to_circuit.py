from numpy import random
import pytest

from pyqcs import X, H, S, CX, CZ, list_to_circuit, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import graph_state_to_circuit
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram

def test_EPR_state():
    g = GraphState.new_zero_state(20)
    g = H(0) * g
    entagle = list_to_circuit([CX(target, target - 1) for target in range(1, 20)])
    g = entagle * g

    applied = graph_state_to_circuit(g) * State.new_zero_state(20)

    assert applied == g.to_naive_state()

def S_with_extra_arg(act, i):
    return S(act)

def do_test_q4_l10():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)
    print("original")
    print(circuit_to_diagram(circuit))

    naive = circuit * naive
    graph = circuit * graph
    ng = graph_state_to_circuit(graph) * State.new_zero_state(4)
    print("converted")
    print(circuit_to_diagram(graph_state_to_circuit(graph)))

    if(naive != ng):
        print("naive", naive)
        print("graph", graph.to_naive_state())
        print("from circuit", ng)
    assert naive == ng

@pytest.mark.slow
def test_random_q4_l10():
    random.seed(2)
    for _ in range(5000):
        do_test_q4_l10()
