import pytest
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import H, M, State, CZ, CX, S, X, Z, list_to_circuit
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram

def S_with_extra_arg(act, i):
    return S(act)

def do_test_q4_l10_abs():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    nr = State.new_zero_state(4) @ naive
    gr = GraphState.new_zero_state(4) @ graph

    if(abs(gr) != pytest.approx(abs(nr))):
        print("naive", naive)
        print("graph", graph.to_naive_state())
        print("graph:", graph._g_state.to_lists())
    assert abs(gr) == pytest.approx(abs(nr))

@pytest.mark.deprecated
def test_random_q4_l10_abs():
    np.random.seed(0xdeadbeef)
    for _ in range(1000):
        do_test_q4_l10_abs()

def do_test_q4_l10():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    nr = State.new_zero_state(4) @ naive
    gr = GraphState.new_zero_state(4) @ graph

    if(gr != pytest.approx(nr)):
        #print("naive", naive)
        #print("graph", graph.to_naive_state())
        print("graph :", graph._g_state.to_lists())
        print("expect:", nr)
        print("got   :", gr)
        print("abs OK:", abs(nr) == pytest.approx(abs(gr)))
        print("phase :", np.exp(1j * graph._g_state.get_phase()))
        print("circuit:")
        print(circuit_to_diagram(circuit))
    assert gr == pytest.approx(nr)

@pytest.mark.slow
def test_random_q4_l10():
    np.random.seed(0xdeadbeef)
    for _ in range(2000):
        do_test_q4_l10()

def do_test_q4_l100(i):
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4)
    circuit = random_circuit(4, 100, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    nr = State.new_zero_state(4) @ naive
    gr = GraphState.new_zero_state(4) @ graph

    if(gr != pytest.approx(nr)):
        print("FAILED #:", i)
        #print("naive", naive)
        #print("graph", graph.to_naive_state())
        print("graph :", graph._g_state.to_lists())
        print("expect:", nr)
        print("got   :", gr)
        print("abs OK:", abs(nr) == pytest.approx(abs(gr)))
        print("phase :", np.exp(1j * graph._g_state.get_phase()))
        print("circuit:")
        print(circuit_to_diagram(circuit))
    assert gr == pytest.approx(nr)

@pytest.mark.skip(reason="WIP")
def test_random_q4_l100():
    np.random.seed(0xdeadbeef)
    for i in range(1000):
        do_test_q4_l100(i)
