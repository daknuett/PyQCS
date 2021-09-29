import pytest
import numpy as np
from pyqcs import CZ, H, S, X, M, State
from pyqcs.graph.state import GraphState
from pyqcs.util.random_circuits import random_circuit
from pyqcs.util.to_diagram import circuit_to_diagram

def S_with_extra_arg(act, i):
    return S(act)

def do_test_q4_l10():
    naive = State.new_zero_state(4)
    graph = GraphState.new_zero_state(4, copy=True)
    circuit = random_circuit(4, 10, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    results_naive = {}
    results_graph = {}
    cnt = 0
    while((0 not in results_naive
                or 1 not in results_naive
                or 0 not in results_graph
                or 1 not in results_graph)
            and cnt < 100):
        g_bar = M(0) * graph
        n_bar = M(0) * naive

        if(n_bar._cl_state[0] in results_naive
                and g_bar._measured[0] in results_graph):
            cnt += 1
        results_naive[n_bar._cl_state[0]] = n_bar
        results_graph[g_bar._measured[0]] = g_bar

    for (kn, vn), (kg, vg) in zip(sorted(results_naive.items())
                                , sorted(results_graph.items())):
        assert kn == kg
        vn._cl_state[0] = -1
        if(vn != vg.to_naive_state()):
            print("before:")
            print(naive)
            print("lists:", graph._g_state.to_lists())
            print()
            print("expected:")
            print(vn)
            print("got:")
            print(vg.to_naive_state())
            print("lists:", vg._g_state.to_lists())
            print()
            print("all results:")
            print("expect:")
            print(results_naive)
            print("got:")
            print({k: v.to_naive_state() for k,v in results_graph.items()})
        assert vn == vg.to_naive_state()

@pytest.mark.slow
def test_random_q4_l10_m0():
    np.random.seed(0xdeadbeef)
    for _ in range(1000):
        do_test_q4_l10()

def do_test_q5_l100():
    naive = State.new_zero_state(5)
    graph = GraphState.new_zero_state(5, copy=True)
    circuit = random_circuit(5, 100, X, H, S_with_extra_arg, CZ)

    naive = circuit * naive
    graph = circuit * graph

    results_naive = {}
    results_graph = {}
    cnt = 0
    while((0 not in results_naive
                or 1 not in results_naive
                or 0 not in results_graph
                or 1 not in results_graph)
            and cnt < 100):
        g_bar = M(0) * graph
        n_bar = M(0) * naive

        if(n_bar._cl_state[0] in results_naive
                and g_bar._measured[0] in results_graph):
            cnt += 1
        results_naive[n_bar._cl_state[0]] = n_bar
        results_graph[g_bar._measured[0]] = g_bar

    for (kn, vn), (kg, vg) in zip(sorted(results_naive.items())
                                , sorted(results_graph.items())):
        assert kn == kg
        vn._cl_state[0] = -1
        if(vn != vg.to_naive_state()):
            print("before:")
            print(naive)
            print("lists:", graph._g_state.to_lists())
            print()
            print("expected:")
            print(vn)
            print("got:")
            print(vg.to_naive_state())
            print("lists:", vg._g_state.to_lists())
            print()
            print("all results:")
            print("expect:")
            print(results_naive)
            print("got:")
            print({k: v.to_naive_state() for k,v in results_graph.items()})
        assert vn == vg.to_naive_state()

@pytest.mark.slow
def test_random_q5_l100_m0():
    np.random.seed(0xdeadbeef)
    for _ in range(1000):
        do_test_q5_l100()
