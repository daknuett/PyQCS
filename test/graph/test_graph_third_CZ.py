import pytest
import numpy as np
from pyqcs import CZ, H, S, X, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import vop_to_circuit

def first_CZ_GraphState():
    g = GraphState.new_plus_state(3)
    g = CZ(1, 0) * g
    return g

def first_CZ_State():
    s = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
    s = CZ(1, 0) * s
    return s

vops = list(range(24))

@pytest.mark.slow
def test_graph_third_CZ():
    failing = []
    for v1_ in vops:
        v1 = vop_to_circuit(0, v1_)
        for v2_ in vops:
            v2 = vop_to_circuit(1, v2_)
            g = (v1 | v2) * first_CZ_GraphState()
            s = (v1 | v2) * first_CZ_State()

            g_bar = CZ(0, 2) * g
            s_bar = CZ(0, 2) * s

            if(g_bar.to_naive_state() != s_bar):
                failing.append((v1_, v2_, s_bar, g_bar))

    print()
    print("#failed:", len(failing))
    print()
    for v1, v2, s, g in failing:
        print()
        print("----------")
        print("vops are:", v1, v2)
        print("graph lists are:", g._g_state.to_lists())
        print()
        print("expected")
        print(s)
        print("got")
        print(g.to_naive_state())
    assert failing == []
