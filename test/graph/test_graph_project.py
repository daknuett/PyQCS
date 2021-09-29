import pytest

from itertools import product
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import State, H, CZ
from pyqcs.util.to_circuit import graph_state_to_circuit, vop_to_circuit, C_L

vops = list(range(24))

@pytest.fixture
def plus_ket3():
    return (H(0) | H(1) | H(2)) * State.new_zero_state(3)

def test_vops_preserve_phase_unentangled(plus_ket3):
    for v1 in vops:
        g = GraphState.new_plus_state(1)
        n = H(0) * State.new_zero_state(1)
        g = vop_to_circuit(0, v1) * g
        n = vop_to_circuit(0, v1) * n

        print("vops:", v1)
        res = g.project_to(0, "Z")
        print("lists: ", g._g_state.to_lists())
        print("graph: ", g.to_naive_state())
        print("dsv before projection: ", n)
        print("dsv after projection: ", n.project_Z(0, 0))

        if(res == 0):
            assert n.project_Z(0, 0) == 0
        else:
            assert np.abs(g.to_naive_state() @ n.project_Z(0, 0)) == pytest.approx(1)

@pytest.mark.slow
def test_vops_preserve_phase_singly_entangled(plus_ket3):
    for v1,v2 in product(vops, vops):
        g = CZ(0, 1) * GraphState.new_plus_state(2)
        n = (H(1) | H(0) | CZ(0, 1)) * State.new_zero_state(2)
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n

        print("vops:", v1, v2)
        res = g.project_to(0, "Z")
        print(g._g_state.to_lists())

        if(res == 0):
            assert n.project_Z(0, 0) == 0
        else:
            assert g.to_naive_state() == n.project_Z(0, 0)

@pytest.mark.slow
def test_vops_preserve_phase_star_entangled(plus_ket3):
    for v1, v2, v3 in product(vops, vops, vops):
        g = (CZ(0, 1) | CZ(0, 2)) * GraphState.new_plus_state(3)
        n = (CZ(0, 1) | CZ(0, 2)) * plus_ket3
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print("vops:", v1, v2, v3)
        res = g.project_to(0, "Z")
        print(g._g_state.to_lists())

        if(res == 0):
            assert n.project_Z(0, 0) == 0
        else:
            assert np.abs(g.to_naive_state() @ n.project_Z(0, 0)) == pytest.approx(1)

@pytest.mark.slow
def test_vops_preserve_phase_loop_entangled(plus_ket3):
    for v1, v2, v3 in product(vops, vops, vops):
        g = (CZ(0, 1) | CZ(0, 2) | CZ(1, 2)) * GraphState.new_plus_state(3)
        n = (CZ(0, 1) | CZ(0, 2) | CZ(1, 2)) * plus_ket3
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print("vops:", v1, v2, v3)
        res = g.project_to(0, "Z")
        print(g._g_state.to_lists())

        if(res == 0):
            assert n.project_Z(0, 0) == 0
        else:
            assert np.abs(g.to_naive_state() @ n.project_Z(0, 0)) == pytest.approx(1)
