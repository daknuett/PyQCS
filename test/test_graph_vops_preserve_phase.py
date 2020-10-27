import pytest

from itertools import product

from pyqcs.graph.state import GraphState
from pyqcs import State, H, CZ
from pyqcs.util.to_circuit import graph_state_to_circuit, vop_to_circuit, C_L

vops = list(range(24))

@pytest.fixture
def plus_ket3():
    return (H(0) | H(1) | H(2)) * State.new_zero_state(3)

@pytest.mark.skip(reason="WIP")
def test_vops_preserve_phase_fully_entangled(plus_ket3):
    g = (CZ(0, 1) | CZ(0, 2) | CZ(1, 2)) * GraphState.new_plus_state(3)
    n = (CZ(0, 1) | CZ(0, 2) | CZ(1, 2)) * plus_ket3

    for v1, v2, v3 in product(vops, vops, vops):
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print(f"current vops: {C_L[v1]}(0) | {C_L[v2]}(1) | {C_L[v3]}(2)")
        assert g @ GraphState.new_plus_state(3) == pytest.approx(n @ plus_ket3)
        assert g @ GraphState.new_zero_state(3) == pytest.approx(n @ State.new_zero_state(3))

@pytest.mark.skip(reason="WIP")
def test_vops_preserve_phase_linearly_entangled(plus_ket3):
    g = (CZ(0, 1) | CZ(0, 2)) * GraphState.new_plus_state(3)
    n = (CZ(0, 1) | CZ(0, 2)) * plus_ket3

    for v1, v2, v3 in product(vops, vops, vops):
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print(f"current vops: {C_L[v1]}(0) | {C_L[v2]}(1) | {C_L[v3]}(2)")
        assert g @ GraphState.new_plus_state(3) == pytest.approx(n @ plus_ket3)
        assert g @ GraphState.new_zero_state(3) == pytest.approx(n @ State.new_zero_state(3))

@pytest.mark.skip(reason="WIP")
def test_vops_preserve_phase_singly_entangled(plus_ket3):
    g = (CZ(0, 1)) * GraphState.new_plus_state(3)
    n = (CZ(0, 1)) * plus_ket3

    for v1, v2, v3 in product(vops, vops, vops):
        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print(f"current vops: {C_L[v1]}(0) | {C_L[v2]}(1) | {C_L[v3]}(2)")
        #print("current state (w/ phase) :", g.to_naive_state())
        #print("current state (w/o phase):", g.to_naive_state(global_phase=False))
        #print("current naive state      :", n)
        assert g @ GraphState.new_plus_state(3) == pytest.approx(n @ plus_ket3)
        assert g @ GraphState.new_zero_state(3) == pytest.approx(n @ State.new_zero_state(3))

#@pytest.mark.slow
#@pytest.mark.selected
@pytest.mark.skip(reason="WIP")
def test_vops_preserve_phase_unentangled(plus_ket3):
    for v1, v2, v3 in product(vops, vops, vops):
        g = GraphState.new_plus_state(3)
        n = plus_ket3

        g = vop_to_circuit(0, v1) * g
        g = vop_to_circuit(1, v2) * g
        g = vop_to_circuit(2, v3) * g
        n = vop_to_circuit(0, v1) * n
        n = vop_to_circuit(1, v2) * n
        n = vop_to_circuit(2, v3) * n

        print("vops:", v1, v2, v3)
        #print(f"current vops: {C_L[v1]}(0) | {C_L[v2]}(1) | {C_L[v3]}(2)")
        #print("current state (w/ phase) :", g.to_naive_state())
        #print("current state (w/o phase):", g.to_naive_state(global_phase=False))
        #print("current naive state      :", n)
        #assert g @ GraphState.new_plus_state(3) == pytest.approx(n @ plus_ket3)
        assert g @ GraphState.new_zero_state(3) == pytest.approx(n @ State.new_zero_state(3))
