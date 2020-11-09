import pytest

from pyqcs import CZ, H, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import vop_to_circuit

@pytest.fixture
def vop_free_test_states():
    g = GraphState.new_plus_state(3)
    g = CZ(0, 2) * g
    n = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
    n = CZ(0, 2) * n
    return g,n

def test_clear_vops(vop_free_test_states):
    v = 3
    g, n = vop_free_test_states
    g = vop_to_circuit(0, v) * g
    n = vop_to_circuit(0, v) * n

    g = CZ(0, 1) * g
    n = CZ(0, 1) * n

    assert g.to_naive_state() @ n == pytest.approx(1)

def test_clear_vops2(vop_free_test_states):
    g, n = vop_free_test_states
    v1 = 1 # VOP_smiZ
    v2 = 6 # VOP_siX
    g = (vop_to_circuit(0, v1) | vop_to_circuit(0, v2)) * g
    n = (vop_to_circuit(0, v1) | vop_to_circuit(0, v2)) * n

    g = CZ(0, 1) * g
    n = CZ(0, 1) * n

    assert g.to_naive_state() @ n == pytest.approx(1)
