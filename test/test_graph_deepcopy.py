import pytest

from pyqcs.graph.state import GraphState

@pytest.fixture
def graph_plus_state4():
    return GraphState.new_plus_state(4)

def test_deepcopy_plus_state(graph_plus_state4):
    g = graph_plus_state4.deepcopy()

    assert g.to_naive_state() == graph_plus_state4.to_naive_state()
    
def test_deepcopy_plus_state_mod(graph_plus_state4):
    g = graph_plus_state4.deepcopy()
    g._g_state.apply_C_L(0, 0)

    assert g.to_naive_state() != graph_plus_state4.to_naive_state()

