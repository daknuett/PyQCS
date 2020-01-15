from pyqcs.graph.state import GraphState
from pyqcs import CZ, X, Z, H
from pyqcs.util.from_lists import graph_lists_to_graph_state

def test_graph_from_list():
    g1 = GraphState.new_plus_state(11)
    g1 = (H(10) | CZ(4, 5) | X(5) | CZ(5, 9) | CZ(9, 10) | H(1) | X(1) | CZ(1, 2) | CZ(2, 4)) * g1

    g2 =graph_lists_to_graph_state(g1._g_state.to_lists())

    assert g1.to_naive_state() == g2.to_naive_state()
