from pyqcs.util.random_graphs import random_graph_state

from pyqcs.graph.state import GraphState

def test_type():
    g = random_graph_state(10, 5)

    assert isinstance(g, GraphState)
