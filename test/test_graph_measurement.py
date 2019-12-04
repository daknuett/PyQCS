import numpy as np

from pyqcs.graph.backend.raw_state import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state
from pyqcs import H, M, State

def test_deterministic_unentangled_measurement():
    g = RawGraphState(4)

    g.apply_C_L(0, 0)

    for _ in range(10):
        assert g.measure(0, 0.4) == 0


def test_random_z_graph_update():
    for i in range(10):
        s = (H(0) | H(1) | H(2) | H(3)) * State.new_zero_state(4)
        s_bar = M(0) * s
        result = s_bar._cl_state[0]
        s_bar._cl_state[0] = -1

        g = RawGraphState(4)
        r = g.measure(0, result)

        assert r == result
        assert graph_lists_to_naive_state(g.to_lists()) == s_bar
