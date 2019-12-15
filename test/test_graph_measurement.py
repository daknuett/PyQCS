import numpy as np

from pyqcs.graph.backend.raw_state import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state
from pyqcs import H, M, State, CZ, R

def test_deterministic_unentangled_measurement():
    g = RawGraphState(4)

    g.apply_C_L(0, 0)

    for _ in range(10):
        assert g.measure(0, 0.4) == 0


def test_random_z_graph_update():
    for i in range(10):
        s = (H(0) | H(1) ) * State.new_zero_state(2)
        s_bar = M(0) * s
        result = s_bar._cl_state[0]
        s_bar._cl_state[0] = -1

        g = RawGraphState(2)
        r = g.measure(0, result)

        print(result)
        print("naive", s_bar)
        print("converted", graph_lists_to_naive_state(g.to_lists()))
        assert r == result
        assert graph_lists_to_naive_state(g.to_lists()) == s_bar

def test_random_z_graph_update_entangled():
    for i in range(10):
        s = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
        s = (CZ(0, 1) | CZ(0, 2)) * s
        s_bar = M(0) * s
        result = s_bar._cl_state[0]
        s_bar._cl_state[0] = -1

        g = RawGraphState(3)
        g.apply_CZ(0, 1)
        g.apply_CZ(0, 2)
        r = g.measure(0, result)

        print(result)
        print("naive", s_bar)
        print("converted", graph_lists_to_naive_state(g.to_lists()))
        print("lists:", g.to_lists())
        assert r == result
        assert graph_lists_to_naive_state(g.to_lists()) == s_bar

def test_random_y_graph_update():
    for i in range(10):
        s = (H(0) | H(1) ) * State.new_zero_state(2)
        s = (H(0) | R(0, np.pi/2)) * s
        s_bar = M(0) * s
        result = s_bar._cl_state[0]
        s_bar._cl_state[0] = -1

        g = RawGraphState(2)
        g.apply_C_L(0, 10)
        print("graph", g.to_lists())
        r = g.measure(0, result)

        print(result)
        print("naive", s_bar)
        print("converted", graph_lists_to_naive_state(g.to_lists()))
        print("graph", g.to_lists())
        assert r == result
        assert graph_lists_to_naive_state(g.to_lists()) == s_bar

def test_random_y_graph_update_entangled():
    for i in range(10):
        s = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
        s = (CZ(0, 1)# | CZ(0, 2)
                ) * s
        s = (H(0) | R(0, np.pi/2)) * s
        s_bar = M(0) * s
        result = s_bar._cl_state[0]
        s_bar._cl_state[0] = -1

        g = RawGraphState(3)
        g.apply_CZ(0, 1)
        #g.apply_CZ(0, 2)
        g.apply_C_L(0, 10)
        r = g.measure(0, result)

        print(result)
        print("naive", s_bar)
        print("converted", graph_lists_to_naive_state(g.to_lists()))
        print("lists:", g.to_lists())
        assert r == result
        assert graph_lists_to_naive_state(g.to_lists()) == s_bar
