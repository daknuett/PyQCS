import pytest

from pyqcs.graph.rawstate import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state

from pyqcs import H, X, Z, CZ, State

@pytest.fixture
def plus_state():
     return (H(0) | H(1)) * State.new_zero_state(2)


def test_h0(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(0, 0)
    test_state = H(0) * plus_state

    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == test_state

def test_h01(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(0, 0)
    g.apply_C_L(1, 0)
    test_state = (H(0) | H(1)) * plus_state

    converted = graph_lists_to_naive_state(g.to_lists())

    print(g.to_lists())
    print("converted", converted)
    print("test_state", test_state)
    assert converted == test_state

def test_z0(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(0, 5)
    test_state = Z(0) * plus_state

    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == test_state

def test_z1(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(1, 5)
    test_state = Z(1) * plus_state

    converted = graph_lists_to_naive_state(g.to_lists())

    print(g.to_lists())
    print("converted", converted)
    print("test_state", test_state)
    assert converted == test_state


def test_cz01_id(plus_state):
    g = RawGraphState(2)
    g.apply_CZ(0, 1)
    test_state = (CZ(0, 1)) * plus_state

    converted = graph_lists_to_naive_state(g.to_lists())

    print(g.to_lists())
    print("converted", converted)
    print("test_state", test_state)
    assert converted == test_state


def test_graph_lists2naive_state():
    g = RawGraphState(3)
    g.apply_CZ(1, 0)
    g.apply_CZ(2, 0)
    g.apply_CZ(1, 2)

    state = (H(0) | H(1) | H(2) | CZ(1, 0) | CZ(2, 0) | CZ(1, 2)) * State.new_zero_state(3)
    g.apply_C_L(0, 0)
    g.apply_C_L(1, 14)
    g.apply_CZ(1, 0)
    state = (H(0) | X(1) | CZ(1, 0)) * state

    print("converted", graph_lists_to_naive_state(g.to_lists()))
    print("naive", state)
    assert graph_lists_to_naive_state(g.to_lists()) == state

