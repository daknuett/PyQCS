import pytest

from pyqcs.graph.backend.raw_state import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state

from pyqcs import H, Z, CZ, State

@pytest.fixture
def plus_state():
     return (H(0) | H(1)) * State.new_zero_state(2)


def test_h0(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(0, 0)
    test_state = H(0) * plus_state
    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == test_state

def test_z0(plus_state):
    g = RawGraphState(2)
    g.apply_C_L(0, 5)
    test_state = Z(0) * plus_state
    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == test_state
