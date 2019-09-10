import pytest

from pyqcs.gates.builtins import H, M, R, X, C
from pyqcs.state.state import BasicState

@pytest.fixture
def zero_state():
    return BasicState.new_zero_state(2)

def test_X0(zero_state):
    new_state = X(0) * zero_state

    assert new_state._qm_state == pytest.approx([0, 1, 0, 0])

def test_X01(zero_state):
    new_state = (X(0) | X(1)) * zero_state

    assert new_state._qm_state == pytest.approx([0, 0, 0, 1])
