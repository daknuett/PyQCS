import pytest

from pyqcs.gates.builtins import H, M, R, X, CX, CZ
from pyqcs.state.state import DSVState


@pytest.fixture
def zero_state():
    return DSVState.new_zero_state(2)


def test_X0(zero_state):
    new_state = X(0) * zero_state

    assert new_state._qm_state == pytest.approx([0, 1, 0, 0])


def test_X01(zero_state):
    new_state = (X(0) | X(1)) * zero_state

    assert new_state._qm_state == pytest.approx([0, 0, 0, 1])


def test_CZ01_10(zero_state):
    new_state = (X(0) | CZ(0, 1)) * zero_state

    assert new_state._qm_state == pytest.approx([0, 1, 0, 0])


def test_CZ01_11(zero_state):
    new_state = (X(1) | X(0) | CZ(0, 1)) * zero_state

    assert new_state._qm_state == pytest.approx([0, 0, 0, -1])


def test_CZ01_01(zero_state):
    new_state = (X(1) | CZ(0, 1)) * zero_state

    print(new_state._qm_state)
    assert new_state._qm_state == pytest.approx([0, 0, 1, 0])
