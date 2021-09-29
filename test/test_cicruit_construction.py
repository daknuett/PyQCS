import pytest

from pyqcs import State, CX, H, X, R


@pytest.fixture
def zero_state():
    return State.new_zero_state(2)
