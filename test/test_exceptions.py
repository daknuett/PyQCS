import pytest

from pyqcs import State, X


def test_missing_qbit1():
    state = State.new_zero_state(2)

    with pytest.raises(ValueError):
        assert X(2) * state


def test_missing_qbitmany():
    state = State.new_zero_state(2)

    with pytest.raises(ValueError):
        assert X(200) * state
