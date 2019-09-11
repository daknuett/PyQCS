import pytest

from pyqcs import State, C, H, X, R

@pytest.fixture
def zero_state():
    return State.new_zero_state(2)

def test_repeating_gates():
    base = X(0) | H(0)

    multiplied = base * 5

    assert multiplied.to_executor().to_gate_list() == (5 * base.to_executor().to_gate_list())

def test_repeating_gates_inverted():
    base = X(0) | H(0)

    multiplied = 5 * base

    assert multiplied.to_executor().to_gate_list() == (5 * base.to_executor().to_gate_list())
