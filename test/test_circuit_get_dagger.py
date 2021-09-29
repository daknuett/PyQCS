import pytest
from pyqcs import  M, CZ, CZ, H, R, S, X, State
from pyqcs.graph.state import GraphState
from pyqcs.gates.exceptions import UnitarityError


def test_c_dg_1():
    circuit = ((H(0) | S(1) | H(3) | S(3) | CZ(0, 2) | CZ(1, 3) | H(1) | CZ(0, 2))
                | (X(0) | CZ(0, 1) | H(1) | S(1) | CZ(1, 2)))
    circuit_dagger = circuit.get_dagger()
    state = State.new_zero_state(4)

    s_bar = circuit * state
    s_barbar = circuit_dagger * s_bar

    assert s_barbar == state


def test_c_dg_1g():
    circuit = ((H(0) | S(1) | H(3) | S(3) | CZ(0, 2) | CZ(1, 3) | H(1) | CZ(0, 2))
                | (X(0) | CZ(0, 1) | H(1) | S(1) | CZ(1, 2)))
    circuit_dagger = circuit.get_dagger()
    state = GraphState.new_plus_state(4)

    s_bar = circuit * state
    s_barbar = circuit_dagger * s_bar

    assert s_barbar.to_naive_state() == state.to_naive_state()


def test_S_dg():
    circuit = S(0)

    circuit_dagger = circuit.get_dagger()

    assert (circuit | circuit_dagger) * State.new_zero_state(1) == State.new_zero_state(1)


def test_M_dg():
    circuit = H(0) | M(0)

    with pytest.raises(UnitarityError):
        circuit_dagger = circuit.get_dagger()

    assert 1

