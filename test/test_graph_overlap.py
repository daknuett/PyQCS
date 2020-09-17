import pytest
import numpy as np

from pyqcs.graph.state import GraphState
from pyqcs import H, M, State, CZ, CX, S, X, Z, list_to_circuit

@pytest.fixture
def predefined_test_circuits():
    return [
            list_to_circuit([H(i) for i in range(4)])
            , list_to_circuit([S(i) for i in range(4)])
            , list_to_circuit([Z(i) for i in range(4)])
            , list_to_circuit([CZ(0, i) for i in range(1, 4)])
            , list_to_circuit([CX(0, i) for i in range(1, 4)])
            ]

@pytest.mark.slow
def test_trivial_overlap_1(predefined_test_circuits):
    for circuit in predefined_test_circuits:

        g = circuit * GraphState.new_plus_state(4)

        assert g @ g == pytest.approx(1)

@pytest.mark.slow
def test_overlap_0():
    gb = GraphState.new_plus_state(4)

    g = list_to_circuit([Z(i) for i in range(4)]) * GraphState.new_plus_state(4)

    assert g @ gb == 0

@pytest.mark.slow
def test_overlap_j_2():
    gb = GraphState.new_plus_state(4)

    g = S(0) * GraphState.new_plus_state(4)

    assert g @ gb == pytest.approx((1 + 1j) / 2)

