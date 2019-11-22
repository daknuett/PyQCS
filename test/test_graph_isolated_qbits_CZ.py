import pytest
from itertools import product

from pyqcs import State, H, X, Z, CZ, list_to_circuit
from pyqcs.graph.backend.raw_state import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state, C_L
from pyqcs import GenericGate


def possible_input_configurations():
    for c0, c1, e in product(range(24), range(24), (True, False)):
        g0 = GenericGate(0, C_L[c0])
        g1 = GenericGate(1, C_L[c1])

        yield(g0, g1, c0, c1, e)


@pytest.mark.slow
def test_two_qbits_isolated():
    for g0, g1, c0, c1 ,e in possible_input_configurations():
        g = RawGraphState(2)
        s = (H(0) | H(1)) * State.new_zero_state(2)
        if(e):
            g.apply_CZ(0, 1)
            s = CZ(0, 1) * s
        g.apply_C_L(0, c0)
        g.apply_C_L(1, c1)
        s = g0 * s
        s = g1 * s


        assert graph_lists_to_naive_state(g.to_lists()) == s

@pytest.mark.slow
def test_two_qbits_isolated_CZ():
    for g0, g1, c0, c1 ,e in possible_input_configurations():
        g = RawGraphState(2)
        s = (H(0) | H(1)) * State.new_zero_state(2)
        if(e):
            g.apply_CZ(0, 1)
            s = CZ(0, 1) * s
        g.apply_C_L(0, c0)
        g.apply_C_L(1, c1)
        s = g0 * s
        s = g1 * s

        g.apply_CZ(0, 1)
        s = CZ(0, 1) * s

        assert graph_lists_to_naive_state(g.to_lists()) == s
