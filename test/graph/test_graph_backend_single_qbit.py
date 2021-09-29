from itertools import product
from pyqcs.graph.rawstate import RawGraphState
from pyqcs import State, H
from pyqcs.graph.util import graph_lists_to_naive_state, vop_factorization_circuit


def test_h0():
    state = RawGraphState(1)
    assert state.to_lists() == ([2], [[]])

    state.apply_C_L(0, 0)

    assert state.to_lists() == ([0], [[]])


def test_all_clifford_gates_single():
    for i in range(24):
        g = RawGraphState(1)
        s = (H(0)) * State.new_zero_state(1)

        g.apply_C_L(0, i)
        s = vop_factorization_circuit(0, i) * s

        assert graph_lists_to_naive_state(g.to_lists()) == s


def test_all_clifford_gates_two():
    for c0, c1 in product(range(24), range(24)):
        g = RawGraphState(1)
        s = (H(0)) * State.new_zero_state(1)

        g.apply_C_L(0, c0)
        s = vop_factorization_circuit(0, c0) * s

        g.apply_C_L(0, c1)
        s = vop_factorization_circuit(0, c1) * s

        assert graph_lists_to_naive_state(g.to_lists()) == s
