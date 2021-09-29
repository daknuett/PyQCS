import pytest

from pyqcs import State, H, X, Z, CZ, list_to_circuit
from pyqcs.graph.rawstate import RawGraphState
from pyqcs.graph.util import graph_lists_to_naive_state

VOP_H = 0
VOP_Z = 5
VOP_X = 14

@pytest.fixture
def graph_zero_state_10():
    g = RawGraphState(10)
    for i in range(10):
        g.apply_C_L(i, VOP_H)

    return g

@pytest.fixture
def naive_zero_state_10():
    return State.new_zero_state(10)

@pytest.fixture
def graph_plus_state_10():
    return RawGraphState(10)

@pytest.fixture
def naive_plus_state_10():
    state = State.new_zero_state(10)
    circuit = list_to_circuit([H(i) for i in range(10)])

    return circuit * state

def test_CZ_plus_state(graph_plus_state_10):
    g = graph_plus_state_10
    g.apply_CZ(0, 1)

    lists = g.to_lists()

    print(lists)
    assert lists == ([2] * 10, [[1], [0]] + [[]]*8)


def test_many_CZ_plus_state(graph_plus_state_10, naive_plus_state_10):
    edges = [(1, 2), (2, 0), (5, 9), (8, 4), (7, 0), (5, 7), (8, 9)]
    circuit = list_to_circuit([CZ(*e) for e in edges])
    g = graph_plus_state_10
    for e in edges:
        g.apply_CZ(*e)

    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == circuit * naive_plus_state_10

def test_many_CZ_plus_state(graph_plus_state_10, naive_plus_state_10):
    edges = [(1, 2), (2, 0), (5, 9), (8, 4), (7, 0), (5, 7), (8, 9)]
    circuit = list_to_circuit([CZ(*e) for e in edges])
    g = graph_plus_state_10
    for e in edges:
        g.apply_CZ(*e)

    converted = graph_lists_to_naive_state(g.to_lists())

    assert converted == circuit * naive_plus_state_10


def test_few_CZ_clear_vop1(graph_zero_state_10, naive_zero_state_10):
    edges = [(1, 2)
            , (2, 0)]
    circuit1 = list_to_circuit([CZ(*e) for e in edges])
    state1 = circuit1 * naive_zero_state_10
    g = graph_zero_state_10
    for e in edges:
        g.apply_CZ(*e)

    assert graph_lists_to_naive_state(g.to_lists()) == state1

def test_few_CZ_clear_vop2(graph_zero_state_10, naive_zero_state_10):
    edges = [(1, 2)
            ]
    circuit1 = list_to_circuit([CZ(*e) for e in edges])
    state1 = circuit1 * naive_zero_state_10
    g = graph_zero_state_10
    for e in edges:
        g.apply_CZ(*e)

    g.apply_C_L(2, VOP_X)
    g.apply_CZ(2, 0)

    state2 = (X(2) | CZ(2, 0)) * state1

    print(g.to_lists())
    assert graph_lists_to_naive_state(g.to_lists()) == state2

def test_few_CZ_clear_vop3(graph_zero_state_10, naive_zero_state_10):
    edges = [(1, 2)
            ]
    circuit1 = list_to_circuit([CZ(*e) for e in edges])
    state1 = circuit1 * naive_zero_state_10
    g = graph_zero_state_10
    for e in edges:
        g.apply_CZ(*e)

    g.apply_C_L(2, VOP_X)
    g.apply_C_L(0, VOP_X)
    g.apply_CZ(2, 0)

    state2 = (X(2) | X(0) | CZ(2, 0)) * state1

    assert graph_lists_to_naive_state(g.to_lists()) == state2

def test_few_CZ_clear_vops(graph_zero_state_10, naive_zero_state_10):
    edges = [(1, 2)
            , (2, 0)
           ]
    circuit1 = list_to_circuit([CZ(*e) for e in edges])
    state1 = circuit1 * naive_zero_state_10
    g = graph_zero_state_10
    for e in edges:
        g.apply_CZ(*e)

    g.apply_C_L(2, VOP_X)
    g.apply_CZ(2, 4)

    state2 = (X(2) | CZ(2, 4)) * state1

    assert graph_lists_to_naive_state(g.to_lists()) == state2

def test_many_CZ_clear_vops(graph_zero_state_10, naive_zero_state_10):
    edges = [(1, 2)
            , (2, 0)
            , (5, 9)
            , (8, 4)
            , (7, 0)
            , (5, 7)
            , (8, 9)
            , (1, 4)
            , (2, 5)
            , (4, 5)]
    circuit1 = list_to_circuit([CZ(*e) for e in edges])
    state1 = circuit1 * naive_zero_state_10
    g = graph_zero_state_10
    for e in edges:
        g.apply_CZ(*e)

    g.apply_C_L(2, VOP_X)
    g.apply_C_L(4, VOP_H)
    state2 = (X(2) | H(4)) * state1
    g.apply_CZ(2, 4)
    state3 = CZ(2, 4) * state2

    print(g.to_lists())
    print("converted", graph_lists_to_naive_state(g.to_lists()))
    print("state", state3)
    assert graph_lists_to_naive_state(g.to_lists()) == state3



def test_many_CZ_clear_vops_precomputed():
    g = RawGraphState(3)
    s = State.new_zero_state(3)
    s = (H(0) | H(1) | H(2)) * s

    s = (CZ(1, 0) | CZ(2, 0) | CZ(1, 2)) * s
    g.apply_CZ(1, 0)
    g.apply_CZ(2, 0)
    g.apply_CZ(1, 2)

    assert graph_lists_to_naive_state(g.to_lists()) == s

    s = (H(0) | X(1)) * s
    g.apply_C_L(0, VOP_H)
    g.apply_C_L(1, VOP_X)

    assert graph_lists_to_naive_state(g.to_lists()) == s

    s = CZ(1, 0) * s
    g.apply_CZ(1, 0)

    assert g.to_lists() == ([2, 2, 20], [[2], [2], [0, 1]])
    assert graph_lists_to_naive_state(g.to_lists()) == s

def test_3qbit_CZ():
    s = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
    s = (CZ(0, 1) | CZ(0, 2)) * s

    g = RawGraphState(3)
    g.apply_CZ(0, 1)
    g.apply_CZ(0, 2)

    assert graph_lists_to_naive_state(g.to_lists()) == s

