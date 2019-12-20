from numpy import array

from ..gates.builtins import H, CZ, GenericGate
from ..state.state import BasicState

C_L = [array([[ 0.70710678+0.j,  0.70710678+0.j], [ 0.70710678+0.j, -0.70710678+0.j]])
    , array([[1.+0.j, 0.+0.j], [0.+0.j, 0.+1.j]])
    , array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])
    , array([[0.70710678+0.j, 0.70710678+0.j], [0.+0.70710678j, 0.-0.70710678j]])
    , array([[0.70710678+0.j, 0.+0.70710678j], [0.70710678+0.j, 0.-0.70710678j]])
    , array([[ 1.+0.j,  0.+0.j], [ 0.+0.j, -1.+0.j]])
    , array([[0.70710678+0.j, 0.+0.70710678j], [0.+0.70710678j, 0.70710678+0.j]])
    , array([[ 0.70710678+0.j, -0.70710678+0.j], [ 0.70710678+0.j,  0.70710678+0.j]])
    , array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]])
    , array([[ 0.70710678+0.j, -0.70710678+0.j], [ 0.+0.70710678j,  0.+0.70710678j]])
    , array([[0.70710678+0.j, 0.-0.70710678j], [0.70710678+0.j, 0.+0.70710678j]])
    , array([[ 0.70710678+0.j,  0.-0.70710678j], [ 0.+0.70710678j, -0.70710678+0.j]])
    , array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]])
    , array([[ 0.70710678+0.j,  0.70710678+0.j], [-0.70710678+0.j,  0.70710678+0.j]])
    , array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
    , array([[0.70710678+0.j, 0.70710678+0.j], [0.-0.70710678j, 0.+0.70710678j]])
    , array([[0.+0.j, 1.+0.j], [0.+1.j, 0.+0.j]])
    , array([[ 0.5-0.5j,  0.5+0.5j], [-0.5+0.5j,  0.5+0.5j]])
    , array([[0.+0.j, 0.+1.j], [1.+0.j, 0.+0.j]])
    , array([[ 0.70710678+0.j,  0.+0.70710678j], [ 0.-0.70710678j, -0.70710678+0.j]])
    , array([[ 0.5-0.5j, -0.5+0.5j], [-0.5+0.5j, -0.5+0.5j]])
    , array([[ 0.+0.j, -1.+0.j], [ 1.+0.j,  0.+0.j]])
    , array([[ 0.70710678+0.j, -0.70710678+0.j], [ 0.-0.70710678j,  0.-0.70710678j]])
    , array([[ 0.5-0.5j, -0.5-0.5j], [-0.5+0.5j, -0.5-0.5j]])
]


def graph_lists_to_naive_state(lists):
    vops, entanglements = lists

    state = BasicState.new_zero_state(len(vops))

    for i in range(len(vops)):
        state = H(i) * state


    handled_enganglements = set()

    for i, neighbors in enumerate(entanglements):
        for j in neighbors:
            edge = tuple(sorted((i, j)))
            if(edge not in handled_enganglements):
                state = CZ(*edge) * state
                handled_enganglements |= {edge}

    for i, vop in enumerate(vops):
        gate = GenericGate(i, C_L[vop])
        state = gate * state

    return state

