from .. import list_to_circuit
from ..gates.builtins import H, S, CZ, Z, X
from ..graph.state import GraphState
from ..graph.c_l import C_L


def graph_state_to_circuit(state):
    """
    This function basically converts a graph state to a circuit
    that (applied to a zero-state) will yield the same state.

    In particular for some graph state g::

        graph_state_to_circuit(g) * GraphState.new_zero_state(g._nqbits)

    will copy g.
    """

    if(not isinstance(state, GraphState)):
        raise TypeError(f"GraphState is required, but got {type(state)}")

    prepare = list_to_circuit([H(i) for i in range(state._nqbits)])

    vops, edges = state._g_state.to_lists()

    entanglements = []

    handled_edges = set()
    for i, ngbhd in enumerate(edges):
        for j in ngbhd:
            edge = tuple(sorted((i, j)))
            if(edge in handled_edges):
                continue
            entanglements.append(CZ(*edge))
            handled_edges |= {edge}

    entanglements = list_to_circuit(entanglements)

    vop_circuit = list_to_circuit([vop_to_circuit(i, vop) for i,vop in enumerate(vops)])

    if(entanglements):
        return prepare | entanglements | vop_circuit
    else:
        return prepare | vop_circuit


def vop_to_circuit(act, vop):
    s2m = {"H": H, "S": S, "X": X, "Z": Z}
    return list_to_circuit([s2m[c](act) for c in reversed(C_L[vop])])
