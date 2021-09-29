from ..graph.state import GraphState
from .. import CZ


def graph_lists_to_graph_state(lists, **kwargs):
    vops, entanglements = lists

    state = GraphState.new_plus_state(len(vops), **kwargs)

    handled_enganglements = set()

    for i, neighbors in enumerate(entanglements):
        for j in neighbors:
            edge = tuple(sorted((i, j)))
            if(edge not in handled_enganglements):
                state = CZ(*edge) * state
                handled_enganglements |= {edge}

    for i, vop in enumerate(vops):
        state._g_state.apply_C_L(i, vop)

    return state
