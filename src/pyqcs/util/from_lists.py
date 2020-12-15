from ..graph.state import GraphState
from .. import CZ

# FIXME:
# This is really bad and will yield inconsistent and
# wrong results. Change the graph lists API in future
# versions. This is a high priority issue.
default_phases = [0]*24
default_phases[0] = -2
default_phases[1] = 1
default_phases[2] = 4
default_phases[5] = 2
default_phases[14] = 2
default_phases[8] = -1

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
        state._g_state.apply_C_L(i, vop, default_phases[vop])

    return state

