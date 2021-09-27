from ..utils import list_to_circuit
from ..gates.builtins import H, Z, X, S, CZ
from ..state.state import DSVState

decompositions = ['H',
                 'S',
                 'ZZ',   # For technical reasons we
                         # require a non-empty string here.
                 'SH',
                 'HS',
                 'Z',
                 'SHS',
                 'HZ',
                 'ZS',
                 'SHZ',
                 'SHSH',
                 'SHZS',
                 'HSH',
                 'ZH',
                 'X',
                 'ZSH',
                 'SX',
                 'ZHS',
                 'XS',
                 'ZSHS',
                 'ZHZ',
                 'XZ',
                 'XSH',
                 'ZSHSH'
 ]


def vop_factorization_circuit(i, vop):
    gates = {"H": H, "S": S, "X": X, "Z": Z}
    lst = [gates[word](i) for word in reversed(decompositions[vop])]
    return list_to_circuit(lst)


def graph_lists_to_naive_state(lists):
    vops, entanglements = lists

    state = DSVState.new_zero_state(len(vops))

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
        circuit = vop_factorization_circuit(i, vop)
        state = circuit * state

    return state
