from numpy.random import choice

from itertools import product

from .from_lists import graph_lists_to_graph_state

def random_twoclique_graph(nother, n0):
    if(n0 > nother):
        raise ValueError("n0 must be <= nother")
    V_bar = [i + 2 for i in range(nother)]
    P_0 = V_bar[:n0]
    P_1 = V_bar[n0:]
    A = [0]
    B = [1]

    E = set()

    for i in P_0:
        a = choice(A)
        E |= {frozenset((i, a))}
        A.append(i)

    for i in P_1:
        b = choice(B)
        E |= {frozenset((i, b))}
        B.append(i)

    V = set(V_bar) | {0, 1}

    return V, E

def random_nobipartite_graph(nother, n0, naddedges):
    V, E = random_twoclique_graph(nother, n0)
    E_m = {frozenset((i,j)) for i,j in product(V, V) if i != j} - E
    if(naddedges < 0
            or naddedges > len(E_m)):
        raise ValueError(f"naddedges must be > 0 and < {len(E_m)}")

    N = set(choice(list(E_m), size=naddedges))

    return V, E | N


def random_graph_lists(nqbits, naddedges):
    if(nqbits < 4):
        raise ValueError("at least 4 qbits are required for a meaningful random graph")
    n0 = choice(nqbits - 2)
    vops = choice(24, size=nqbits)

    V, E = random_nobipartite_graph(nqbits - 2, n0, naddedges)

    ngbhds = []
    for v in V:
        ngbhds.append(set())
        for i,j in E:
            if(i == v):
                ngbhds[v] |= {j}
            if(j == v):
                ngbhds[v] |= {i}

    return vops, [list(sorted(n)) for n in ngbhds]

def random_graph_state(nqbits, naddedges):
    return graph_lists_to_graph_state(random_graph_lists(nqbits, naddedges))
