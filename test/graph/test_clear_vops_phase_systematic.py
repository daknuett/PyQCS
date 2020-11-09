from itertools import product

import pytest

from pyqcs import CZ, H, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import vop_to_circuit

topologies = [
        [(0, 2)], [(0, 2), (0, 1)]
        , [(0, 2), (0, 3)] , [(0, 2), (0, 3), (0, 1)] , [(0, 2), (0, 3), (2, 3)]
        , [(0, 2), (1, 3), (0, 3)] , [(0, 2), (1, 3), (0, 3), (0, 1)]
        , [(0, 2), (0, 3)] , [(0, 2), (0, 3), (2, 3)]
        , [(0, 2), (0, 3), (2, 3), (1, 3)] , [(0, 2), (0, 3), (2, 3), (1, 3), (0, 1)]
        , [(0, 2), (1, 3), (0, 3), (1, 2)] , [(0, 2), (1, 3), (0, 3), (1, 2), (2, 3)] , [(0, 2), (1, 3), (0, 3), (1, 2), (2, 3), (0, 1)]
        ]

vops = list(range(24))

def topology_to_vopfree_states(topology):
    g = GraphState.new_plus_state(4)
    n = (H(0) | H(1) | H(2) | H(3)) * State.new_zero_state(4)
    for ent in topology:
        g = CZ(*ent) * g
        n = CZ(*ent) * n
    return g, n

@pytest.mark.selected
def test_clear_vops_systematically_ID_neighbours():
    for topology in topologies:
        for v1, v2 in product(vops, vops):
            g1 = vop_to_circuit(0, v1)
            g2 = vop_to_circuit(1, v2)
            g, n = topology_to_vopfree_states(topology)
            g = (g1 | g2) * g
            n = (g1 | g2) * n

            g = CZ(0, 1) * g
            n = CZ(0, 1) * n

            assert g.to_naive_state() @ n == pytest.approx(1)

@pytest.mark.skip
def test_clear_vops_systematically_non_ID_neighbours():
    for topology in topologies:
        for v1, v2, v3 in product(vops, vops, vops):
            g1 = vop_to_circuit(0, v1)
            g2 = vop_to_circuit(1, v2)
            g3 = vop_to_circuit(2, v3)
            g, n = topology_to_vopfree_states(topology)
            g = (g1 | g2) * g
            n = (g1 | g2) * n

            g = CZ(0, 1) * g
            n = CZ(0, 1) * n

            assert g.to_naive_state() @ n == pytest.approx(1)

