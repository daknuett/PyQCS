import pytest
import numpy as np
from itertools import product

from pyqcs import CZ, H, S, X, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_circuit import vop_to_circuit


vops = list(range(24))

@pytest.mark.slow
def test_g2n_no_CZ():
    for v0, v1, v2 in product(vops, vops, vops):
        ov0 = vop_to_circuit(0, v0)
        ov1 = vop_to_circuit(1, v1)
        ov2 = vop_to_circuit(2, v2)

        s = (H(0) | H(1) | H(2)) * State.new_zero_state(3)
        g = GraphState.new_plus_state(3)

        s = (ov0 | ov1 | ov2) * s
        g = (ov0 | ov1 | ov2) * g

        assert g._g_state.to_lists()[0] == [v0, v1, v2]

        assert g.to_naive_state() == s

@pytest.mark.slow
def test_g2n_CZ01():
    for v0, v1, v2 in product(vops, vops, vops):
        ov0 = vop_to_circuit(0, v0)
        ov1 = vop_to_circuit(1, v1)
        ov2 = vop_to_circuit(2, v2)

        s = (H(0) | H(1) | H(2) | CZ(0, 1)) * State.new_zero_state(3)
        g = CZ(0, 1) * GraphState.new_plus_state(3)

        s = (ov0 | ov1 | ov2) * s
        g = (ov0 | ov1 | ov2) * g

        assert g._g_state.to_lists()[0] == [v0, v1, v2]

        assert g.to_naive_state() == s

@pytest.mark.slow
def test_g2n_CZ0201():
    for v0, v1, v2 in product(vops, vops, vops):
        ov0 = vop_to_circuit(0, v0)
        ov1 = vop_to_circuit(1, v1)
        ov2 = vop_to_circuit(2, v2)

        s = (H(0) | H(1) | H(2) | CZ(0, 2) | CZ(0, 1)) * State.new_zero_state(3)
        g = (CZ(0, 2) | CZ(0, 1)) * GraphState.new_plus_state(3)

        s = (ov0 | ov1 | ov2) * s
        g = (ov0 | ov1 | ov2) * g

        assert g._g_state.to_lists()[0] == [v0, v1, v2]

        assert g.to_naive_state() == s

