import numpy as np
from pyqcs import H, CX, State, tree_amplitudes

def test_tree_amplitudes_plus_state():
    state = (H(0) | H(1) | H(2) | H(3)) * State.new_zero_state(4)

    outcome = tree_amplitudes(state)

    assert np.allclose(list(sorted(outcome, key=lambda x: x[0])), [[i, 0.5**4] for i in range(2**4)])

def test_tree_amplitudes_double_bell_state():
    state = (H(0) | CX(1, 0) | H(2) | CX(3, 2)) * State.new_zero_state(4)

    outcome = tree_amplitudes(state)

    assert np.allclose(outcome, [[0b0000, 0.2500000000000001],
                         [0b1100, 0.2500000000000001],
                         [0b0011, 0.2500000000000001],
                         [0b1111, 0.2500000000000001]])
