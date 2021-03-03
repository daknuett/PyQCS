import pytest

import numpy as np

from pyqcs import State, X, H, R, CX
from pyqcs.measurement import compute_amplitude
from pyqcs.gates.implementations.compute_amplitude import compute_phase as compute_phase_c

def test_compute_phase():
    state = State.new_zero_state(5)
    state = (X(1) | H(0) | CX(2, 0) | H(3) | X(4)) * state

    result = compute_amplitude(state, [0, 2, 3], [1, 1, 1])
    result_c = compute_phase_c(state._qm_state, np.array([0, 2, 3], dtype=int), np.array([1, 1, 1], dtype=np.uint8))

    assert np.allclose(result, result_c)
