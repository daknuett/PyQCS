import pytest

import numpy as np
from itertools import product

from pyqcs import State, X, H, R, CX
from pyqcs.measurement import py_compute_amplitude
from pyqcs.gates.implementations.compute_amplitude import compute_amplitude

def test_compute_phase():
    state = State.new_zero_state(5)
    state = (X(1) | H(0) | CX(2, 0) | H(3) | X(4)) * state
    single_qbit_outcomes = [0, 1]

    for outcome in product(*[single_qbit_outcomes]*3):
        result = py_compute_amplitude(state, [0, 2, 3], outcome)
        result_c = compute_amplitude(state._qm_state
                                    , np.array([0, 2, 3], dtype=int)
                                    , np.array(outcome, dtype=np.uint8))

    assert np.allclose(result, result_c)
