import numpy as np
from pyqcs import State, H, sample
from sqc import state, operator

def test_collaps_H4():
    for _ in range(4):
        s = (H(0) | H(1) | H(2) | H(3)) * State.new_zero_state(4)
        results = sample(s, 0b1, 10, keep_states=True)
        results = {i: v for v,i in results.keys()}

        s = operator(4).H(0).H(1).H(2).H(3) * state(4)
        rs, r = s.measure(0)

        assert np.allclose(rs.v, results[r]._qm_state)
