import pytest
import numpy as np

try:
    import sqc
except:
    pytestmark = pytest.mark.skip("sqc is not present; install it to run these tests.")

from pyqcs import CX, H, R, S, X, State

def test_S():
    nbits = 4
    s_p = State.new_zero_state(nbits)
    s_s = sqc.state(nbits)
    op = sqc.operator(nbits)
    op = op.H(2).Rz(2, np.pi/2)

    s_s = op * s_s
    s_p = (H(2) | S(2)) * s_p

    assert np.allclose(s_s.v, s_p._qm_state)
