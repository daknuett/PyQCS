import numpy as np

from pyqcs.gates.implementations.basic_gates import BasicGate

def test_raw_x():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    gate = BasicGate('X', 0, 0, 0.0)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0 
    assert qm_state_new[1] == 1
