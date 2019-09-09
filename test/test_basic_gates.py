import numpy as np
import pytest

from pyqcs.gates.implementations.basic_gates import BasicGate

def test_raw_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    gate = BasicGate('X', 0, 0, 0.0)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0 
    assert qm_state_new[1] == 1
    assert qm_state_new[2] == 0 
    assert qm_state_new[3] == 0 

def test_raw_x2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    gate = BasicGate('X', 1, 0, 0.0)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0 
    assert qm_state_new[1] == 0
    assert qm_state_new[2] == 1 
    assert qm_state_new[3] == 0

def test_raw_h1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    gate = BasicGate('H', 0, 0, 0.0)
    M_SQRT1_2 = 0.70710678118654752440

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)


    assert pytest.approx(qm_state_new) == [M_SQRT1_2, M_SQRT1_2, 0, 0]


def test_raw_h2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    gate = BasicGate('H', 1, 0, 0.0)
    M_SQRT1_2 = 0.70710678118654752440

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [M_SQRT1_2, 0, M_SQRT1_2, 0]

def test_raw_h1_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    M_SQRT1_2 = 0.70710678118654752440
    gate = BasicGate('X', 0, 0, 0.0)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('H', 0, 0, 0.0)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state_new)

    print(qm_state_new)
    assert pytest.approx(qm_state_new) == [M_SQRT1_2, -M_SQRT1_2, 0, 0]

def test_raw_h2_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = np.zeros(nbits, dtype=np.double)
    M_SQRT1_2 = 0.70710678118654752440
    gate = BasicGate('X', 0, 0, 0.0)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('H', 1, 0, 0.0)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state_new)

    print(qm_state_new)
    assert pytest.approx(qm_state_new) == [0, M_SQRT1_2, 0, M_SQRT1_2]

