import numpy as np
from numpy import random
import pytest

from pyqcs.gates.implementations.basic_gates import BasicGate

def nop():
    return 0.0

def test_raw_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 0, 0, 0.0, nop)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0
    assert qm_state_new[1] == 1
    assert qm_state_new[2] == 0
    assert qm_state_new[3] == 0
    assert measured == 0

def test_raw_x2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 1, 0, 0.0, nop)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0
    assert qm_state_new[1] == 0
    assert qm_state_new[2] == 1
    assert qm_state_new[3] == 0
    assert measured == 0

def test_raw_x12():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state, cl_state, measured = gate(qm_state, cl_state)

    gate = BasicGate('X', 1, 0, 0.0, nop)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0
    assert qm_state_new[1] == 0
    assert qm_state_new[2] == 0
    assert qm_state_new[3] == 1
    assert measured == 0

def test_raw_h1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('H', 0, 0, 0.0, nop)
    M_SQRT1_2 = 0.70710678118654752440

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)


    assert pytest.approx(qm_state_new) == [M_SQRT1_2, M_SQRT1_2, 0, 0]
    assert measured == 0


def test_raw_h2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('H', 1, 0, 0.0, nop)
    M_SQRT1_2 = 0.70710678118654752440

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [M_SQRT1_2, 0, M_SQRT1_2, 0]
    assert measured == 0

def test_raw_h1_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    M_SQRT1_2 = 0.70710678118654752440
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('H', 0, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state_new)

    assert pytest.approx(qm_state_new) == [M_SQRT1_2, -M_SQRT1_2, 0, 0]
    assert measured == 0

def test_raw_h2_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    M_SQRT1_2 = 0.70710678118654752440
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('H', 1, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state_new)

    assert pytest.approx(qm_state_new) == [0, M_SQRT1_2, 0, M_SQRT1_2]
    assert measured == 0


def test_raw_cnot_00():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('C', 1, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [1, 0, 0, 0]

def test_raw_cnot_01():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[1] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('C', 1, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [0, 0, 0, 1]

def test_raw_cnot_10():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[2] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('C', 1, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [0, 0, 1, 0]

def test_raw_cnot_11():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[3] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('C', 1, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert pytest.approx(qm_state_new) == [0, 1, 0, 0]

def test_deterministic_measurement0():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('M', 0, 0, 0, np.random.uniform)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new == pytest.approx(qm_state)
    assert cl_state_new == pytest.approx([0, -1])
    assert measured == 1

def test_deterministic_measurement1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[1] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('M', 0, 0, 0, np.random.uniform)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new == pytest.approx(qm_state)
    assert cl_state_new == pytest.approx([1, -1])
    assert measured == 1

def test_non_deterministic_measurement_01():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('H', 0, 0, 0, nop)
    qm_state, _, measured = gate(qm_state, cl_state)
    gate = BasicGate('M', 0, 0, 0, np.random.uniform)

    qm_state_new, cl_state, measured = gate(qm_state, cl_state)

    if(cl_state[0] == 0):
        assert cl_state == pytest.approx([0, -1])
        assert qm_state_new == pytest.approx([1, 0, 0, 0])
    else:
        assert cl_state == pytest.approx([1, -1])
        assert qm_state_new == pytest.approx([0, 1, 0, 0])
    assert measured == 1

def test_raw_z1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('Z', 0, 0, 0.0, nop)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 1
    assert qm_state_new[1] == 0
    assert qm_state_new[2] == 0
    assert qm_state_new[3] == 0
    assert measured == 0

def test_raw_xz1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state, cl_state, measured = gate(qm_state, cl_state)
    gate = BasicGate('Z', 0, 0, 0.0, nop)


    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert qm_state_new[0] == 0
    assert qm_state_new[1] == -1
    assert qm_state_new[2] == 0
    assert qm_state_new[3] == 0
    assert measured == 0
