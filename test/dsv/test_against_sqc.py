import pytest
import numpy as np

from pyqcs.gates.implementations.basic_gates import BasicGate

try:
    import sqc
except:
    pytestmark = pytest.mark.skip("sqc is not present; install it to run these tests.")

def nop():
    return 0.0

def test_raw_x1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 0, 0, 0.0, nop)
    state = sqc.operator(2).X(0) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)


def test_raw_x2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 1, 0, 0.0, nop)
    state = sqc.operator(2).X(1) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)

def test_raw_h1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('H', 0, 0, 0.0, nop)
    state = sqc.operator(2).H(0) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)

def test_raw_h2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('H', 1, 0, 0.0, nop)
    state = sqc.operator(2).H(1) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)


def test_raw_r1_0():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    state = sqc.state(2)

    tests = [(
                (sqc.operator(2).Rz(0, r) * state).v
                , BasicGate('R', 0, 0, r, nop)(qm_state, cl_state)[0])
                    for r in np.arange(0, 2 * np.pi, 0.1)]

    for expect, got in tests:
        assert expect == pytest.approx(got)

def test_raw_r2_0():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    state = sqc.state(2)

    tests = [(
                (sqc.operator(2).Rz(1, r) * state).v
                , BasicGate('R', 1, 0, r, nop)(qm_state, cl_state)[0])
                    for r in np.arange(0, 2 * np.pi, 0.1)]

    for expect, got in tests:
        assert expect == pytest.approx(got)
def test_raw_r1_1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[1] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    state = sqc.state(2)


    tests = [(
                (sqc.operator(2).X(0).Rz(0, r) * state).v
                , BasicGate('R', 0, 0, r, nop)(qm_state, cl_state)[0])
                    for r in np.arange(0, 2 * np.pi, 0.1)]

    for expect, got in tests:
        print("expect", expect)
        print("got", got)
        assert expect == pytest.approx(got)

def test_raw_r2_1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    state = sqc.state(2)

    qm_state, cl_state, measured = BasicGate('X', 1, 0, 0.0, nop)(qm_state, cl_state)

    tests = [(
                (sqc.operator(2).X(1).Rz(1, r) * state).v
                , BasicGate('R', 1, 0, r, nop)(qm_state, cl_state)[0])
                    for r in np.arange(0, 2 * np.pi, 0.1)]

    for expect, got in tests:
        assert expect == pytest.approx(got)

def test_raw_hx11():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    state = sqc.operator(2).X(0).H(0) * sqc.state(2)
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('H', 0, 0, 0.0, nop)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state_new)

    assert state.v == pytest.approx(qm_state_new)

def test_raw_cnot_0():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('C', 1, 0, 0.0, nop)
    state = sqc.operator(2).CNOT(1, 0) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)

def test_raw_cnot_1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('X', 0, 0, 0.0, nop)
    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)
    gate = BasicGate('C', 1, 0, 0.0, nop)
    state = sqc.operator(2).X(0).CNOT(0, 1) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state_new, cl_state)

    assert state.v == pytest.approx(qm_state_new)

def test_raw_z1():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('Z', 0, 0, 0.0, nop)
    state = sqc.operator(2).Rz(0, np.pi) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)


def test_raw_z2():
    nbits = 2
    ndim = 2**nbits
    qm_state = np.zeros(ndim, dtype=np.cdouble)
    qm_state[0] = 1
    cl_state = -1 * np.ones(nbits, dtype=np.int8)
    gate = BasicGate('Z', 1, 0, 0.0, nop)
    state = sqc.operator(2).Rz(1, np.pi) * sqc.state(2)

    qm_state_new, cl_state_new, measured = gate(qm_state, cl_state)

    assert state.v == pytest.approx(qm_state_new)
