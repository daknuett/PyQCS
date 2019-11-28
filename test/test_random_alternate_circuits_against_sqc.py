import numpy as np

import pytest

import sqc
from pyqcs.util.random_circuits import random_basis_values

from pyqcs import Z, H, R, CZ, list_to_circuit, State, sample

def get_circuits(random_basis_values, nqbits):

    def _get_circuits_pyqcs():
        for i, k, x in zip(*random_basis_values):
            if(i <= nqbits):
                yield Z(i - 1)
            elif(i <= 2*nqbits):
                yield H(i - nqbits - 1)
            elif(i <= 3*nqbits):
                yield R(i - 2*nqbits - 1, x)

            else:
                if(k <= i - 3*nqbits - 1):
                    k -= 1
                yield CZ(i - 3*nqbits - 1, k)

    def _get_circuits_sqc():
        o = sqc.operator(nqbits)
        for i, k, x in zip(*random_basis_values):
            if(i <= nqbits):
                o = o.H(i - 1).X(i - 1).H(i - 1)
            elif(i <= 2*nqbits):
                o = o.H(i - nqbits - 1)
            elif(i <= 3*nqbits):
                o = o.Rz(i - 2*nqbits - 1, x)

            else:
                if(k <= i - 3*nqbits - 1):
                    k -= 1
                o = o.H(i - 3*nqbits - 1).CNOT(k, i - 3*nqbits - 1).H(i - 3*nqbits - 1)
        return o

    return (list_to_circuit(list(_get_circuits_pyqcs())), _get_circuits_sqc())


def do_test_q4_l10():
    qcs = State.new_zero_state(4)
    sq = sqc.state(4)
    random_values = random_basis_values(4, 10)
    qcs_circuit, sqc_circuit = get_circuits(random_values, 4)

    sq = sqc_circuit * sq
    qcs = qcs_circuit * qcs

    assert np.allclose(sq.v, qcs._qm_state)

def do_test_q10_l100():
    qcs = State.new_zero_state(10)
    sq = sqc.state(10)
    random_values = random_basis_values(10, 100)
    qcs_circuit, sqc_circuit = get_circuits(random_values, 10)

    sq = sqc_circuit * sq
    qcs = qcs_circuit * qcs

    assert np.allclose(sq.v, qcs._qm_state)

def test_random_q4_l10():
    for _ in range(1000):
        do_test_q4_l10()

@pytest.mark.slow
def test_random_q10_l100():
    for _ in range(4000):
        do_test_q10_l100()


def do_test_q4_l10_sample_half():
    qcs = State.new_zero_state(4)
    sq = sqc.state(4)
    random_values = random_basis_values(4, 10)
    qcs_circuit, sqc_circuit = get_circuits(random_values, 4)

    sq = sqc_circuit * sq
    qcs = qcs_circuit * qcs

    np.random.seed(0xdeadbeef)
    result_sqc = sqc.sample(sq, 10, [0, 1])
    np.random.seed(0xdeadbeef)
    result_pyqcs = sample(qcs, 0b11, 10)

    assert result_sqc == result_pyqcs

@pytest.mark.slow
def test_random_q10_l100_sample_half():
    for i in range(1000):
        do_test_q10_l100_sample_half()

def do_test_q10_l100_sample_half():
    qcs = State.new_zero_state(10)
    sq = sqc.state(10)
    random_values = random_basis_values(10, 100)
    qcs_circuit, sqc_circuit = get_circuits(random_values, 10)

    sq = sqc_circuit * sq
    qcs = qcs_circuit * qcs

    np.random.seed(0xdeadbeef)
    result_sqc = sqc.sample(sq, 10, list(range(5)))
    np.random.seed(0xdeadbeef)
    result_pyqcs = sample(qcs, 0b11111, 10)

    assert result_sqc == result_pyqcs

@pytest.mark.slow
def test_random_q4_l10_sample_half():
    for i in range(100):
        do_test_q4_l10_sample_half()
