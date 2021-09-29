import pytest
import numpy as np
from pyqcs import measure, sample, State, X, H


def test_measurement_deterministic00():
    state = State.new_zero_state(4)

    assert measure(state, 0b11)[1] == 0
    assert measure(state, 0b11)[0]._qm_state == pytest.approx(state._qm_state)
    assert measure(state, 0b11)[0]._cl_state == pytest.approx([0, 0, -1, -1])


def test_measurement_deterministic01():
    state = X(1) * State.new_zero_state(4)

    assert measure(state, 0b11)[1] == 2
    assert measure(state, 0b11)[0]._qm_state == pytest.approx(state._qm_state)
    assert measure(state, 0b11)[0]._cl_state == pytest.approx([0, 1, -1, -1])


def test_measure_hadamard():
    state = H(0) * State.new_zero_state(2)

    new_state, result = measure(state, 0b11)

    assert result in (0, 1)

    if(result == 0):
        assert new_state._qm_state == pytest.approx(State.new_zero_state(2)._qm_state)
        assert new_state._cl_state == pytest.approx([0, 0])

    if(result == 1):
        assert new_state._qm_state == pytest.approx((X(0) * State.new_zero_state(2))._qm_state)
        assert new_state._cl_state == pytest.approx([1, 0])


def test_sample_hadamard():
    np.random.seed(0xdeadbeef)
    state = H(0) * State.new_zero_state(2)

    result = sample(state, 0b11, 50)

    assert result == {1: 26, 0: 24}


def test_sample_hadamard_10qbit():
    np.random.seed(0xdeadbeef)
    state = H(0) * State.new_zero_state(10)

    result = sample(state, 0b1111111111, 50)

    assert result == {0: 26, 1: 24}
