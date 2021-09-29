import pytest

from pyqcs.gates.builtins import X, H
from pyqcs.state.state import DSVState


def test_XH():
    M_SQRT1_2 = 0.70710678118654752440
    state = DSVState.new_zero_state(2)

    state = (X(0) | H(0)) * state

    assert state.export_numpy() == pytest.approx([M_SQRT1_2, -M_SQRT1_2, 0, 0])
