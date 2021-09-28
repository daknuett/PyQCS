import pytest
from pyqcs import X, H, State

def test_project_to_z1():
    s = State.new_zero_state(2)

    sb = s.project_Z(0, 0)

    assert sb @ s == pytest.approx(1)

def test_project_to_z2():
    s = State.new_zero_state(2)

    sb = s.project_Z(0, 1)

    assert sb == 0

def test_project_to_z3():
    s = H(0) * State.new_zero_state(2)

    sb = s.project_Z(0, 0)
    print(s)
    print(sb)

    assert sb @ (H(0) * s) == pytest.approx(1)

def test_project_to_z4():
    s = H(0) * State.new_zero_state(2)

    sb = s.project_Z(0, 1)

    assert sb @ ((H(0) | X(0)) * s) == pytest.approx(1)
