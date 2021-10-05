from pyqcs import State, S, Z, X, H, CX, CZ, R
import numpy as np


def test_SS_Z():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (S(0) | S(0)) * s
    s2 = Z(0) * s

    assert s1 == s2


def test_ZZ_I():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (Z(0) | Z(0)) * s
    s2 = s

    assert s1 == s2


def test_XX_I():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (X(0) | X(0)) * s
    s2 = s

    assert s1 == s2


def test_HH_I():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (H(0) | H(0)) * s
    s2 = s

    assert s1 == s2


def test_HXH_Z():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (H(0) | X(0) | H(0)) * s
    s2 = Z(0) * s

    assert s1 == s2


def test_HZH_X():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = (H(0) | Z(0) | H(0)) * s
    s2 = X(0) * s

    assert s1 == s2


def test_HCZH_CX():
    s = State.new_zero_state(2)
    s.randomize()

    s1 = (H(0) | CZ(0, 1) | H(0)) * s
    s2 = CX(0, 1) * s

    assert s1 == s2


def test_HCXH_CZ():
    s = State.new_zero_state(2)
    s.randomize()

    s1 = (H(0) | CX(0, 1) | H(0)) * s
    s2 = CZ(0, 1) * s

    assert s1 == s2


def test_CZCZ_I():
    s = State.new_zero_state(2)
    s.randomize()

    s1 = (CZ(0, 1) | CZ(0, 1)) * s
    s2 = s

    assert s1 == s2


def test_CXCX_I():
    s = State.new_zero_state(2)
    s.randomize()

    s1 = (CX(0, 1) | CX(0, 1)) * s
    s2 = s

    assert s1 == s2


def test_CZ_symmetry():
    s = State.new_zero_state(2)
    s.randomize()

    s1 = CZ(0, 1) * s
    s2 = CZ(1, 0) * s

    assert s1 == s2


def test_R_Z():
    s = State.new_zero_state(1)
    s.randomize()

    s1 = R(0, np.pi) * s
    s2 = Z(0) * s

    assert s1 == s2