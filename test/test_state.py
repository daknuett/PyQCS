from pyqcs import State, X, M

def test_state_eq():
    s0 = State.new_zero_state(2)
    s1 = X(0) * s0
    s2 = M(0) * s1
    s3 = X(0) * s2
    s4 = X(0) * s1

    assert s0 != s1
    assert s0 != s2
    assert s1 != s2
    assert s3 != s4
    assert s4 == s0


def test_state_hash():
    s0 = State.new_zero_state(2)
    s1 = X(0) * s0
    s2 = M(0) * s1
    s3 = X(0) * s2
    s4 = X(0) * s1

    assert hash(s0) != hash(s1)
    assert hash(s0) != hash(s2)
    assert hash(s1) != hash(s2)
    assert hash(s3) != hash(s4)
    assert hash(s4) == hash(s0)


def test_state_insert_into_dict():
    s0 = State.new_zero_state(2)
    s1 = X(0) * s0
    d = dict()

    d[s0] = 0
    d[s1] = 1

    assert d == {s0: 0, s1: 1}
