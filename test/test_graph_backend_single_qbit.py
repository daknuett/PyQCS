from pyqcs.graph.backend.raw_state import RawGraphState

def test_h0():
    state = RawGraphState(1)
    assert state.to_lists() == ([0], [[]])

    state.apply_C_L(0, 0)

    assert state.to_lists() == ([2], [[]])
