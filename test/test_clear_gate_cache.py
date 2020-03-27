from pyqcs import H, S, clear_builtin_gate_cache
from pyqcs.gates.builtins import _H, _S


def test_cache_is_active():
    gates_S = [S(i) for i in range(10)]
    gates_H = [H(i) for i in range(20)]

    for s in gates_S:
        assert s in _S._registry.values()

    for h in gates_H:
        assert h in _H._registry.values()

def test_clear_cache():
    gates_S = [S(i) for i in range(10)]
    gates_H = [H(i) for i in range(20)]

    clear_builtin_gate_cache()

    assert _S._registry == {}
    assert _H._registry == {}
