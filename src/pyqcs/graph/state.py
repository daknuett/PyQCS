import numpy.random

from .rawstate import RawGraphState
from .util import graph_lists_to_naive_state
from ..gates.circuits import Circuit
from ..gates.gate import Capabilities

_local_clifford_gates = ["H", "S", "X", "Z"]
_CZ_gates = ["CZ"]
_decomposable_gates = ["CX"]
_measurement_gates = ["M"]

_local_clifford_gate_vop = {"H": 0, "S": 1, "X": 14, "Z": 5}
_gate_decompositions = {"CX": (("H", 0), ("CZ", 0, 1), ("H", 0))}


class DummyGate(object):
    def __init__(self, name, *args):
        self._name = name
        self._act = args[0]
        self._control = None
        if(len(args) > 1):
            self._control = args[1]


class GraphState(object):
    __slots__ = ["_g_state", "_nqbits", "_measured", "_copy", "_rne"]
    _has_capabilities = Capabilities.clifford()

    def __init__(self, g_state, nbits, copy=False, measured=None, rne=None):
        self._g_state = g_state
        self._nqbits = nbits
        self._copy = copy
        self._measured = measured

        if(rne is None):
            self._rne = lambda : numpy.random.uniform(0, 1)
        else:
            self._rne = rne
        if(measured is None):
            self._measured = dict()

    @classmethod
    def new_plus_state(cls, nbits, **kwargs):
        if(nbits <= 0):
            raise ValueError("nbits must be greater than 0")

        g_state = RawGraphState(nbits)

        return cls(g_state, nbits, **kwargs)

    def deepcopy(self, **kwargs):
        key_word_arguments = {"copy": self._copy
                            , "measured": self._measured
                            , "rne": self._rne}
        key_word_arguments.update(kwargs)
        return type(self)(self._g_state.deepcopy(), self._nqbits, **key_word_arguments)

    def to_naive_state(self):
        # FIXME: use the backend method for that.
        return graph_lists_to_naive_state(self._g_state.to_lists())

    def __str__(self):
        return str(self.to_naive_state())

    def check_capabilities(self, circuit):
        return circuit._requires_capabilities <= type(self)._has_capabilities

    def check_qbits(self, circuit):
        return circuit._requires_qbits < (1 << self._nqbits)

    @classmethod
    def new_zero_state(cls, nbits, **kwargs):
        if(nbits <= 0):
            raise ValueError("nbits must be greater than 0")

        g_state = RawGraphState(nbits)
        for i in range(nbits):
            g_state.apply_C_L(i, 0)

        return cls(g_state, nbits, **kwargs)

    def __rmul__(self, other):
        if(not isinstance(other, Circuit)):
            raise TypeError()
        if(not self.check_capabilities(other)):
            raise ValueError(f"capabilities insuffient (required:"
                            f"{other._requires_capabilities},"
                            f" got: {self._has_capabilities})")
        if(not self.check_qbits(other)):
            raise ValueError(f"insufficient qbits (required: "
                            f"{other._requires_qbits.bit_length()}"
                            f", got:{self._nqbits})")

        state = self
        if(self._copy):
            state = self.deepcopy()

        for gate in other._gate_list:
            state.__apply_gate(gate)

        return state

    def __apply_gate(self, gate):
        if(gate._name in _local_clifford_gates):
            self._g_state.apply_C_L(gate._act, _local_clifford_gate_vop[gate._name])
            return
        if(gate._name in _CZ_gates):
            self._g_state.apply_CZ(gate._act, gate._control)
            return
        if(gate._name in _measurement_gates):
            result = self._g_state.measure(gate._act, self._rne())
            self._measured[gate._act] = result
            return
        if(gate._name in _decomposable_gates):
            args = (gate._act, gate._control)
            for word in _gate_decompositions[gate._name]:
                if(len(word) == 2):
                    dummy = DummyGate(word[0], args[word[1]])
                else:
                    dummy = DummyGate(word[0], args[word[1]], args[word[2]])
                self.__apply_gate(dummy)

    def is_normalized(self):
        return True

    def __matmul__(self, other):
        if(not isinstance(other, GraphState)):
            raise TypeError()
        if(not self._nqbits == other._nqbits):
            raise ValueError("cannot compute overlap of states with different qbit count")
        return other._g_state.mul_to(self._g_state)

    def project_to(self, qbit, observable):
        """
        Apply the projection operator of ``observable`` to ``qbit``.
        ``observable`` is a string in ``["Z", "Y", "X", "-Z", "-Y", "-X"]``.
        """
        observables = ["Z", "Y", "X", "-Z", "-Y", "-X"]
        if(observable.upper() not in observables):
            raise ValueError(f"unknown observable, must be one of {observables}")

        observable = observables.index(observable.upper())

        return self._g_state.project_to(qbit, observable)

    def redo_normalization(self):
        pass
