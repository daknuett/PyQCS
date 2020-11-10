from .abc import AbstractGraphState
from .backend.raw_state import RawGraphState
from .util import graph_lists_to_naive_state
from.gate import GraphGate

class GraphState(AbstractGraphState):
    __slots__ = ["_g_state", "_nbits", "_measured", "_force_new_state"]
    def __init__(self, g_state, nbits, force_new_state=False, measured=None):
        AbstractGraphState.__init__(self)
        self._is_graph = True
        self._g_state = g_state
        self._nbits = nbits
        self._force_new_state = force_new_state
        self._measured = measured
        if(measured is None):
            self._measured = dict()


    @classmethod
    def new_plus_state(cls, nbits, **kwargs):
        if(nbits <= 0):
            raise ValueError("nbits must be greater than 0")

        g_state = RawGraphState(nbits)

        return cls(g_state, nbits, **kwargs)


    def deepcopy(self, **kwargs):
        key_word_arguments = {"force_new_state": self._force_new_state
                            , "measured": self._measured}
        key_word_arguments.update(kwargs)
        return GraphState(self._g_state.deepcopy(), self._nbits, **key_word_arguments)

    def to_naive_state(self):
        return graph_lists_to_naive_state(self._g_state.to_lists())

    def __str__(self):
        return str(self.to_naive_state())

    def check_qbits(self, gate_circuit):
        if(gate_circuit._uses_qbits < (1 << self._nbits)):
            return True
        return False

    @classmethod
    def new_zero_state(cls, nbits, **kwargs):
        if(nbits <= 0):
            raise ValueError("nbits must be greater than 0")

        g_state = RawGraphState(nbits)
        for i in range(nbits):
            g_state.apply_C_L(i, 0)

        return cls(g_state, nbits, **kwargs)

    def apply_gate(self, gate, force_new_state=False):
        if(not isinstance(gate, GraphGate)):
            raise TypeError("gate must be a GraphGate but got {}".format(str(type(gate))))

        state = self
        if(force_new_state or self._force_new_state):
            state = self.deepcopy()

        for op in gate._operation_list:
            has_measured, result = op.apply_to_raw_state(state._g_state)
            if(has_measured):
                state._measured[result[0]] = result[1]

        return state

    def is_normalized(self):
        return True

    def __matmul__(self, other):
        if(not isinstance(other, GraphState)):
            raise TypeError()
        if(not self._nbits == other._nbits):
            raise ValueError("cannot compute overlap of states with different qbit count")
        cmp_state = other._g_state.deepcopy()
        return cmp_state.mul_to(self._g_state)

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
