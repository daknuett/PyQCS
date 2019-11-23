import numpy
from ..state.abc import AbstractState
from .backend.raw_state import RawGraphState
from .util import graph_lists_to_naive_state

class GraphState(AbstractState):
    def __init__(self, g_state, nbits):
        AbstractState.__init__(self)
        self._is_graph = True
        self._g_state = g_state
        self._nbits = nbits

    @classmethod
    def new_plus_state(cls, nbits, **kwargs):
        if(nbits <= 0):
            raise ValueError("nbits must be greater than 0")

        g_state = RawGraphState(nbits)

        return cls(g_state, nbits, **kwargs)

    def get_last_measurement(self):
        # FIXME
        raise NotImplementedError("measurement is not yet implemented")
     
    def deepcopy(self):
        return GraphState(self._g_state.deepcopy(), self._nbits)

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
            g.apply_C_L(i, 0)

        return cls(g_state, nbits, **kwargs)

    def apply_gate(self, gate):
        raise NotImplementedError("todo")

