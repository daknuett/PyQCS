from .abc import AbstractState
import numpy as np

class BasicState(AbstractState):
    __slots__ = ["_nbits", "_ndim", "_qm_state", "_cl_state", "_last_measured"]
    def __init__(self, qm_state, cl_state, nbits, last_measured, **kwargs):
        self._qm_state = qm_state
        self._cl_state = cl_state
        self._nbits = nbits
        self._ndim = 2**nbits
        self._last_measured = last_measured

    @classmethod
    def new_zero_state(cls, nbits, **kwargs):
        if(nbits > 29):
            # (2**29 * 16) / 2**30 = 8 GiB.
            raise Warning("State will require more than 8GiB RAM. " 
                        "Using such big states can lead to random crashes.")
        ndim = 2**nbits
        qm_state = np.array([1] + ([0] * ndim - 1), dtype=np.cfloat)
        last_measured = 0
        cl_state = np.zeros(nbits)

        return cls(qm_state, cl_state, nbits, last_measured, **kwargs)

    def get_last_measurement(self):
        return self._cl_state, self._last_measured
    def check_qbits(self, gate_circuit):
        if(gate_circuit._uses_qbits < (1 << self._nbits + 1)):
            return True
        return False

    def deepcopy(self):
        return BasicState(self._qm_state.copy(), self._cl_state.copy(), self._nbits, self._last_measured)
    def apply_gate(self, gate, force_new_state=False):
        if(gate.is_inplace() and force_new_state):
            qm_state, cl_state, last_measured = gate(self._qm_state.copy(), self._cl_state.copy())
            return BasicState(qm_state, cl_state, self._nbits,  last_measured)
        if(gate.is_inplace()):
            gate(self._qm_state, self._cl_state)
            return self
        qm_state, cl_state, last_measured = gate(self._qm_state, self._cl_state)
        return BasicState(qm_state, cl_state, self._nbits,  last_measured)




