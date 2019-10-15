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
        qm_state = np.zeros(ndim, dtype=np.cfloat)
        qm_state[0] = 1
        last_measured = 0
        cl_state = -1 * np.ones(nbits, dtype=np.int8)

        return cls(qm_state, cl_state, nbits, last_measured, **kwargs)

    def get_last_measurement(self):
        return self._cl_state, self._last_measured

    def check_qbits(self, gate_circuit):
        if(gate_circuit._uses_qbits < (1 << self._nbits)):
            return True
        return False

    def deepcopy(self):
        return BasicState(self._qm_state.copy(), self._cl_state.copy(), self._nbits, self._last_measured)

    def apply_gate(self, gate, force_new_state=False):
        qm_state, cl_state, last_measured = gate(self._qm_state, self._cl_state)
        return BasicState(qm_state, cl_state, self._nbits,  last_measured)

    def _easy_format_state_part(self, cf, i):
        return "{}*|{}>".format(str(cf), bin(i))

    def __str__(self):
        eps = 1e-13

        s = " + ".join((self._easy_format_state_part(self._qm_state[i], i)
                        for i in range(self._ndim) if (abs(self._qm_state[i]) > eps)))
        return s

    def __repr__(self):
        return str(self)
