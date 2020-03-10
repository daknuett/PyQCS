from .abc import AbstractState
import numpy as np

class BasicState(AbstractState):
    __slots__ = ["_nbits", "_ndim", "_qm_state", "_cl_state", "_last_measured"]
    def __init__(self
                , qm_state
                , cl_state
                , nbits
                , last_measured
                , lenght_error=1e-8
                , check_normalization=False
                , **kwargs):
        AbstractState.__init__(self)
        self._is_naive = True
        self._qm_state = qm_state
        self._cl_state = cl_state
        self._nbits = nbits
        self._ndim = 2**nbits
        self._last_measured = last_measured
        self._length_error = lenght_error
        self._check_normalization = check_normalization

        if(check_normalization and not self.is_normalized()):
            raise ValueError("State has non-one length")

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

    def check_qbits(self, gate_circuit):
        if(gate_circuit._uses_qbits < (1 << self._nbits)):
            return True
        return False

    def deepcopy(self, **kwargs):
        keyword_arguments = {"check_normalization": self._check_normalization
                            , "lenght_error": self._length_error}
        keyword_arguments.update(kwargs)
        return BasicState(self._qm_state.copy()
                        , self._cl_state.copy()
                        , self._nbits
                        , self._last_measured
                        , **keyword_arguments)

    def apply_gate(self, gate, force_new_state=False):
        qm_state, cl_state, last_measured = gate(self._qm_state, self._cl_state)
        return BasicState(qm_state
                        , cl_state
                        , self._nbits
                        ,  last_measured
                        , lenght_error=self._length_error
                        , check_normalization=self._check_normalization)

    def _easy_format_state_part(self, cf, i):
        return "{}*|{}>".format(str(cf), bin(i))

    def __str__(self):
        eps = 1e-13

        s = " + ".join((self._easy_format_state_part(self._qm_state[i], i)
                        for i in range(self._ndim) if (abs(self._qm_state[i]) > eps)))
        return s

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((tuple(self._qm_state), tuple(self._cl_state)))

    def __eq__(self, other):
        if(isinstance(other, BasicState)):
            if(self._nbits != other._nbits):
                return False
            if(not np.allclose(self._cl_state, other._cl_state)):
                return False

            overlap = self @ other
            if(not np.isclose(np.absolute(overlap), 1)):
                return False
            return True

        raise TypeError()

    def __matmul__(self, other):
        if(not isinstance(other, BasicState)):
            raise TypeError()
        if(not (other.is_normalized() and self.is_normalized())):
            raise ValueError("states must be normalized")
        if(self._nbits != other._nbits):
            raise ValueError("states must have same qbit count")

        return self._qm_state.conjugate().dot(other._qm_state)

    def is_normalized(self):
        return np.isclose(np.sum(np.absolute(self._qm_state)**2), 1, atol=self._length_error)
