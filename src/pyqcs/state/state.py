import numpy.random
import numpy
from .dsv import RawDSVState
from ..gates.circuits import Circuit
from ..gates.gate import Capabilities, Gate

_simple_gates = ["H", "Z", "X", "S"]
_parametric_gates = ["R"]
_two_qbit_gates = ["CX", "CZ"]
_measurement_gates = ["M"]


class DSVState(object):
    """
    The Dense State Vector simmulator's state class. Uses a C
    extnsion backend for faster simulation.
    """
    __slots__ = ["_backend_state", "_cl_state", "_nqbits"
                , "_rne", "_gate_executors", "_copy"]
    _has_capabilities = Capabilities.universal()

    def __init__(self, backend_state, cl_state, nqbits, rne=None, copy=True):
        self._nqbits = nqbits
        self._cl_state = cl_state
        self._backend_state = backend_state
        self._copy = copy

        if(rne is None):
            self._rne = lambda: numpy.random.uniform(0, 1)
        else:
            self._rne = rne

        self._gate_executors = {
            gn: self.__apply_simple_gate for gn in _simple_gates
        }
        self._gate_executors.update({
            gn: self.__apply_parametric_gate for gn in _parametric_gates
        })
        self._gate_executors.update({
            gn: self.__apply_two_qbit_gate for gn in _two_qbit_gates
        })
        self._gate_executors.update({
            gn: self.__do_measure for gn in _measurement_gates
        })

    @classmethod
    def new_zero_state(cls, nqbits, rne=None, copy=True):
        if(not isinstance(nqbits, int)):
            raise TypeError("nqbits must be int > 0")
        if(nqbits <= 0):
            raise ValueError("nqbits must be > 0")
        if(nqbits > 32):
            raise ValueError("nqbits is excessively large (max: 31)"
                            "use a HPC simulator for larger states")

        backend = RawDSVState(nqbits)
        cl_state = -numpy.ones(nqbits, dtype=int)
        return cls(backend, cl_state, nqbits, rne, copy)

    def check_qbits(self, circuit):
        return circuit._requires_qbits < (1 << self._nqbits)

    def deepcopy(self, **kwargs):
        nqbits = self._nqbits
        cl_state = self._cl_state.copy()
        backend = self._backend_state.deepcopy()
        kwa = {"rne": self._rne, "copy": self._copy}
        kwa.update(kwargs)
        return type(self)(backend, cl_state, nqbits, **kwa)

    def check_capabilities(self, circuit):
        return circuit._requires_capabilities <= type(self)._has_capabilities

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
            executor = state._gate_executors[gate._name]
            executor(gate)
        return state

    def __apply_simple_gate(self, gate: Gate):
        self._backend_state.apply_simple_gate(gate._act, gate._name)

    def __apply_parametric_gate(self, gate: Gate):
        self._backend_state.apply_parametric_gate(gate._act
                , gate._phi, gate._name)

    def __apply_two_qbit_gate(self, gate: Gate):
        self._backend_state.apply_two_qbit_gate(gate._act
                , gate._control, gate._name)

    def __do_measure(self, gate:Gate):
        result = self._backend_state.measure(gate._act, self._rne())
        self._cl_state[gate._act] = result

    def get_statistic(self):
        return self._backend_state.statistic()

    def __str__(self):
        data = self._backend_state.export_numpy()

        def fmt_elements(data, eps):
            for i, v in enumerate(data):
                if(abs(v) < eps):
                    continue
                yield f"{v}*|{bin(i)}>"
        return " + ".join(fmt_elements(data, 1e-3))

    def __repr__(self):
        data = self._backend_state.export_numpy()

        def fmt_elements(data, eps):
            for i, v in enumerate(data):
                if(abs(v) < eps):
                    continue
                yield f"{v}*|{bin(i)}>"
        return " + ".join(fmt_elements(data, 1e-3))

    def export_numpy(self):
        return self._backend_state.export_numpy()

    @property
    def _qm_state(self):
        return self._backend_state.export_numpy()

    def __eq__(self, other):
        if(not isinstance(other, DSVState)):
            raise TypeError()
        if(self._nqbits != other._nqbits):
            raise ValueError("states must have same number of qbits")

        if(not numpy.allclose(self._cl_state, other._cl_state)):
            return False

        overlap = self._backend_state.overlap(other._backend_state)
        return numpy.allclose(abs(overlap), 1)

    def __matmul__(self, other):
        if(not isinstance(other, DSVState)):
            raise TypeError()

        return self._backend_state.overlap(other._backend_state)

    def redo_normalization(self):
        self._backend_state.redo_normalization()

    def __hash__(self):
        return hash((tuple(self._qm_state), tuple(self._cl_state)))

    def randomize(self):
        self._backend_state.randomize(int(self._rne() * ((1 << 64) - 1)))
