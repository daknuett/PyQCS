from abc import ( ABCMeta
                , abstractclassmethod
                , abstractmethod
                , abstractstaticmethod)


from .executor import GateListExecutor
from ..state.abc import AbstractState

class AbstractGateCircuit(metaclass=ABCMeta):
    def __init__(self, qbits, identities):
        self._uses_qbits = qbits
        self._identities = identities

    def __mul__(self, other):
        if(not isinstance(other, AbstractState)):
            raise TypeError()
        if(not other.check_qbits(self)):
            raise ValueError(
                    "Gate circuit requires qbits {}, but given state has less.".format(
                        self._uses_qbits
                        ))
        return GateListExecutor(self.to_gate_list())(other)
    @abstractmethod
    def to_gate_list(self):
        pass

    @abstractmethod
    def gate_list_generator(self):
        pass

    def add_identity(self, identity):
        self._identities.append(identity)

    @abstractmethod
    def __or__(self, other):
        pass

    @abstractmethod
    def __ror__(self, other):
        pass

class AbstractNamedGateCircuit(AbstractGateCircuit):
    def __init__(self, qbits, identities, name):
        AbstractGateCircuit.__init__(self, qbits, identities)
        self._name = name

class AbstractCompoundGateCircuit(AbstractGateCircuit):
    __slots__ = ["_subcircuits"]
    def __init__(self, qbits, identities, subcircuits):
        AbstractGateCircuit.__init__(self, qbits, identities)
        self._subcircuits = subcircuits

class AbstractGate(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, qm_state, cl_state):
        pass

