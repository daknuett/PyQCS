from abc import ( ABCMeta
                , abstractclassmethod
                , abstractmethod
                , abstractstaticmethod)


from .executor import GateListExecutor, RepeatingGateListExecutorSpawner
from ..state.abc import AbstractState

class AbstractGateCircuit(metaclass=ABCMeta):
    def __init__(self, qbits, identities):
        self._uses_qbits = qbits
        self._identities = identities
        self._executor = GateListExecutor

    def __mul__(self, other):
        if(isinstance(other, AbstractState)):
            if(not other.check_qbits(self)):
                raise ValueError(
                        "Gate circuit requires qbits {}, but given state has less.".format(
                            bin(self._uses_qbits)
                            ))
            return self._executor(self.get_child_executors())(other)
        if(isinstance(other, int)):
            return self.new_from_circuit_with_executor(RepeatingGateListExecutorSpawner(other))
        raise TypeError()

    def __rmul__(self, other):
        if(isinstance(other, int)):
            return self.new_from_circuit_with_executor(RepeatingGateListExecutorSpawner(other))
        raise TypeError()

            
    def add_identity(self, identity):
        self._identities.append(identity)

    def to_executor(self):
        return self._executor(self.get_child_executors())

    @abstractmethod
    def __or__(self, other):
        pass

    @abstractmethod
    def __ror__(self, other):
        pass
    
    @abstractmethod
    def get_child_executors(self):
        pass

    @abstractmethod
    def new_from_circuit_with_executor(self, executor):
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

