from abc import ( ABCMeta
                , abstractmethod)


from .executor import GateListExecutor, RepeatingGateListExecutorSpawner
from ..state.abc import AbstractState
from ..graph.abc import AbstractGraphState

class AbstractGateCircuit(metaclass=ABCMeta):
    def __init__(self, qbits, identities):
        self._has_graph = False
        self._has_naive = False
        self._uses_qbits = qbits
        self._identities = identities
        self._executor = GateListExecutor

    def __mul__(self, other):
        if(isinstance(other, AbstractGraphState)):
            if(not self._has_graph):
                raise TypeError("Cannot apply circuit to graph state. Check your gates.")
            if(not other.check_qbits(self)):
                raise ValueError(
                        "Gate circuit requires qbits {}, but given state has less.".format(
                            bin(self._uses_qbits)
                            ))
            return self._executor(self.get_child_executors(graph=True))(other)
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

    def to_executor(self, graph=False):
        return self._executor(self.get_child_executors(graph=graph))

    @abstractmethod
    def __or__(self, other):
        pass

    @abstractmethod
    def __ror__(self, other):
        pass

    @abstractmethod
    def get_child_executors(self, graph=False):
        pass

    @abstractmethod
    def new_from_circuit_with_executor(self, executor):
        pass

    @abstractmethod
    def get_dagger(self):
        pass

class AbstractNamedGateCircuit(AbstractGateCircuit):
    def __init__(self, qbits, identities, name, descr):
        AbstractGateCircuit.__init__(self, qbits, identities)
        self._name = name
        self._descr = descr

class AbstractCompoundGateCircuit(AbstractGateCircuit):
    __slots__ = ["_subcircuits"]
    def __init__(self, qbits, identities, subcircuits):
        AbstractGateCircuit.__init__(self, qbits, identities)
        self._subcircuits = subcircuits


class AbstractGate(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, qm_state, cl_state):
        pass
    @abstractmethod
    def get_dagger(self):
        pass

