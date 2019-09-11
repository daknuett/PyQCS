from .abc import AbstractGateCircuit, AbstractNamedGateCircuit, AbstractCompoundGateCircuit
from .executor import GateListExecutor

class SingleGateCircuit(AbstractNamedGateCircuit):
    def __init__(self, qbits, identities, name, gate):
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name)
        self._gate = gate


    def get_child_executors(self):
        return [GateListExecutor([self._gate])]

    def gate_list_generator(self):
        yield self._gate
    def __or__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([self, other])

    def __ror__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([other, self])


    def new_from_circuit_with_executor(self, executor):
        new_circuit = AnonymousCompoundGateCircuit([self], self._uses_qbits)
        new_circuit._executor = executor
        return new_circuit


class AnonymousCompoundGateCircuit(AbstractCompoundGateCircuit):
    def __init__(self, subcircuit_list, qbits=None):
        if(qbits is None):
            qbits = 0
            for subcircuit in subcircuit_list:
                qbits |= subcircuit._uses_qbits
        AbstractCompoundGateCircuit.__init__(self, qbits, [], subcircuit_list)

    def __or__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([self, other])

    def __ror__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([other, self])

    def new_from_circuit_with_executor(self, executor):
        new_circuit = AnonymousCompoundGateCircuit(self._subcircuits, self._uses_qbits)
        new_circuit._executor = executor
        return new_circuit
        
    def get_child_executors(self):
        return [c.to_executor() for c in self._subcircuits]

class NamedCompoundGateCircuit(AnonymousCompoundGateCircuit, AbstractNamedGateCircuit):
    def __init__(self, subcircuit_list, name, identities=[]):
        qbits = 0
        for subcircuit in subcircuit_list:
            qbits |= subcircuit._uses_qbits
        AnonymousCompoundGateCircuit.__init__(self, subcircuit_list, qbits=qbits)
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name)

    @classmethod
    def from_anonymous(cls, anonymous, name, identities=[]):
        if(not isinstance(anonymous, AnonymousCompoundGateCircuit)):
            raise TypeError("anonymous must be of type AnonymousCompoundGateCircuit")
        subcircuit_list = anonymous._subcircuits
        return cls(subcircuit_list, name, identities)


    def new_from_circuit_with_executor(self, executor):
        # Don't give that new circuit a name. It is a new 
        # circuit.
        new_circuit = AnonymousCompoundGateCircuit([self], self._uses_qbits)
        new_circuit._executor = executor
        return new_circuit

