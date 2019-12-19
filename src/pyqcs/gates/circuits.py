from .abc import AbstractGateCircuit, AbstractNamedGateCircuit, AbstractCompoundGateCircuit
from .executor import GateListExecutor

class SingleGateCircuit(AbstractNamedGateCircuit):
    def __init__(self, qbits, identities, name, descr, gate_naive, gate_graph):
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name, descr)
        if(gate_naive is not None):
            self._has_naive = True
        self._gate = gate_naive
        if(gate_graph is not None):
            self._has_graph = True
        self._gate_g = gate_graph


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
    def __init__(self, subcircuit_list, has_graph=None, has_naive=None, qbits=None):
        if(qbits is None):
            qbits = 0
            for subcircuit in subcircuit_list:
                qbits |= subcircuit._uses_qbits

        if(has_graph is None or has_naive is None):
            has_graph = True
            has_naive = True
            for subc in subcircuit_list:
                has_graph = has_graph and subc._has_graph
                has_naive = has_naive and subc._has_naive
            self._has_naive = has_naive
            self._has_graph = has_graph

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
        new_circuit._has_graph = self._has_graph
        new_circuit._has_naive = self._has_naive
        return new_circuit

    def get_child_executors(self):
        return [c.to_executor() for c in self._subcircuits]

class NamedCompoundGateCircuit(AnonymousCompoundGateCircuit, AbstractNamedGateCircuit):
    def __init__(self, subcircuit_list, name, descr, identities=[]):
        qbits = 0
        for subcircuit in subcircuit_list:
            qbits |= subcircuit._uses_qbits
        has_graph = True
        has_naive = True
        for subc in subcircuit_list:
            has_graph = has_graph and subc._has_graph
            has_naive = has_naive and subc._has_naive

        AnonymousCompoundGateCircuit.__init__(self, subcircuit_list, has_graph=has_graph, has_naive=has_naive, qbits=qbits)
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name, descr)

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

