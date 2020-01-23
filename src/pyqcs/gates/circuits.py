from .abc import AbstractGateCircuit, AbstractNamedGateCircuit, AbstractCompoundGateCircuit
from .executor import GateListExecutor

class SingleGateCircuit(AbstractNamedGateCircuit):
    def __init__(self
            , qbits
            , identities
            , name
            , descr
            , gate_naive
            , gate_graph
            , make_dagger):
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name, descr)
        if(gate_naive is not None):
            self._has_naive = True
        self._gate = gate_naive
        if(gate_graph is not None):
            self._has_graph = True
        self._gate_g = gate_graph

        self._make_dagger = make_dagger


    def get_child_executors(self, graph=False):
        if(not graph):
            return [GateListExecutor([self._gate])]
        else:
            return [GateListExecutor([self._gate_g])]

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
        new_circuit = AnonymousCompoundGateCircuit([self], qbits=self._uses_qbits
                                                    , has_graph=self._has_graph
                                                    , has_naive=self._has_naive)
        new_circuit._executor = executor
        return new_circuit

    def get_dagger(self):
        return self._make_dagger()


class AnonymousCompoundGateCircuit(AbstractCompoundGateCircuit):
    def __init__(self, subcircuit_list, has_graph=None, has_naive=None, qbits=None):
        if(qbits is None):
            qbits = 0
            for subcircuit in subcircuit_list:
                qbits |= subcircuit._uses_qbits

        AbstractCompoundGateCircuit.__init__(self, qbits, [], subcircuit_list)

        if(has_graph is None or has_naive is None):
            has_graph = True
            has_naive = True
            for subc in subcircuit_list:
                has_graph = has_graph and subc._has_graph
                has_naive = has_naive and subc._has_naive

        self._has_naive = has_naive
        self._has_graph = has_graph


    def __or__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([self, other])

    def __ror__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([other, self])

    def new_from_circuit_with_executor(self, executor):
        new_circuit = AnonymousCompoundGateCircuit(self._subcircuits, qbits=self._uses_qbits)
        new_circuit._executor = executor
        new_circuit._has_graph = self._has_graph
        new_circuit._has_naive = self._has_naive
        return new_circuit

    def get_child_executors(self, graph=False):
        return [c.to_executor(graph=graph) for c in self._subcircuits]

    def get_dagger(self):
        return AnonymousCompoundGateCircuit([s.get_dagger() for s in reversed(self._subcircuits)]
                                            , has_graph=self._has_graph
                                            , has_naive=self._has_naive
                                            , qbits=self._uses_qbits)

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

