from .abc import AbstractGateCircuit, AbstractNamedGateCircuit, AbstractCompoundGateCircuit

class SingleGateCircuit(AbstractNamedGateCircuit):
    def __init__(self, qbits, identities, name, gate):
        AbstractNamedGateCircuit.__init__(self, qbits, identities, name)
        self._gate = gate

    #def to_dot(self):
    #    return self.stock_to_dot()

    def to_gate_list(self):
        return [self._gate]

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


class AnonymousCompoundGateCircuit(AbstractCompoundGateCircuit):
    def __init__(self, subcircuit_list, qbits=None):
        if(qbits is None):
            qbits = 0
            for subcircuit in subcircuit_list:
                qbits |= subcircuit._uses_qbits
        AbstractCompoundGateCircuit.__init__(self, qbits, [], subcircuit_list)

    def to_gate_list(self):
        return list(self.gate_list_generator())
    def gate_list_generator(self):
        for subcircuit in self._subcircuits:
            yield from subcircuit.gate_list_generator()
    def __or__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([self, other])

    def __ror__(self, other):
        if(not isinstance(other, AbstractGateCircuit)):
            raise TypeError()
        return AnonymousCompoundGateCircuit([other, self])
        
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

