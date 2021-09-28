from .gate import max_capabilities


class Circuit(object):
    __slots__ = [
        "_requires_capabilities"
        , "_gate_list"
        , "_requires_qbits"]

    def __init__(self, capabilities, gates, required_qbits):
        self._requires_capabilities = capabilities
        self._gate_list = gates
        self._requires_qbits = int(required_qbits)

    def __or__(self, other):
        if(not isinstance(other, Circuit)):
            raise TypeError()
        qbits = self._requires_qbits | other._requires_qbits
        capabilities = max_capabilities(self._requires_capabilities
                                        , other._requires_capabilities)
        gates = self._gate_list + other._gate_list
        return Circuit(capabilities, gates, qbits)

    def get_dagger(self):
        gates = [g for gate in reversed(self._gate_list)
                    for g in gate.get_dagger()]
        return Circuit(self._requires_capabilities
                        , gates
                        , self._requires_qbits)
