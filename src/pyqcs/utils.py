from .gates.circuits import Circuit
from .util.to_diagram import CircuitPNGFormatter


def list_to_circuit(list_of_circuits):
    if(not list_of_circuits):
        return
    if(not isinstance(list_of_circuits, list)):
        list_of_circuits = list(list_of_circuits)

    gates = [g for c in list_of_circuits for g in c._gate_list]
    capabilities = max((c._requires_capabilities for c in list_of_circuits)
                        , key=lambda x: x._bitmask)
    qbits = 0
    for c in list_of_circuits:
        qbits |= c._requires_qbits

    return Circuit(capabilities, gates, qbits)


def circuitpng(circuit, **kwargs):
    return CircuitPNGFormatter(circuit, **kwargs)
