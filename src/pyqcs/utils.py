from .gates.circuits import Circuit
from .util.to_diagram import CircuitPNGFormatter


def list_to_circuit(list_of_circuits: Circuit):
    gates = [g for g in c._gate_list for c in list_of_circuits]
    capabilities = max((c._requires_capabilities for c in list_of_circuits)
                        , key=lambda x: x._bitmask)
    qbits = 0
    for c in list_of_circuits:
        qbits |= c._requires_qbits

    return Circuit(capabilities, gates, qbits)


def circuitpng(circuit, **kwargs):
    return CircuitPNGFormatter(circuit, **kwargs)
