from .gates.circuits import AnonymousCompoundGateCircuit, NamedCompoundGateCircuit
from .util.to_diagram import CircuitPNGFormatter

def list_to_circuit(list_of_circuits, name=None):
    if(not name):
        return AnonymousCompoundGateCircuit(list_of_circuits)
    return NamedCompoundGateCircuit(list_of_circuits, name, (name,))

def circuitpng(circuit, **kwargs):
    return CircuitPNGFormatter(circuit, **kwargs)
