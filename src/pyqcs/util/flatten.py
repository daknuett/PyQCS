from ..gates.circuits import SingleGateCircuit, AbstractCompoundGateCircuit

def _flatten_generator(circuit):
    if(isinstance(circuit, SingleGateCircuit)):
        yield circuit
        return

    if(isinstance(circuit, AbstractCompoundGateCircuit)):
        for subc in circuit._subcircuits:
            yield from _flatten_generator(subc)
        return

    raise ValueError(f"cannot flatten {type(circuit)}")

def flatten(circuit):
    """
    Flattens a (possibly compound) circuit to a list of SingleGateCircuits.
    """
    return list(_flatten_generator(circuit))
