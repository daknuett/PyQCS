"""
PyQCS
*****

This package contains a quantum computing simulator.

The following gates are imported:

    CX = C(act, control): CNOT
    X(act): NOT, Pauli X gate
    H(act): Hadamard
    R(act, phi): Rotation
    M(act): Measurement
    Z(act): Pauli Z gate
    CZ(act, control): Controlled Pauli Z gate


The following functions are provided to make measurements easier:

    measure(state, bit_mask) -> (new_state, classical_result)
    sample(state, bit_mask, nsamples, keep_states=False) -> {classical_result: count} or {(new_state, classical_result): count}

Note that ``measure(state, 1)`` is not equivalent to ``M(1) * state``, as ``measure`` will NOT keep previous measurements.
The same applies for ``sample``
"""

from collections import Counter
import numpy as np

from .gates.builtins import C, H, X, R, M, Z, S, CX, CZ, GenericGate
from .state.state import BasicState as State
from .gates.circuits import AnonymousCompoundGateCircuit, NamedCompoundGateCircuit

def list_to_circuit(list_of_circuits, name=None):
    if(not name):
        return AnonymousCompoundGateCircuit(list_of_circuits)
    return NamedCompoundGateCircuit(list_of_circuits, name, (name,))

def build_measurement_circuit(bit_mask):
    if(isinstance(bit_mask, int)):
        c_list = [M(i) for i in range(bit_mask.bit_length()) if bit_mask & (1 << i)]
    elif(isinstance(bit_mask, (list, tuple))):
        c_list = [M(i) for i in bit_mask]
    else:
        raise TypeError("bit_mask must be either int, list or tuple, but {} is given".format(type(bit_mask)))
    return list_to_circuit(c_list)


def measure(state, bit_mask):
    circuit = build_measurement_circuit(bit_mask)

    state = state.deepcopy()
    if(isinstance(state, State)):
        state._cl_state[:] = -1
    else:
        state._measured = dict()
    new_state = circuit * state
    return new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])


def _do_sample(state, circuit, nsamples):
    for _ in range(nsamples):
        new_state = circuit * state
        if(isinstance(new_state, State)):
            yield new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])
        else:
            yield new_state, sum([1 << i for i,v in new_state._measured.items() if v == 1])

def sample(state, bit_mask, nsamples, keep_states=False):
    circuit = build_measurement_circuit(bit_mask)

    state = state.deepcopy(force_new_state=True)
    if(isinstance(state, State)):
        state._cl_state[:] = -1
    else:
        state._measured = dict()

    if(keep_states):
        return Counter(_do_sample(state, circuit, nsamples))

    return Counter((i[1] for i in _do_sample(state, circuit, nsamples)))
