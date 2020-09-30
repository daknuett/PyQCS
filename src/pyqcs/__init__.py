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

import numpy as np

from .gates.builtins import C, H, X, R, M, Z, S, CX, CZ, GenericGate, clear_builtin_gate_cache
from .state.state import BasicState as State
from .gates.circuits import AnonymousCompoundGateCircuit, NamedCompoundGateCircuit


from .measurement import measure, sample, tree_amplitudes
from .utils import list_to_circuit, circuitpng

