"""
PyQCS
*****

This package contains a quantum computing simulator.

The following gates are imported:

    C(act, control): CNOT 
    X(act): NOT
    H(act): Hadamard
    R(act, phi): Rotation
    M(act): Measurement

The following functions are provided to make measurements easier:

    measure(state, bit_mask) -> (new_state, classical_result)
    sample(state, bit_mask, nsamples, keep_states=False) -> {classical_result: count} or {(new_state, classical_result): count}

Note that ``measure(state, 1)`` is not equivalent to ``M(1) * state``, as ``measure`` will NOT keep previous measurements.
The same applies for ``sample``
"""

from collections import Counter
import numpy as np

from .gates.builtins import C, H, X, R, M
from .state.state import BasicState as State

def measure(state, bit_mask):
    c_list = [M(i) for i in range(bit_mask.bit_length()) if bit_mask & (1 << i)]
    circuit = c_list[0]
    for c in c_list[1:]:
        circuit |= c

    state = State(state._qm_state, np.array([-1 for i in state._cl_state], dtype=np.int8), state._nbits, 0)
    new_state = circuit * state
    return new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])
    

def _do_sample(state, circuit, nsamples):
    for i in range(nsamples):
        new_state = circuit * state
        yield new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])

def sample(state, bit_mask, nsamples, keep_states=False):
    c_list = [M(i) for i in range(bit_mask.bit_length()) if bit_mask & (1 << i)]
    circuit = c_list[0]
    for c in c_list[1:]:
        circuit |= c

    state = State(state._qm_state, np.array([-1 for i in state._cl_state], dtype=np.int8), state._nbits, 0)

    if(keep_states):
        return Counter(_do_sample(state, circuit, nsamples))
    
    return Counter((i[1] for i in _do_sample(state, circuit, nsamples)))


