from .gates.builtins import M
from .utils import list_to_circuit
from .state.state import BasicState as State
from collections import Counter

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
    if(isinstance(state, State)):
        return new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])
    else:
        return new_state, sum([1 << i for i,v in new_state._measured.items() if v == 1])


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

