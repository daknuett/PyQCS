from collections import Counter, deque

import numpy as np

from .gates.builtins import M
from .utils import list_to_circuit
from .state.state import DSVState


def build_measurement_circuit(bit_mask):
    if(isinstance(bit_mask, int)):
        c_list = [M(i) for i in range(bit_mask.bit_length()) if bit_mask & (1 << i)]
    elif(isinstance(bit_mask, (list, tuple))):
        c_list = [M(i) for i in bit_mask]
    else:
        raise TypeError("bit_mask must be either int, list or tuple, but {} is given".format(type(bit_mask)))
    return list_to_circuit(c_list)


def measure(state, bit_mask):
    """
    Measures the qbits ``bit_mask`` (in the order given by
    ``bit_mask``, if bit_mask is a list, or least significant
    bit first).

    The original state is unchanged.

    Returns ``new_state, bit_string: int``.
    """
    circuit = build_measurement_circuit(bit_mask)

    state = state.deepcopy()
    state.redo_normalization()
    if(isinstance(state, DSVState)):
        state._cl_state[:] = -1
    else:
        state._measured = dict()
    new_state = circuit * state
    if(isinstance(state, DSVState)):
        return new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])
    else:
        return new_state, sum([1 << i for i,v in new_state._measured.items() if v == 1])


def _do_sample(state, circuit, nsamples):
    for _ in range(nsamples):
        new_state = circuit * state
        if(isinstance(new_state, DSVState)):
            yield new_state, sum([1 << i for i,v in enumerate(new_state._cl_state) if v == 1])
        else:
            yield new_state, sum([1 << i for i,v in new_state._measured.items() if v == 1])


def sample(state, bit_mask, nsamples, keep_states=False):
    """
    Measures the qbits given in ``bit_mask`` ``nsamples`` times
    and returns a counter ``{result: count}``.

    If ``keep_states is True`` the resulting states are included in ``result``.
    This does not work for graphical states because there is currently no
    meaningful way to hash graphical states.

    The original state is unchanged.
    """
    circuit = build_measurement_circuit(bit_mask)

    state = state.deepcopy(copy=True)
    state.redo_normalization()
    if(isinstance(state, DSVState)):
        state._cl_state[:] = -1
    else:
        state._measured = dict()

    if(keep_states):
        return Counter(_do_sample(state, circuit, nsamples))

    return Counter((i[1] for i in _do_sample(state, circuit, nsamples)))


def tree_amplitudes(state, bit_mask=None, eps=1e-5):
    """
    Compute the probability amplitudes for all (eps-)possible
    outcomes. ``bit_mask`` is either ``None`` or a permutation
    of ``list(range(state._nqbits))``.

    Only available for dense vector states.

    The original state is unchanged.

    Amplitudes (and collapsed states) are computed in the order given
    by ``bit_mask``. If ``bit_mask is None``, ``list(range(state._nqbits))``
    is used.
    """

    if(not isinstance(state, DSVState)):
        raise TypeError("tree_amplitudes currently works for dense vector states only")
    state.redo_normalization()

    if(bit_mask is None):
        bit_mask = list(range(state._nqbits))

    if(list(sorted(bit_mask)) != list(range(state._nqbits))):
        raise ValueError("bit_mask must be either None or a permutation of list(range(state._nqbits)))")

    next_queue = [(1, 0, state.deepcopy()._qm_state)]
    qbit_mapping = np.arange(0, 2**state._nqbits, 1, dtype=int)

    for qbit in bit_mask:
        this_queue = next_queue
        next_queue = deque()
        while(this_queue):
            prob, prev_outcome, handle_now = this_queue.pop()

            bit_mask_up = np.zeros_like(qbit_mapping, dtype=bool)
            bit_mask_up[np.where(qbit_mapping & (1 << qbit))] = 1

            amplitude_up = np.linalg.norm(handle_now[bit_mask_up])**2
            amplitude_down = np.linalg.norm(handle_now[~bit_mask_up])**2

            if(amplitude_up > eps):
                handle_next = handle_now.copy()
                handle_next[bit_mask_up] /= np.sqrt(amplitude_up)
                handle_next[~bit_mask_up] = 0
                next_queue.append((prob * amplitude_up, prev_outcome | (1 << qbit), handle_next))
            if(amplitude_down > eps):
                handle_next = handle_now.copy()
                handle_next[~bit_mask_up] /= np.sqrt(amplitude_down)
                handle_next[bit_mask_up] = 0
                next_queue.append((prob * amplitude_down, prev_outcome, handle_next))

    return [[outcome, prob] for prob, outcome, state in next_queue]
