from collections import Counter, deque
from itertools import product

import numpy as np

from .gates.builtins import M
from .utils import list_to_circuit
from .state.state import BasicState as State
from .gates.implementations.compute_amplitude import compute_amplitude

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
    """
    Measures the qbits given in ``bit_mask`` ``nsamples`` times
    and returns a counter ``{result: count}``.

    If ``keep_states is True`` the resulting states are included in ``result``.
    This does not work for graphical states because there is currently no
    meaningful way to hash graphical states.

    The original state is unchanged.
    """
    circuit = build_measurement_circuit(bit_mask)

    state = state.deepcopy(force_new_state=True)
    if(isinstance(state, State)):
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
    of ``list(range(state._nbits))``.

    Only available for dense vector states.

    The original state is unchanged.

    Amplitudes (and collapsed states) are computed in the order given
    by ``bit_mask``. If ``bit_mask is None``, ``list(range(state._nbits))``
    is used.
    """

    if(not isinstance(state, State)):
        raise TypeError("tree_amplitudes currently works for dense vector states only")

    if(bit_mask is None):
        bit_mask = list(range(state._nbits))

    if(list(sorted(bit_mask)) != list(range(state._nbits))):
        raise ValueError("bit_mask must be either None or a permutation of list(range(state._nbits)))")

    next_queue = [(1, 0, state.deepcopy()._qm_state)]
    qbit_mapping = np.arange(0, 2**state._nbits, 1, dtype=int)

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

def py_compute_amplitude(state, qbits, bitstr):
    """
    This function is deprecated and will be removed in future versions.
    The function is replaced by the generalized UFunc ``compute_amplitude``.


    ``state`` is a ``pyqcs.State`` object, ``qbits`` is a list of qbits.
    The list ``bitstr`` contains the bitstring, for which the amplitude should be computed.
    ``bitstr[i]`` corresponds to the qbit ``qbits[i]``.

    Use ``compute_amplitudes`` instead of this function.
    """
    if(not isinstance(qbits, (list, tuple))):
        raise TypeError("qbits must be list, or tuple")
    if(not isinstance(bitstr, (list, tuple))):
        raise TypeError("bitstr must be list, or tuple")

    check_bits = sum(1 << qbit for qbit in qbits)
    bit_mask = sum(1 << bit for bit,msk in zip(qbits, bitstr) if msk)
    if(max(qbits) > state._nbits):
        raise ValueError(f"qbit {max(qbits)} out of range: {state._nbits}")

    amplitude = 0
    for i,v in enumerate(state._qm_state):
        bits_that_matter = i & check_bits
        if(bits_that_matter ^ bit_mask == 0):
            amplitude += (v*v.conj()).real
    return amplitude


def compute_amplitudes(state, qbits, eps=1e-8, asint=True):
    """
    ``state`` must be a ``pyqcs.State`` object and remains unchanged.

    Computes the amplitudes for all at-least-eps-probable measurement
    coutcomes for the ``qbits`` given either as an integer bit mask or
    a list of qbit indices.

    Returns a dict ``{outcome: probability}``.
    If ``asint == True`` the ``outcome`` is converted to an integer bit mask,
    in the other case ``outcome`` is a tuple of 0s and 1s where
    ``outcome[i]`` corresponds to ``qbits[i]``.

    In previous versions this used ``py_compute_amplitude`` to compute the
    individual amplitudes. To improve performance the UFunc
    ``pyqcs.gates.implementations.compute_amplitude.compute_amplitude``
    is used in newer versions.
    """
    if(isinstance(qbits, int)):
        qbits = [i for i in range(qbits.bit_length()) if qbits & (1 << i)]
    if(not isinstance(qbits, (list, tuple))):
        raise TypeError("qbits must be int, list, or tuple")
    if(max(qbits) > state._nbits):
        raise ValueError(f"qbit {max(qbits)} out of range: {state._nbits}")

    single_qbit_outcomes = [0, 1]

    results = dict()

    for outcome in product(*[single_qbit_outcomes]*len(qbits)):
        amplitude = compute_amplitude(state._qm_state
                                    , np.array(qbits, dtype=int)
                                    , np.array(outcome, dtype=np.uint8))
        if(amplitude > eps):
            if(asint):
                results[sum(1 << bit for bit,msk in zip(qbits, outcome) if msk)] = amplitude
            else:
                results[outcome] = amplitude

    return results

