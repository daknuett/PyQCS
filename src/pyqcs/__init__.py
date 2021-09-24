"""
PyQCS
*****

This package contains a quantum computing simulator.

Notes on Using this Package
###########################

The following gates are imported:

    CX = C(act, control): CNOT
    X(act): NOT, Pauli X gate
    H(act): Hadamard
    R(act, phi): Rotation
    M(act): Measurement
    Z(act): Pauli Z gate
    S(act): S (pi over 4) gate
    CZ(act, control): Controlled Pauli Z gate


The following functions are provided to make measurements easier::

    measure(state, bit_mask) -> (new_state, classical_result)
    sample(state, bit_mask, nsamples, keep_states=False) -> {classical_result: count} or {(new_state, classical_result): count}

Note that ``measure(state, 1)`` is not equivalent to ``M(1) * state``, as
``measure`` will NOT keep previous measurements.  The same applies for
``sample``. Also refer to the section `Imported Utilities`_.

If you need the graphical simulation backend import
``pyqcs.graph.state.GraphState``.


For a quantum simulator state ``state`` one can apply a circuit using the
product::

    new_state = circuit * state

Note that the old state might or might not be modified: If ``state`` is a dense
vector state it will not be affected. If ``state`` is a graphical state and
``force_new_state`` is not specified the old state will be affected.

For a given state class create new states by calling the classmethod
``new_zero_state`` or ``new_plus_state`` (availability depends on the used
state class). See also: `Imported Classes`_.

Imported Utilities
##################

``measure``
    Measures once. Refer to function docstring.
``sample``
    Samples from a state. Refer to function docstring.
``compute_amplitudes``
    Computes amplitudes for a measrement outcomes. Refer to function docstring.
``tree_amplitudes``
    Computes amplitudes from a state. Refer to function docstring.
``list_to_circuit``
    Converts a list of circuits to a circuit.
``circuitpng``
    Renders a PNG representation of a given circuit in a jupyter notebook.
    Quality of the output varies.

Imported Classes
################

``State``
    This is ``pyqcs.state.state.BasicState``, the state used for
    dense vector simulations. The ``__str__`` method is somewhat
    rudimentary but does the trick.
``PrettyState``
    This is ``pyqcs.state.pretty_state.PrettyState``. It is a
    drop-in replacement for ``State`` and just comes with a fancier
    ``__str__`` method (which can be both helpful and annoying).

Graphical states are not imported, as it would lead to cyclic imports.
If you need the graphical simulation backend import
``pyqcs.graph.state.GraphState``.

Other Modules
#############

``pyqcs.util.random_circuits``
    Contains a way to generate random circuits. Refer to the module docstring
    for details.
``pyqcs.util.random_graphs``
    The function ``random_graph_state`` creates a graphical state with random
    vertex operators and randomized edges. Refer to the code for details.
``pyqcs.util.to_diagram``
    Contains ``circuit_to_diagram`` which converts a circuit to LaTeX code for
    qcircuit. Quality of the produced code varies.
``pyqcs.util.bytecode``
    Contains ``circuit_to_byte_code`` which converts a Clifford circuit to
    bytecode that can be executed to the pure C graphical simulator. Returns
    a bytecode object. You need to create and add a ``ByteCodeHeader`` to
    execute the bytecode.
``pyqcs.util.flatten``
    Refer to the docstring of function ``flatten``.
``pyqcs.util.from_lists``
    Refer to the code. Mostly for internal use.

Examples
########

See https://github.com/daknuett/PyQCS/tree/master/examples.

"""

import numpy as np

from .gates.builtins import H, X, R, M, Z, S, CX, CZ
from .state.state import DSVState as State


from .measurement import measure, sample, tree_amplitudes
from .utils import list_to_circuit, circuitpng

