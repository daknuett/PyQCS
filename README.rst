PyQCS -- A quantum computing simulator
**************************************

.. image:: https://travis-ci.org/daknuett/PyQCS.svg?branch=master
    :target: https://travis-ci.org/daknuett/PyQCS
.. image:: https://api.codacy.com/project/badge/Grade/a6ca800c070a46f297216d03f9351129
    :target: https://www.codacy.com/manual/daknuett_2/PyQCS?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=daknuett/PyQCS&amp;utm_campaign=Badge_Grade

.. contents::

Links
=====

- `the github repository <https://github.com/daknuett/pyqcs>`_
- `the pypi repository <https://pypi.org/project/pyqcs/>`_
- `some examples on github <https://github.com/daknuett/PyQCS/tree/master/examples>`_

What Is PyQCS?
==============

PyQCS is a Quantum Computing Simulator built for physics. It currently features
two simulator backends for different purposes, and a way to construct circuits
in a relatively readable manner.

By default PyQCS employs a relatively slow simulator backend using dense state
vectors stored in NumPy arrays. The gates are implemented as NumPy ufuncs.
Because the states are implemented as NumPy arrays there is a significant
overhead compared to other simulators. Note that any implementation using dense
state vectors shows exponential growth in the number of qbits. Therefore this
simulator backend is limited to qbit numbers  below 30 for reasonable
performance. For simulations requiring more qbits we recommend a high
performance framework, such as `GPT's QIS module
<https://github.com/lehner/gpt>`_.

The second backend uses a graphical state representation (see for instance
`arXiv:quant-ph/0504117 <https://arxiv.org/abs/quant-ph/0504117v2>`_) which
allows for the simulation of stabilizer states and -circuits. The graphical
simulator is considerably faster, in particular it does not exhibit exponential
growth in the number of qbits. The graphical states are available as
``pyqcs.graph.state.GraphState``.

Unlike other simulators PyQCS focuses on the state: Users start from a state, modify
the state (using circuits) and then either look at the state or sample from the state.
This direct access to the state is useful when debugging circuits or when considering
physical problems. However, it slows down compuations.

Using PyQCS
===========

To do some computation one has to build a quantum circuit and apply it to a state.
States are created using ``pyqcs.State.new_zero_state(<number of qbits>)``.

Circuits are built from the fundamental gates (see `Built-in Gates`_) by joining them
together using the ``|`` operator::

	from pyqcs import H, CX, X

	circuit = H(0) | CX(1, 0) | X(1)

The usage of the ``|`` is in analogy to the UNIX pipe: gates are applied from left to
right. This is in agreement with the Feynman quantum circuit diagrams.

**Note**: the circuit above would have the following matrix representation::

	X(1) CZ(1,0) H(0)

Applying a circuit to a state is done using multiplication::

	from pyqcs import State

	state = State.new_zero_state(2)
	resulting_state = circuit * state


New in ``v2.2.0`` is the ``circuitpng`` function that allows displaying circuits as PNGs
(using a ``pdflatex`` implementation and ``imagemagick``)::

      from pyqcs import H, CX, circuitpng
      circuit = (H(1) | H(2)) | CX(2, 1) | (H(1) | H(2))
      circuitpng(circuit)
	

Built-in Gates
==============

PyQCS currently has the following gates built-in:

``X``
	Pauli-X or NOT gate. Flips the respective qbit.
``H``
	Hadamard gate. 
``C = CX``
	CNOT (controlled NOT) gate. Flips the act-qbit, if the control-qbit is set.
``R``
	R, Rz or R_phi, the rotation gate. Rotates the respective qbit around a given angle.
``M``
	Measurement gate: this gate measures the respective gate, collapsing the wave function
	and storing the result in the classical part of the state.
``Z``
	Pauli-Z gate.
``B = CZ``
	Controlled Z gate.


TODOs
=====

- Add a subclass of ``pyqcs.state.state.BasicState`` that has an improved ``__str__`` method.
- Write lot's of documentation.
- Add more tests.
- Add a ``NoisyGateListExecutor`` that allows to implement a noise model.
- Allow graphical states to be multiplied with each other to compute the overlap.
- Add a way to use graphical states as basis states for compression.
- Add a fast dense state vector simulator.
- Add a way to export circuits to GPT's QIS module.




.. [1] Real quantum computers have an intrinsic time evolution. This is omitted
       in PyQCS and reintroduced for error simulation. PyQCS therefore operates
       on a discrete quasi-time with every time-site being before or after a gate
       application.

