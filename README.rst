PyQCS -- A quantum computing simulator
**************************************

.. image:: https://travis-ci.org/daknuett/PyQCS.svg?branch=master
    :target: https://travis-ci.org/daknuett/PyQCS
.. image:: https://api.codacy.com/project/badge/Grade/a6ca800c070a46f297216d03f9351129
    :target: https://www.codacy.com/manual/daknuett_2/PyQCS?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=daknuett/PyQCS&amp;utm_campaign=Badge_Grade
.. image:: https://github.com/daknuett/pyqcs/actions/workflows/test-tox-fast.yml/badge.svg

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
vectors. Note that any implementation using dense state vectors shows
exponential growth in the number of qbits. Therefore this simulator backend is
limited to qbit numbers  below 30 for reasonable performance. For simulations
requiring more qbits we recommend using a high performance framework, such as
`GPT's QIS module <https://github.com/lehner/gpt>`_.

The second backend uses a graphical state representation (see for instance
`arXiv:quant-ph/0504117 <https://arxiv.org/abs/quant-ph/0504117v2>`_) which
allows for the simulation of stabilizer states and -circuits. The graphical
simulator is considerably faster, in particular it does not exhibit exponential
growth in the number of qbits. The graphical states are available as
``pyqcs.graph.state.GraphState``.

Unlike other simulators PyQCS focuses on the state: Users start from a state, modify
the state (using circuits) and then either look at the state or sample from the state.
This direct access to the state is useful when debugging circuits or when considering
physical problems. However, it can slow down compuations.

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

**Note**: the circuit above would have the following matrix representation:

.. math::

	X_1 CZ_{1,0} H_0

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
``CX``
	CNOT (controlled NOT) gate. Flips the act-qbit, if the control-qbit is set.
``R``
	R, Rz or R_phi, the rotation gate. Rotates the respective qbit around a given angle.
``M``
	Measurement gate: this gate measures the respective gate, collapsing the wave function
	and storing the result in the classical part of the state.
``Z``
	Pauli-Z gate.
``S``
	Clifford-S gate.
``CZ``
	Controlled Z gate.


Using the C++ Backend
=====================

Starting from version 3.0.0 PyQCS has a pure C++ backend omitting the
previously used numpy arrays. The python package uses handwritten adapters to
this backend.

The backend can be used as a stand-alone library. It can be built and installed
using the ``src/backend/meson.build`` file. Its usage is a bit different from
what one would usually expect from a simulator: Operations like measurement are
not implemented explicitly but should be implemented by the user using a random
number generator and the provided ``compute_amplitude`` methods.

Besides that the code should be pretty much self-explaining. See the adapter
code for some ideas how to use the backend.

TODOs
=====

- Add pretty printers for states.
- Write lot's of documentation.
- Add more tests.
- Add a noise model.
- Add a way to export circuits to GPT's QIS module.




.. [1] Real quantum computers have an intrinsic time evolution. This is omitted
       in PyQCS and reintroduced for error simulation. PyQCS therefore operates
       on a discrete quasi-time with every time-site being before or after a gate
       application.

