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

	

Basic Design Layout
===================

PyQCS has two fundamental classes for simulating the quantum computation:
A state which represents the total simulator state at a single point in 
quasi-time [1]_ and gate circuits that can be applied to such a state;
yielding a new state.

PyQCS states
------------

A PyQCS state contains a representation of the quantum mechanical state in which
the simulator is; using a numpy array. The application of a gate will return a new state with 
a changed qm state. 

The state also contains a representation of the last measurement and information which qbits 
have been measured. This information will be used by classical parts of an algorithm.

PyQCS gates
-----------

A PyQCS gate is essentially a function mapping a ``2**N`` dimensional ``cdouble`` array and an
``N`` dimensional ``double`` array to a ``2**N`` dimensional ``cdouble`` array,
an ``N`` dimensional ``double`` array and a ``int64`` scalar.

PyQCS gates usually are implemented as objects with a numpy ufunc backend and some data. 

A normal user will never access the gates directly but use either `PyQCS gate circuits`_ or
`PyQCS gate circuit builders`_

PyQCS gate circuits
-------------------

Circuits are way one describes how the gates are applied to the state. Even single gate applications are
described as circuits as those are more convenient to use. Gate circuits can also be used to optimize
circuits in an abstract way and implement the error simulation.

PyQCS gate circuit builders
---------------------------

Gate circuit builders are a more abstract way to construct gate circuits and are used typically to reduce
the effort to construct a circuit. When called a circuit builder returns a new circuit with the
given parameters. Typical cases are circuits that can be applied to different qbits.
 

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
- Allow states to be multiplied with each other to compute the overlap.




.. [1] Real quantum computers have an intrinsic time evolution. This is omitted
       in PyQCS and reintroduced for error simulation. PyQCS therefore operates
       on a discrete quasi-time with every time-site being before or after a gate
       application.

