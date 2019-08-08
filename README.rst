PyQCS -- A quantum computing simulator
**************************************

.. contents::

Basic Design Layout
===================

PyQCS has two fundamental classes for simulating the quantum computation:
A state which represents the total simulator state at a single point in 
quasi-time [1]_ and gate circuits that can be applied to such a state 
yielding a new state.

PyQCS states
------------

A PyQCS state contains a representation of the quantum mechanical state in which
the simulator is using a numpy array. The application of a gate will return a new state with 
a changed qm state. 

The state also contains a representation of the last measurement and information which qbits 
have been measured. This information will be used by classical parts of an algorithm.

PyQCS gates
-----------

A PyQCS gate is essentially a function mapping a ``2**N`` dimensional ``cfloat`` array and an
``N`` dimensional ``double`` array to a ``2**N`` dimensional ``cfloat`` array,
an ``N`` dimensional ``double`` array and a ``int64`` scalar. To save memory these functions
can also be implemented in-place [2]_ for large states (i.e. more than 24 qbits, which will require 
``0.25GiB`` RAM).

PyQCS gates usually are implemented as objects with a numpy ufunc backend, some data and a function
``is_inplace()`` to check whether the computation is done in-place. 

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
 





.. [1] Real quantum computers have an intrinsic time evolution. This is omitted
       in PyQCS and reintroduced for error simulation. PyQCS therefore operates
       on a discrete quasi-time with every time-site being before or after a gate
       application.

.. [2] This feature is experimental and should not be used in production.
