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






.. [1] Real quantum computers have an intrinsic time evolution. This is omitted
       in PyQCS and reintroduced for error simulation. PyQCS therefore operates
       on a discrete quasi-time with every time-site being before or after a gate
       application.
