PyQCS to GQCS Byte Code
***********************

The PyQCS to GQCS byte code describes how
circuits constructed using PyQCS can be 
executed more efficiently using GQCS a pure C
implementation of the ``pyqcs.graph`` package.

.. contents::

Magic Word and Header
=====================

- The first ``4`` bytes of any file interpreted by GQCS must always be 
  ``[0x47, 0x51, 0x43, 0x53]`` (UTF-8 encoded ``GQCS``). This is
  called the magic word of the byte code.

- The next ``8`` bytes of the file are a ``64`` bit unsigned integer 
  that is interpreted as the number of qbits.

- The ``12`` th byte must have the value ``0x62`` (UTF-8 encoded ``b``).

- Bytes ``13-14`` (two bytes) are an unsigned ``16`` bit
  integer and are interpreted as the number of samples.

- The ``15`` th byte must have the value ``0x73`` (UTF-8 encoded ``s``).

- Byte ``16-24`` are a ``64`` bit unsigned integer that is interpreted
  as the number of qbits to sample. This variable will from now on be called
  ``q``.

- The ``25`` th byte must have the value ``0x71`` (UTF-8 encoded ``q``).

- The ``q*8`` bytes are interpreted as a list of unsigned ``64`` bit integers
  and are the qbits to sample after execution of the circuit. They
  are sampled in the order of their occurance.

- The ``8`` bytes starting from ``25 + q*8`` th must have the value ``0xff``.


Instructions
============

Instructions have ``17`` bytes length. The first byte identifies the 
command (see `Commands`_), the bytes ``1-9`` are the act-qbit.
Bytes ``10-17`` are data for the command.

The first byte of an instruction must always have one of the following values:
``L``, ``Z``, ``M``.

Commands
========

Commands available in the byte code are local Clifford gates (``L``),
controlled Z gate (``Z``) and measurement (``M``). Currently 
there is no way to store measurement results, just the sampling at the
end of the program will produce output.

Local Clifford Gates
--------------------

Local Clifford gates take the clifford index in the 0th byte of the data
(10th byte of the instruction).


Controlled Z Gate
-----------------

The CZ gate takes the control-qbit as a 64 bit unsigned integer in 
the data.

Measurement
-----------

The data is ignored.



