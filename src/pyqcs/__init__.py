"""
PyQCS
*****

This package contains a quantum computing simulator.

The following gates are imported:

    C(act, control): CNOT 
    X(act): NOT
    H(act): Hadamard
    R(act, phi): Rotation
    M(act): Measurement
"""

from .gates.builtins import C, H, X, R, M
from .state.state import BasicState as State
