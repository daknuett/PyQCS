from .gate import BuiltinGate, GenericGate as _GenericGate
from .circuits import SingleGateCircuit
from ..build.abc import AbstractSingleGateCircuitBuilder

class BuiltinGateBuilder(AbstractSingleGateCircuitBuilder):
    __slots__ = ["_type"]
    def __init__(self, type_):
        AbstractSingleGateCircuitBuilder.__init__(self)
        self._type = type_

    def __call__(self, act, *args):
        if((act, *args) not in self._registry):
            gate = BuiltinGate(self._type, act,*args)
            circuit = SingleGateCircuit((1 << act)
                        , []
                        , self._type + "(" + ",".join((str(a) for a in args)) + ")"
                        , (self._type, act, *args)
                        , gate)
            self._registry[(act, *args)] = circuit
        return self._registry[(act, *args)]

_H = BuiltinGateBuilder('H')
def H(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _H(act, 0, 0)

_X = BuiltinGateBuilder('X')
def X(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _X(act, 0, 0)

_M = BuiltinGateBuilder('M')
def M(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _M(act, 0, 0)

_C = BuiltinGateBuilder('C')
def C(act, control):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    if(act == control):
        raise ValueError("act qbit and control qbit must be different")
    return _C(act, control, 0)

CX = C
CNOT = C

_R = BuiltinGateBuilder('R')
def R(act, r):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _R(act, 0, r)

_Z = BuiltinGateBuilder('Z')
def Z(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _Z(act, 0, 0)

_B = BuiltinGateBuilder('B')
def CZ(act, control):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    if(act == control):
        raise ValueError("act qbit and control qbit must be different")
    return _B(act, control, 0)

def GenericGate(act, array):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    gate = _GenericGate(act, array)

    return SingleGateCircuit(1 << act, [], str(array), ("GenericGate", act, array), gate)
