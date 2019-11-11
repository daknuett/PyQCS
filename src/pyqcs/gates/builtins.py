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
            circuit = SingleGateCircuit((1 << act), [], self._type + "(" + ",".join((str(a) for a in args)) + ")", gate)
            self._registry[(act, *args)] = circuit
        return self._registry[(act, *args)]

_H = BuiltinGateBuilder('H')
def H(act):
    return _H(act, 0, 0)

_X = BuiltinGateBuilder('X')
def X(act):
    return _X(act, 0, 0)

_M = BuiltinGateBuilder('M')
def M(act):
    return _M(act, 0, 0)

_C = BuiltinGateBuilder('C')
def C(act, control):
    return _C(act, control, 0)

CX = C
CNOT = C

_R = BuiltinGateBuilder('R')
def R(act, r):
    return _R(act, 0, r)

_Z = BuiltinGateBuilder('Z')
def Z(act):
    return _Z(act, 0, 0)

_B = BuiltinGateBuilder('B')
def CZ(act, control):
    return _B(act, control, 0)

def GenericGate(act, array):
    gate = _GenericGate(act, array)

    return SingleGateCircuit(1 << act, [], str(array), gate)
