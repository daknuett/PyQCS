from .gate import BuiltinGate
from .circuits import SingleGateCircuit
from ..build.abc import AbstractSingleGateCircuitBuilder

class BuiltinGateBuilder(AbstractSingleGateCircuitBuilder):
    __slots__ = ["_type"]
    def __init__(self, type_):
        AbstractSingleGateCircuitBuilder.__init__(self)
        self._type = type_

    def __call__(self, act, *args):
        if(args not in self._registry):
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

_R = BuiltinGateBuilder('R')
def R(act, r):
    return _R(act, 0, r)
