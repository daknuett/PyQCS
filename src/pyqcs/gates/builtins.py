import numpy as np
from .gate import BuiltinGate, GenericGate as _GenericGate
from .circuits import SingleGateCircuit
from .exceptions import UnitarityError
from ..build.abc import AbstractSingleGateCircuitBuilder
from ..graph.gate import GraphGate, CLOperation, CZOperation, MeasurementOperation

class BuiltinGateBuilder(AbstractSingleGateCircuitBuilder):
    __slots__ = ["_type"]
    def __init__(self, type_, get_graph_gate, has_control=False, can_dagger=True):
        AbstractSingleGateCircuitBuilder.__init__(self)
        self._type = type_
        self.get_graph_gate = get_graph_gate
        self._has_control = has_control
        self._can_dagger = can_dagger

    def __call__(self, act, *args, dagger=False):
        # This fixes a rare bug when generating circuits
        # with NumPy. When more qbits are used than
        # the NumPy fixed width integer can hold one can
        # experience weird behaviour
        act = int(act)

        if(dagger):
            args = (args[0], -args[1])

        if((act, *args) not in self._registry):
            gate = BuiltinGate(self._type, act, *args)
            gate_graph = self.get_graph_gate(act, *args, dagger=dagger)

            uses_qbits = (1 << act)
            if(self._has_control):
                uses_qbits |= (1 << args[0])

            def mk_dg():
                if(self._can_dagger):
                    return self(act, *args, dagger=True)
                else:
                    raise UnitarityError("gate is not unitary and cannot be daggered")

            circuit = SingleGateCircuit(uses_qbits
                        , []
                        , self._type + "(" + ",".join((str(a) for a in args)) + ")"
                        , (self._type, act, *args)
                        , gate
                        , gate_graph
                        , mk_dg)
            self._registry[(act, *args)] = circuit
        return self._registry[(act, *args)]

def _get_graph_H_gate(act, i1, i2, dagger=False):
    return GraphGate([CLOperation(act, 0)])
_H = BuiltinGateBuilder('H', _get_graph_H_gate)
def H(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _H(act, 0, 0)

def _get_graph_X_gate(act, i1, i2, dagger=False):
    return GraphGate([CLOperation(act, 14)])
_X = BuiltinGateBuilder('X', _get_graph_X_gate)
def X(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _X(act, 0, 0)


def _get_graph_M_gate(act, i1, i2, dagger=False):
    return GraphGate([MeasurementOperation(act)])
_M = BuiltinGateBuilder('M', _get_graph_M_gate, can_dagger=False)
def M(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _M(act, 0, 0)

def _get_graph_C_gate(act, control, i2, dagger=False):
    return GraphGate([CLOperation(act, 0)
                    , CZOperation(act, control)
                    , CLOperation(act, 0)])
_C = BuiltinGateBuilder('C', _get_graph_C_gate, has_control=True)
def C(act, control):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    if(act == control):
        raise ValueError("act qbit and control qbit must be different")
    return _C(act, control, 0)

CX = C
CNOT = C

def _get_graph_R_gate(i0, i1, i2, dagger=False):
    return None
_R = BuiltinGateBuilder('R', _get_graph_R_gate)
def R(act, r):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _R(act, 0, r)

def _get_graph_Z_gate(act, i1, i2, dagger=False):
    return GraphGate([CLOperation(act, 5)])
_Z = BuiltinGateBuilder('Z', _get_graph_Z_gate)
def Z(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _Z(act, 0, 0)

def _get_graph_B_gate(act, control, i2, dagger=False):
    return GraphGate([CZOperation(act, control)])
_B = BuiltinGateBuilder('B', _get_graph_B_gate, has_control=True)
def CZ(act, control):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    if(act == control):
        raise ValueError("act qbit and control qbit must be different")
    return _B(act, control, 0)

def _get_graph_S_gate(act, i1, i2, dagger=False):
    if(not dagger):
        return GraphGate([CLOperation(act, 1)])
    else:
        return GraphGate([CLOperation(act, 8)])

_S = BuiltinGateBuilder('R', _get_graph_S_gate)
def S(act):
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    return _S(act, 0, np.pi/2)


def GenericGate(act, array):
    """
    This is the circuit builder that is used to generate 1-qbit
    generic gates.
    """
    if(act < 0):
        raise ValueError("act qbit must be >= 0")
    gate = _GenericGate(act, array)

    circuit = SingleGateCircuit(1 << act
            , []
            , str(array)
            , ("GenericGate", act, array)
            , gate
            , None
            , None)
    def make_dagger():
        return SingleGateCircuit(1 << act
            , []
            , str(array.transpose().conjugate())
            , ("GenericGate", act, array.transpose().conjugate())
            , gate
            , None
            , lambda: circuit)

    circuit._make_dagger = make_dagger
    return circuit
