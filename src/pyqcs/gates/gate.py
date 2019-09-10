import numpy

from .abc import AbstractGate
from .implementations.basic_gates import BasicGate


class BaseGate(AbstractGate):
    def __init__(self, impl):
        self._impl = impl

    def __call__(self, qm_state, cl_state):
        return self._impl(qm_state, cl_state)
    def is_inplace(self):
        return False

class BuiltinGate(BaseGate):
    def __init__(self, type_, act, control, r):
        BaseGate.__init__(self, BasicGate(type_, act, control, r, numpy.random.uniform))

