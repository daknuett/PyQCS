import numpy

from .abc import AbstractGate
from .implementations.basic_gates import BasicGate
from .implementations.generic_gate import GenericGate as _GenericGate


class BaseGate(AbstractGate):
    def __init__(self, impl):
        self._impl = impl

    def __call__(self, qm_state, cl_state):
        return self._impl(qm_state, cl_state)
    def is_inplace(self):
        return False

    def get_dagger(self):
        return BaseGate(self._impl.get_dagger())

class BuiltinGate(BaseGate):
    def __init__(self, type_, act, control, r):
        BaseGate.__init__(self, BasicGate(type_, act, control, r, numpy.random.uniform))

class GenericGate(BaseGate):
    """
    This is the Wrapper for the GenericGate C object.
    """
    def __init__(self, act, arr):
        if(not isinstance(arr, numpy.ndarray)):
            raise TypeError("matrices must be numpy ndarrays")
        if(len(arr.shape) != 2):
            raise ValueError("matrices must be of dimension NxN")
        if(arr.shape[0] != arr.shape[1]):
            raise ValueError("matrices must be of dimension NxN")

        test = arr.dot(arr.conj().T)
        if(not numpy.isclose(test, numpy.identity(arr.shape[0])).all()):
            raise ValueError("matrices must be unitary")

        #if(not numpy.isclose(numpy.linalg.det(arr), 1).all()):
        #    print(numpy.linalg.det(arr))
        #    print(numpy.isclose(numpy.linalg.det(arr), 1))
        #    raise ValueError("matrices must be in SU(N)")

        if(arr.shape[0] != 2):
            raise ValueError("matrices must be in SU(2)")

        arr = arr.astype(numpy.cdouble)

        BaseGate.__init__(self, _GenericGate(act, arr[0,0], arr[1,1], arr[0,1], arr[1,0]))
        self._array = arr

    def get_dagger(self):
        return GenericGate(self._act, self._array.transpose().conjugate())
