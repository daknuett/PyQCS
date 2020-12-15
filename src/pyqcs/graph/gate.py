from abc import abstractmethod, ABCMeta
from .backend.raw_state import RawGraphState
from numpy.random import uniform

class AbstractGraphOperation(metaclass=ABCMeta):
    def __init__(self, act):
        self._act = act

    @abstractmethod
    def apply_to_raw_state(self, state):
        pass

class CZOperation(AbstractGraphOperation):
    def __init__(self, act, control, dagger):
        AbstractGraphOperation.__init__(self, act)
        self._control = control
        self._dagger = dagger

    def apply_to_raw_state(self, state):
        if(not isinstance(state, RawGraphState)):
            raise TypeError("state must be of type RawGraphState, but got {}".format(str(type(state))))

        state.apply_CZ(self._act, self._control, self._dagger)
        return False, None

class CLOperation(AbstractGraphOperation):
    def __init__(self, act, clifford_index, phase):
        AbstractGraphOperation.__init__(self, act)
        self._phase = phase

        if(not isinstance(clifford_index, int)):
            raise TypeError("clifford_index must be integer")
        if(0 > clifford_index > 23):
            raise ValueError("clifford_index must be between 0 and 23")
        self._clifford_index = clifford_index

    def apply_to_raw_state(self, state):
        if(not isinstance(state, RawGraphState)):
            raise TypeError("state must be of type RawGraphState, but got {}".format(str(type(state))))

        state.apply_C_L(self._act, self._clifford_index, self._phase)
        return False, None

class MeasurementOperation(AbstractGraphOperation):
    def __init__(self, act):
        AbstractGraphOperation.__init__(self, act)

    def apply_to_raw_state(self, state):
        if(not isinstance(state, RawGraphState)):
            raise TypeError("state must be of type RawGraphState, but got {}".format(str(type(state))))

        rnd = uniform()
        result = state.measure(self._act, rnd)

        return True, (self._act, result)


class GraphGate(object):
    def __init__(self, operation_list):
        self._operation_list = operation_list
