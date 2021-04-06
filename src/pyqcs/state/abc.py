from abc import (ABCMeta
        , abstractclassmethod
        , abstractmethod)


class AbstractState(metaclass=ABCMeta):
    __slots__ = ["_is_graph", "_is_naive"]

    def __init__(self):
        self._is_graph = False
        self._is_naive = False

    @abstractmethod
    def apply_gate(self, gate, force_new_state=False):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def check_qbits(self, gate_circuit):
        pass

    @abstractmethod
    def deepcopy(self, **kwargs):
        pass

    @abstractclassmethod
    def new_zero_state(cls, nbits, **kwargs):
        pass

    @abstractmethod
    def is_normalized(self):
        pass

    @abstractmethod
    def redo_normalization(self):
        pass
