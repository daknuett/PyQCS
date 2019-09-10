from abc import ( ABCMeta
        , abstractclassmethod
        , abstractmethod
        , abstractstaticmethod)

class AbstractState(metaclass=ABCMeta):
    @abstractmethod
    def get_last_measurement(self):
        pass
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
    def deepcopy(self):
        pass

    @abstractclassmethod
    def new_zero_state(cls, nbits, **kwargs):
        pass
