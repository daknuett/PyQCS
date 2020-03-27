from abc import abstractmethod, ABCMeta

class AbstractCircuitBuilder(metaclass=ABCMeta):
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, *args):
        pass
class AbstractSingleGateCircuitBuilder(AbstractCircuitBuilder):
    def __init__(self):
        self._registry = dict()
    def clear_gate_registry(self):
        self._registry = dict()
