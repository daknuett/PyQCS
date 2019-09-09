from abc import abstractmethod, ABCMeta

class AbstractCircuitBuilder(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args):
        pass

