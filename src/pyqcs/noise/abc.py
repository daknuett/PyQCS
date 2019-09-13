from abc import ABCMeta, abstractmethod


class AbstractNoiseChannel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state, rng):
        pass

