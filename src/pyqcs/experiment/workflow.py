from abc import ABCMeta, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Instruction(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, params):
        pass

class FunctionInstruction(Instruction):
    def __init__(self, name, fct):
        self.name = name
        self.fct = fct
    def __call__(self, params):
        logger.info(f"executing instruction {self.name}")
        return self.fct(params)


class Workflow(object):
    def __init__(self, name, instructions):
        self.name = name
        self._instructions = instructions

    def execute(self, params):
        logger.info(f"executing workflow {self.name}")
        state = params
        for i,instr in enumerate(self._instructions):
            logger.info(f"workflow {self.name}: instruction {i}")
            state = instr(state)
        return state

