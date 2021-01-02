from abc import ABCMeta, abstractmethod
import logging

logger = logging.getLogger(__name__)

try:
    import ray
except ImportError:
    raise ImportError("pyqcs experiments require ray for parallelization; install it using pip3 install ray")


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


@ray.remote
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

class WorkflowSpawner(object):
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
    def spawn(self):
        return Workflow.remote(self.name, self.instructions)


