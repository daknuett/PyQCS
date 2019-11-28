class GateListExecutor(object):
    def __init__(self, gate_list):
        self._can_flatten = True
        self._gate_and_executor_list = gate_list
        self.flatten()

    def __call__(self, state):
        for gate_or_executor in self._gate_and_executor_list:
            if(isinstance(gate_or_executor, GateListExecutor)):
                state = gate_or_executor(state)
            else:
                state = state.apply_gate(gate_or_executor)
        return state

    def flatten_generator(self):
        if(not self._can_flatten):
            self.flatten()
            yield self
            return
        for gate_or_executor in self._gate_and_executor_list:
            if(not isinstance(gate_or_executor, GateListExecutor)):
                yield gate_or_executor
            else:
                yield from gate_or_executor.flatten_generator()

    def __inline_flatten_generator(self):
        for gate_or_executor in self._gate_and_executor_list:
            if(not isinstance(gate_or_executor, GateListExecutor)):
                yield gate_or_executor
            else:
                yield from gate_or_executor.flatten_generator()

    def flatten(self):
        self._gate_and_executor_list = [i for i in self.__inline_flatten_generator()]

    def to_gate_list(self):
        return list(self.to_gate_list_generator())

    def to_gate_list_generator(self):
        for gate_or_executor in self._gate_and_executor_list:
            if(not isinstance(gate_or_executor, GateListExecutor)):
                yield gate_or_executor
            else:
                yield from gate_or_executor.to_gate_list_generator()


class RepeatingGateListExecutorSpawner(object):
    def __init__(self, times):
        self._times = times
    def __call__(self, gate_or_executor):
        return RepeatingGateListExecutor(self._times, gate_or_executor)

class RepeatingGateListExecutor(GateListExecutor):
    def __init__(self, times, gate_or_executor):
        self._gate_and_executor_list = gate_or_executor
        self._times = times
        self._can_flatten = False

    def __call__(self, state):
        for _ in range(self._times):
            for gate_or_executor in self._gate_and_executor_list:
                if(isinstance(gate_or_executor, GateListExecutor)):
                    state = gate_or_executor(state)
                else:
                    state = state.apply_gate(gate)
        return state
    def to_gate_list_generator(self):
        for _ in range(self._times):
            for gate_or_executor in self._gate_and_executor_list:
                if(not isinstance(gate_or_executor, GateListExecutor)):
                    yield gate_or_executor
                else:
                    yield from gate_or_executor.to_gate_list_generator()
