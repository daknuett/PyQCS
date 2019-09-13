from numpy import random

from ..gates.executor import GateListExecutor, RepeatingGateListExecutorSpawner, RepeatingGateListExecutor
from ..gates.circuits import AbstractNamedGateCircuit

class NoisyGateListExecutorSpawner(object):
    def __init__(self, noise_channels_by_gate_name, rng=random.uniform):
        self._noise_channel_by_gate_name = noise_channels_by_gate_name
        self._rng = rng

    def __call__(self, gate_or_executor):
        return NoisyGateListExecutor(self._noise_channel_by_gate_name, self._rng, gate_or_executor)


class NoisyGateListExecutor(GateListExecutor):
    def __init__(self, noise_channels_by_gate_name, rng, gate_or_executor):
        GateListExecutor.__init__(self, gate_or_executor)
        self._noise_channel_by_gate_name = noise_channels_by_gate_name
        self._rng = rng

    def __call__(self, state):
        for gate_or_executor in self._gate_and_executor_list:
            if(isinstance(gate_or_executor, GateListExecutor)):
                state = noisyfy( gate_or_executor
                                , self._noise_channel_by_gate_name
                                , self._rng)(state)
            else:
                state = state.apply_gate(gate_or_executor)
                if(isinstance(gate_or_executor, AbstractNamedGateCircuit)
                        and gate_or_executor._name in self._noise_channel_by_gate_name):
                state = self._noise_channel_by_gate_name[gate_or_executor._name](state, rng())

        return state

class NoisyRepeatingGateListExecutorSpawner(
                    RepeatingGateListExecutorSpawner
                    , NoisyGateListExecutorSpawner):
    def __init__(self, times, noise_channels_by_gate_name, rng=random.uniform):
        RepeatingGateListExecutorSpawner.__init__(self, times)
        NoisyGateListExecutorSpawner.__init__(self, noise_channels_by_gate_name, rng)

    def __call__(self, gate_or_executor):
        return NoisyRepeatingGateListExecutor(
                    times
                    , self._noise_channel_by_gate_name
                    , self._rng
                    , gate_or_executor)

class NoisyGateListExecutor(
                    NoisyGateListExecutor
                    , RepeatingGateListExecutor):
    def __init__( self
                , times
                , noise_channels_by_gate_name
                , rng
                , gate_or_executor):
        NoisyGateListExecutor.__init__(
                                        self
                                        , noise_channels_by_gate_name
                                        , rng
                                        , gate_or_executor)
        RepeatingGateListExecutor.__init__(self, times, gate_or_executor)

    def __call__(self, state):
        for _ in range(self._times):
            for gate_or_executor in self._gate_and_executor_list:
                if(isinstance(gate_or_executor, GateListExecutor)):
                    state = noisyfy( gate_or_executor
                                    , self._noise_channel_by_gate_name
                                    , self._rng)(state)
                else:
                    state = state.apply_gate(gate_or_executor)
                    if(isinstance(gate_or_executor, AbstractNamedGateCircuit)
                            and gate_or_executor._name in self._noise_channel_by_gate_name):
                    state = self._noise_channel_by_gate_name[gate_or_executor._name](state, rng)

        return state


def noisify(executor, noise_channels_by_gate_name, rng=random.uniform):
    if(isinstance(RepeatingGateListExecutor)):
        return NoisyRepeatingGateListExecutor(
                    executor._times
                    , noise_channels_by_gate_name
                    , rng
                    , executor._gate_and_executor_list)
    if(isinstance(GateListExecutor)):
        return NoisyGateListExecutor(
                    noise_channels_by_gate_name
                    , rng
                    , executor._gate_and_executor_list)

