from .abc import AbstractNoiseChannel

class CompoundNoiseChannel(AbstractNoiseChannel):
    def __init__(self, sub_channels):
        self._subchannels = sub_channels

    def __call__(self, state, rng):
        for channel in self._subchannels:
            state = channel(state, rng)
        return state

class SimpleNoiseChannel(AbstractNoiseChannel):
    def __init__(self, gate_list, probability):
        self._gate_list = gate_list
        self._probability = probability

    def __call__(self, state, rng):
        if(rng() > self._probability):
            return state

        for gate in self._gate_list:
            state = state.apply_gate(gate)

        return state
