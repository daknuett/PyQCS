class GateListExecutor(object):
    def __init__(self, gate_list):
        self._gate_list = gate_list

    def __call__(self, state):
        for gate in self._gate_list:
            state = state.apply_gate(gate)
        return state

