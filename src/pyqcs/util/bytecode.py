import struct
from collections import deque

from ..gates.builtins import CLOperation, CZOperation, MeasurementOperation


class ByteCodeHeader(object):
    def __init__(self, nqbits, sample, nsamples):

        if(isinstance(sample, int)):
            self._sample = [i for i in range(sample.bit_length()) if sample & (1 << i)]
            self._sample_qbit_count = len(self._sample)
        elif(isinstance(sample, (tuple, list))):
            for i in sample:
                if(not isinstance(i, int)):
                    raise TypeError("sample must be int or list of ints")
            self._sample = sample
            self._sample_qbit_count = len(self._sample)
        else:
            raise TypeError("sample must be int or list of ints")

        self._nsamples = nsamples
        self._nqbits = nqbits

        if(max(self._sample) >= self._nqbits):
            raise ValueError("one qbit to sample is out of qbit range")
        if(min(self._sample) < 0):
            raise ValueError("one qbit to sample is out of qbit range")

    def to_bytes(self):
        magic_word = b"GQCS"

        b = struct.pack("Q", self._nqbits)
        s = struct.pack("H", self._nsamples)
        q = struct.pack("Q", self._sample_qbit_count)

        samples = [struct.pack("Q", i) for i in self._sample]

        return magic_word + b + b"b" + s + b"s" + q + b"q" + b"".join(samples) + b"\xff"*8

class ByteCodeInstruction(object):
    cmds = {"M": b"M", "CZ": b"Z", "C_L": b"L"}

    def __init__(self, command, act, argument):
        self._command = self.cmds[command]
        self._act = act
        self._argument = argument

    def to_bytes(self):
        return self._command + struct.pack("Q", self._act) + struct.pack("Q", self._argument)

def circuit_to_byte_code(header, circuit, add_header=True):
    if(not circuit._has_graph):
        raise TypeError("Circuit must be applicable to graph.")
    if(circuit._uses_qbits >= (1 << header._nqbits)):
        raise ValueError("Circuit requires more qbits.")

    executor = circuit._executor(circuit.get_child_executors(graph=True))
    gates = executor.to_gate_list()

    def C_L_to_instruction(op):
        return ByteCodeInstruction("C_L", op._act, op._clifford_index)
    def CZ_to_instruction(op):
        return ByteCodeInstruction("CZ", op._act, op._control)
    def M_to_instruction(op):
        return ByteCodeInstruction("M", op._act, 0)

    o2i = {CLOperation: C_L_to_instruction
            , CZOperation: CZ_to_instruction
            , MeasurementOperation: M_to_instruction}

    instructions = deque()
    for gate in gates:
        for op in gate._operation_list:
            instructions.append(o2i[type(op)](op))

    if(add_header):
        result = header.to_bytes()
    else:
        result = b""

    result += b"".join((i.to_bytes() for i in instructions))
    return result
