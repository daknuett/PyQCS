import struct
from collections import deque

from ..gates.gate import Capabilities

_local_clifford_gates = ["H", "S", "X", "Z"]
_CZ_gates = ["CZ"]
_decomposable_gates = ["CX"]
_measurement_gates = ["M"]

_local_clifford_gate_vop = {"H": 0, "S": 1, "X": 14, "Z": 5}
_gate_decompositions = {"CX": (("H", 0), ("CZ", 0, 1), ("H", 0))}


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


def clifford2i(gate):
    return [ByteCodeInstruction("C_L", gate._act, _local_clifford_gate_vop[gate._name])]


def cz2i(gate):
    return [ByteCodeInstruction("CZ", gate._act, gate._control)]


def measure2i(gate):
    return [ByteCodeInstruction("M", gate._act, 0)]


def decomposable2i(gate):
    args = (gate._act, gate._control)
    result = []
    for word in _gate_decompositions[gate._name]:
        if(len(word) == 2):
            decomp = ByteCodeInstruction("C_L", args[word[1]], _local_clifford_gate_vop[word[0]])
        else:
            decomp = ByteCodeInstruction(gate._name, args[word[1]], args[word[2]])
        result.append(decomp)
    return result


def circuit_to_byte_code(header, circuit, add_header=True):
    if(not (circuit._requires_capabilities <= Capabilities.clifford())):
        raise TypeError("Circuit must be applicable to graph.")
    if(circuit._requires_qbits >= (1 << header._nqbits)):
        raise ValueError("Circuit requires more qbits.")

    gates = circuit._gate_list

    o2i = {"CZ": cz2i, "M": measure2i, "CX": decomposable2i}
    o2i.update({gn: clifford2i for gn in _local_clifford_gates})

    instructions = deque()
    for gate in gates:
        instructions.extend(o2i[gate._name](gate))

    if(add_header):
        result = header.to_bytes()
    else:
        result = b""

    result += b"".join((i.to_bytes() for i in instructions))
    return result
