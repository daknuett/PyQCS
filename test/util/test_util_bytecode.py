from pyqcs.util.bytecode import ByteCodeHeader, ByteCodeInstruction, circuit_to_byte_code
from pyqcs import H, S, CZ, M

def test_header_prewritten():
    header = ByteCodeHeader(5, 0b11111, 100)

    bc = header.to_bytes()

    assert bc == (b"GQCS"
                    b"\x05\x00\x00\x00\x00\x00\x00\x00b"
                    b"\x64\x00s"
                    b"\x05\x00\x00\x00\x00\x00\x00\x00q"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\x02\x00\x00\x00\x00\x00\x00\x00"
                    b"\x03\x00\x00\x00\x00\x00\x00\x00"
                    b"\x04\x00\x00\x00\x00\x00\x00\x00"
                    b"\xff\xff\xff\xff\xff\xff\xff\xff"
                    )

def test_instruction_CZ_prewritten():
    instr = ByteCodeInstruction("CZ", 2, 1)

    bc = instr.to_bytes()

    assert bc == (b"Z"
                    b"\x02\x00\x00\x00\x00\x00\x00\x00"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00")

def test_instruction_H_prewritten():
    instr = ByteCodeInstruction("C_L", 2, 0)

    bc = instr.to_bytes()

    assert bc == (b"L"
                    b"\x02\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00")

def test_instruction_M_prewritten():
    instr = ByteCodeInstruction("M", 2, 0)

    bc = instr.to_bytes()

    assert bc == (b"M"
                    b"\x02\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00")


def test_circuit_prewritten():
    circuit = H(0) | H(1) | CZ(1, 0) | S(1) | H(1) | M(0)
    header = ByteCodeHeader(2, 0b10, 100)

    bc = circuit_to_byte_code(header, circuit)

    assert bc == (b"GQCS"
                    b"\x02\x00\x00\x00\x00\x00\x00\x00b"
                    b"\x64\x00s"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00q"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\xff\xff\xff\xff\xff\xff\xff\xff"
                    b"L"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"L"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"Z"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"L"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"L"
                    b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"M"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00"
                    )
