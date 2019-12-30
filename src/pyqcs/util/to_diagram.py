from collections import deque
from numpy import pi
from .flatten import flatten

def not_implemented(*args):
    raise NotImplementedError()


multi_qbit_gates = ["C", "B"]
formatters = {
    "C": lambda c, r: "X"
    , "B": lambda c, r: "Z"
    , "R": lambda c, r: "R_{%.2f\pi}" % (r / pi)
    , "X": lambda c, r: "X"
    , "H": lambda c, r: "H"
    , "Z": lambda c, r: "Z"
    , "GenericGate": not_implemented
    , "M": not_implemented
}

def circuit_to_table(circuit):
    nbits = int(circuit._uses_qbits).bit_length()

    circuits = flatten(circuit)

    rows = [deque() for _ in range(nbits)]
    last = deque()
    for sgc in circuits:
        if(not sgc._descr[0] in multi_qbit_gates):
            can_insert_in_row = True
            for l in last:
                if(l._descr[1] == sgc._descr[1]):
                    can_insert_in_row = False
                    break

            if(can_insert_in_row):
                rows[sgc._descr[1]].append(sgc._descr)
                last.append(sgc)
            else:
                for i,row in enumerate(rows):
                    if(not i in [l._descr[1] for l in last]):
                        row.append(None)
                rows[sgc._descr[1]].append(sgc._descr)
                last = deque([sgc])

        else:
            for i,row in enumerate(rows):
                if(not i in [l._descr[1] for l in last]):
                    row.append(None)
            rows[sgc._descr[1]].append(sgc._descr)
            for i,row in enumerate(rows):
                if(i != sgc._descr[1]):
                    row.append(None)
            last = deque()

    if(last):
        for i,row in enumerate(rows):
            if(not i in [l._descr[1] for l in last]):
                row.append(None)

    return [list(r) for r in rows]


def circuit_to_diagram(circuit):
    table = circuit_to_table(circuit)

    for i,row in enumerate(table):
        for j,format_descriptor in enumerate(row):
            if(format_descriptor is None and table[i][j] is None):
                table[i][j] = r"& \qw"
            if(isinstance(format_descriptor, tuple)):
                if(format_descriptor[0] in multi_qbit_gates):
                    table[format_descriptor[2]][j] = r"& \ctrl{" + str(i - format_descriptor[2]) + r"}"
                table[i][j] = r"& \gate{" + formatters[format_descriptor[0]](*format_descriptor[2:]) + r"}"

    for row in table:
        row.append(r"&\qw")
        row.append(r"\\")

    inner_tex = "\n".join((" ".join(row) for row in table))
    tex = (r"\Qcircuit @C=1em @R=.7em {" + "\n"
            + inner_tex + "\n"
            + r"}")

    return tex

