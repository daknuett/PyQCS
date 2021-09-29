import os
import subprocess
import shutil
from collections import deque
from tempfile import TemporaryDirectory

from numpy import pi


def not_implemented(*args):
    raise NotImplementedError()


multi_qbit_gates = ["CX", "CZ"]
formatters = {
    "CX": lambda r: "X"
    , "CZ": lambda r: "Z"
    , "R": lambda r: r"R_{%.2f\pi}" % (r / pi)
    , "X": lambda r: "X"
    , "H": lambda r: "H"
    , "Z": lambda r: "Z"
    , "S": lambda r: "S"
    , "M": not_implemented
}


def circuit_to_diagram(circuit):
    table = [deque() for _ in range(circuit._requires_qbits.bit_length())]

    for gate in circuit._gate_list:
        if(gate._name in multi_qbit_gates):
            # fill the table with qwires so we get the same
            # width for control and act.
            if(len(table[gate._act]) < len(table[gate._control])):
                table[gate._act].extend(
                    [r"& \qw"] * (len(table[gate._control])
                                    - len(table[gate._act]))
                )
            if(len(table[gate._act]) > len(table[gate._control])):
                table[gate._control].extend(
                    [r"& \qw"] * (len(table[gate._act])
                                   - len(table[gate._control]))
                )
            table[gate._control].append(
                r"& \ctrl{%d}"
                % (gate._act - gate._control)
            )

        table[gate._act].append(
            r"& \gate{%s}"
            % formatters[gate._name](gate._phi)
        )

    maxlen = max((len(row) for row in table))
    for row in table:
        if(len(row) < maxlen):
            row.extend([r"& \qw"] * (maxlen - len(row)))
        row.append(r"& \qw \\")

    inner_tex = "\n".join((" ".join(row) for row in table))
    tex = (r"\Qcircuit @C=1em @R=.7em {" + "\n"
            + inner_tex + "\n"
            + r"}")

    return tex


class CircuitPNGFormatter(object):
    def __init__(self, circuit
                    , pdflatex="xelatex"
                    , convert="convert"
                    , pdflatex_args=["main.tex"]
                    , convert_args=["-profile", "\"icc\""
                                , "-density", "300"
                                , "main.pdf"
                                , "-quality", "90"
                                , "main.png"]
                    ):
        self.tex = circuit_to_diagram(circuit)
        self.pdflatex = pdflatex
        self.convert = convert
        self.convert_args = convert_args
        self.pdflatex_args = pdflatex_args

        if(shutil.which(self.pdflatex) is None):
            raise OSError(f"pdflatex ({self.pdflatex}) not found, set it using ``pdflatex=<program>``")
        if(shutil.which(self.convert) is None):
            raise OSError(f"imagemagick ({self.convert}) not found, set it using ``convert=<program>``")

    def _repr_png_(self):
        with TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/main.tex", "w") as fout:
                fout.write(self.get_tex_file_content())
            subprocess.run([self.pdflatex] + self.pdflatex_args
                            , cwd=tmpdirname
                            , stdout=subprocess.PIPE
                            , stderr=subprocess.PIPE
            )
            if(not os.path.isfile(tmpdirname + "/main.pdf")):
                raise OSError(f"pdflatex ({self.pdflatex}) did not produce a pdf file")
            subprocess.run([self.convert] + self.convert_args
                            , cwd=tmpdirname
                            , stdout=subprocess.PIPE
                            , stderr=subprocess.PIPE)
            if(not os.path.isfile(tmpdirname + "/main.png")):
                raise OSError(f"imagemagick ({self.convert}) did not produce a png file")
            with open(tmpdirname + "/main.png", "rb") as fin:
                return fin.read()

    def get_tex_file_content(self):
        header = r'''
        \documentclass[preview]{standalone}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}

        \usepackage{qcircuit}

        \title{Drawing Circuits with qcircuit}

        \begin{document}
        '''
        bottom = r'''

        \end{document}

        '''

        return header + self.tex + bottom
