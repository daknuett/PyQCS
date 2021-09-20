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
    "CX": lambda c, r: "X"
    , "CZ": lambda c, r: "Z"
    , "R": lambda c, r: r"R_{%.2f\pi}" % (r / pi)
    , "X": lambda c, r: "X"
    , "H": lambda c, r: "H"
    , "Z": lambda c, r: "Z"
    , "M": not_implemented
}

def circuit_to_table(circuit):
    nbits = int(circuit._requires_qbits).bit_length()

    gates = circuit._gate_list
    raise NotImplementedError()


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
            subprocess.run([self.pdflatex] + self.pdflatex_args, cwd=tmpdirname)
            if(not os.path.isfile(tmpdirname + "/main.pdf")):
                raise OSError(f"pdflatex ({self.pdflatex}) did not produce a pdf file")
            subprocess.run([self.convert] + self.convert_args
                            , cwd=tmpdirname)
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
