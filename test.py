from itertools import product
import subprocess
import numpy as np

def format_int_for_replace(i):
    if(i < 0):
        return f"- {-i}"
    return f"+ {i}"

def prepare_code(i, j):
    with open("src/pyqcs/graph/backend/graph_operations.c.tpl", "r") as fin:
        data = fin.read()

    data = data.replace("EXTRA_PHASE_A", format_int_for_replace(i))
    data = data.replace("EXTRA_PHASE_B", format_int_for_replace(j))

    with open("src/pyqcs/graph/backend/graph_operations.c", "w") as fout:
        fout.write(data)

possible_phases = np.arange(-8, 8, 1, dtype=np.int)

for i,j in product(possible_phases, possible_phases):
    print("RUNNING WITH", i, j)
    prepare_code(i, j)
    result = subprocess.run(["tox", "--", "--onlyselected"])
    if(result.returncode == 0):
        print("FOUND:", i, j)
        break
print("NO PHASES FOUND")
