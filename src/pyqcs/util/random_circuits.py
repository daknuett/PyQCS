import numpy as np
from .. import X, H, R, CX, list_to_circuit


"""
Generate random circuits. Refer to ``random_circuit`` for details.
``random_circuit_XHRC`` uses ``M1=X``, ``M2=H``, ``M3=R``, ``M4=CX``.
"""


def random_basis_values(nqbits, ngates):
    """
    Sample ngates values from {1,...,4 nqbits} x {1,...,nqbits - 1} x [0, 2pi).
    """
    gate_index = np.random.randint(1, high=(nqbits * 4), size=ngates)
    entanglement_index = np.random.randint(1, high=(nqbits - 1), size=ngates)
    phi = np.random.uniform(low=0.0, high=(2 * np.pi), size=ngates)

    return (gate_index, entanglement_index, phi)


def _random_gates(nqbits, ngates, M1, M2, M3, M4):
    for i, k, x in zip(*random_basis_values(nqbits, ngates)):
        if(i <= nqbits):
            yield M1(i - 1)
        elif(i <= 2*nqbits):
            yield M2(i - nqbits - 1)
        elif(i <= 3*nqbits):
            yield M3(i - 2*nqbits - 1, x)

        else:
            if(k <= i - 3*nqbits - 1):
                k -= 1
            yield M4(i - 3*nqbits - 1, k)


def random_circuit(nqbits, ngates, M1, M2, M3, M4):
    """
    Sample ngates gates from (M1, M2, M3, M4) where Mi act on one of the nqbit
    qbits, M1, M2 take no extra arguments (typically M1=X, M2=H), M3 takes a
    float argument (typically M3=R_phi) and M4 is an entanglement gate (typically CX or CZ).

    M1, M2, M3, M4 are equally likely, all qbits are equally likely to be modified.
    Both x for M3 and k for M4 are uniformely distributed.
    """
    return list_to_circuit(list(_random_gates(nqbits, ngates, M1, M2, M3, M4)))


def random_circuit_XHRC(nqbits, ngates):
    return random_circuit(nqbits, ngates, X, H, R, CX)
