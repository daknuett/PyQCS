from .circuits import Circuit
from .gate import Gate, Capabilities
from .exceptions import UnitarityError


def H(act):
    return Circuit(Capabilities.clifford()
                    , [Gate(act, None, None, "H", Capabilities.clifford())]
                    , 1 << act)


def Z(act):
    return Circuit(Capabilities.clifford()
                    , [Gate(act, None, None, "Z", Capabilities.clifford())]
                    , 1 << act)


def X(act):
    return Circuit(Capabilities.clifford()
                    , [Gate(act, None, None, "X", Capabilities.clifford())]
                    , 1 << act)


def R(act, phi):
    def get_R_dagger(gate):
        return [Gate(act, None, -phi, "R", Capabilities.universal())]
    return Circuit(Capabilities.universal()
                    , [Gate(act, None, phi, "R", Capabilities.universal()
                            , get_R_dagger)]
                    , 1 << act)


def CZ(act, control):
    return Circuit(Capabilities.clifford()
                    , [Gate(act, control, None, "CZ", Capabilities.clifford())]
                    , (1 << act) | (1 << control))


def CX(act, control):
    return Circuit(Capabilities.clifford()
                    , [Gate(act, control, None, "CX", Capabilities.clifford())]
                    , (1 << act) | (1 << control))


def S(act):
    def get_S_dagger(gate):
        return [Gate(act, None, None, "Z", Capabilities.clifford())
                , Gate(act, None, None, "S", Capabilities.clifford())]
    return Circuit(Capabilities.clifford()
                    , [Gate(act, None, None, "S", Capabilities.clifford()
                            , get_S_dagger)]
                    , 1 << act)


def M(act):
    def get_M_dagger(gate):
        raise UnitarityError("M is not unitary, cannot be daggered")
    return Circuit(Capabilities.clifford()
                    , [Gate(act, None, None, "M", Capabilities.clifford()
                            , get_M_dagger)]
                    , 1 << act)
