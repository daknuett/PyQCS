
class Capabilities(object):
    __slots__ = [
        "_capability_names"
        , "_bitmask"
    ]

    def __init__(self, names, bitmask):
        self._capability_names = names
        self._bitmask = bitmask

    @classmethod
    def clifford(cls):
        return cls(["clifford"], 0b01)

    @classmethod
    def universal(cls):
        return cls(["clifford", "universal"], 0b11)

    def __str__(self):
        return (f"[capabilities: {', '.join(self._capability_names)}"
                f", level: {self._bitmask}]")

    def __le__(self, other):
        if(not isinstance(other, Capabilities)):
            raise TypeError()
        return self._bitmask <= other._bitmask


def max_capabilities(c1, c2):
    if(not (isinstance(c1, Capabilities) and isinstance(c2, Capabilities))):
        raise TypeError("max_capabilities only supported for class Capabilities")

    if(c1._bitmask > c2._bitmask):
        return c1
    return c2


class Gate(object):
    __slots__ = [
        "_act"
        , "_control"
        , "_phi"
        , "_name"
        , "_requires_capabilities"
        , "_is_Z2"
        , "_adjoint_recipe"
    ]

    def __init__(self, act, control, phi, name, requires_capabilities, adjoint_recipe=None):
        self._act = act
        self._control = control
        self._phi = phi
        self._name = name
        self._requires_capabilities = requires_capabilities

        if(adjoint_recipe is None):
            self._is_Z2 = True
            self._adjoint_recipe = None
        else:
            self._is_Z2 = False
            self._adjoint_recipe = adjoint_recipe

    def get_dagger(self):
        if(self._is_Z2):
            return [self]
        return self._adjoint_recipe(self)
