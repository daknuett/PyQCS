import numpy
from ..state.abc import AbstractState

class GraphState(AbstractState):
    def __init__(self, vertices, edges, cl_state):
        self._vertices = vertices
        self._edges = edges
        self._cl_state = cl_state

    @classmethod
    def new_zero_state(cls, nbits):
        vertices = numpy.array(dtype=numpy.intp)
        edges = numpy.zeros((nbits, 5), dtype=numpy.uint8)
        cl_state = -1 * numpy.ones(nbits, dtype=numpy.int8)

        # We associate sqrt(-iX) with 1, sqrt(iZ) with 2.
        # This is the initial H gate.
        edges[:, 0] = 1
        edges[:, 1] = 2
        edges[:, 2] = 2
        edges[:, 3] = 2
        edges[:, 4] = 1

        return cls(vertices, edges, cl_state)

    def apply_gate(self, gate):
        new_vertices, new_edges new_cl_state= gate(self._vertices, self._edges, self._cl_state)
        return GraphState(new_vertices, new_edges, new_cl_state)
