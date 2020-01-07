from pyqcs import H, Z, X, CX, CZ, S, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_diagram import circuit_to_diagram
from pyqcs.util.to_circuit import graph_state_to_circuit

fragments = [
    CZ(7, 2) | S(8) | Z(9)
    , CZ(2, 1) | Z(4) | S(5) | Z(6) | X(9)
    , CX(3, 0) | H(9)
    , H(0) | Z(1) | CZ(2, 5) | CZ(8, 3) | H(9)
    , X(0) | H(2) | H(5) | H(8) | S(9)
    , S(0) | CZ(1, 2) | CZ(3, 7)
    , X(1) | S(3) | X(7) | CZ(3, 2)
    , CZ(2, 0) | CZ(6, 3)
    , X(3) | CZ(6, 7)
    , CZ(4, 2) | H(6) | Z(7)
    , CZ(2, 8) | H(2) | X(8) | X(7) | CZ(6, 3)
    , CZ(4, 0) | S(6)
    , H(0) | CZ(2, 4) | X(6)
    , Z(0) | S(2) | X(3) | S(4)
    , CZ(5, 0)
    , X(0) | X(2) | Z(3) | CZ(6, 5)
    , S(0) | X(0) | H(0)
    , H(2) | S(3) | CZ(4, 8) | S(5) | H(6) | H(8)
    , CZ(4, 8)
    , CZ(6, 1) | X(4) | Z(8)
    , S(1) | CZ(2, 7)

    ]

def test_fragments():
    s = State.new_zero_state(10)
    g = GraphState.new_zero_state(10)

    for fragment in fragments:
        s = fragment * s
        print("g", g._g_state.to_lists())
        g_bar = fragment * g.deepcopy()
        print("g_bar", g_bar._g_state.to_lists())

        if(g_bar.to_naive_state() != s):
            print("failure occured in")
            print(circuit_to_diagram(fragment))
            print("state before is")
            print(circuit_to_diagram(graph_state_to_circuit(g)))
            print(g._g_state.to_lists())
            print(g.to_naive_state())
            print("state after is")
            print(circuit_to_diagram(graph_state_to_circuit(g_bar)))
            print(g_bar._g_state.to_lists())
            print(g_bar.to_naive_state())
            print()
            print("expected")
            print(s)
            print()
            print("overlap is: ", s @ g_bar.to_naive_state())

        g = g_bar
        assert g_bar.to_naive_state() == s

if(__name__ == "__main__"):
    test_fragments()
