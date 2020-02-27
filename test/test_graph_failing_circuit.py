from pyqcs import H, Z, X, CX, CZ, S, State
from pyqcs.graph.state import GraphState
from pyqcs.util.to_diagram import circuit_to_diagram
from pyqcs.util.to_circuit import graph_state_to_circuit
import numpy as np

fragments_long = [
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

def test_fragments_long():
    s = State.new_zero_state(10)
    g = GraphState.new_zero_state(10)

    for fragment in fragments_long:
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
            q1 = g_bar.to_naive_state()._qm_state
            q2 = s._qm_state * 1j
            print(np.linalg.norm(q1 - q2))

            print()
            print("overlap is: ", s @ g_bar.to_naive_state())
            print("absolute overlap squared is: ", np.absolute(s @ g_bar.to_naive_state())**2)
            print()
            sg = graph_state_to_circuit(g_bar) * State.new_zero_state(10)
            print(sg == s)
            print(sg == g.to_naive_state())
            print(sg @ s)
            print(sg @ g.to_naive_state())


        g = g_bar
        assert g_bar.to_naive_state() == s

fragments_short = [
        H(1) | H(2) | H(3)
        , X(1)
        , CZ(2, 1)
        , S(1) | H(1) | H(2) | S(3)
        , CZ(1, 0)
        ]
def test_fragments_short():
    s = State.new_zero_state(4)
    g = GraphState.new_zero_state(4)

    for fragment in fragments_short:
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
            q1 = g_bar.to_naive_state()._qm_state
            q2 = s._qm_state * 1j
            print(np.linalg.norm(q1 - q2))

            print()
            print("overlap is: ", s @ g_bar.to_naive_state())
            print("absolute overlap squared is: ", np.absolute(s @ g_bar.to_naive_state())**2)
            print()
            sg = graph_state_to_circuit(g_bar) * State.new_zero_state(4)
            print("sg == s", sg == s)
            print("sg == g", sg == g.to_naive_state())
            print("sg @ s", sg @ s)
            print("sg @ g", sg @ g.to_naive_state())


        g = g_bar
        assert g_bar.to_naive_state() == s

def test_qbit_gets_isolated():
    g = GraphState.new_plus_state(4)
    s = (H(0) | H(1) | H(2) | H(3)) * State.new_zero_state(4)
    circuit = (CZ(0, 1) | CZ(1, 3) | CZ(3, 2) | CZ(0, 3) | CZ(1, 2))
    g = circuit * g
    s = circuit * s
    circuit = S(1) | S(3) | H(1) | H(3)
    g = circuit * g
    s = circuit * s

    print(g._g_state.to_lists())
    g = CZ(1, 3) * g
    s = CZ(1, 3) * s
    print(g._g_state.to_lists())

    assert g.to_naive_state() == s

def test_b_picks_up_neighbour():
    g = GraphState.new_plus_state(3)
    g = (CZ(0, 1) | CZ(1, 2)) * g
    g._g_state.apply_C_L(1, 12)
    s = g.to_naive_state()

    s = CZ(0, 1) * s
    g = CZ(0, 1) * g

    assert s == g.to_naive_state()

if(__name__ == "__main__"):
    test_fragments_long()
