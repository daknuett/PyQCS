import numpy as np
import pytest

from pyqcs.util.random_graphs import random_graph_lists
from pyqcs.util.from_lists import graph_lists_to_graph_state
from pyqcs import CZ

def do_test5_5():
    lists = random_graph_lists(5, 5)
    g = graph_lists_to_graph_state(lists)
    n =  CZ(0, 1) * g.to_naive_state()
    g = CZ(0, 1) * g

    if(g.to_naive_state() @ n != pytest.approx(1)):
        print("before:", lists)
        print("got:", g._g_state.to_lists())

        print()

        print("expected")
        print(n)
        print("got")
        print(g.to_naive_state())

    assert g.to_naive_state() == n

@pytest.mark.slow
def test_random_graph5_5():
    np.random.seed(1)
    for _ in range(1000):
        do_test5_5()
