import pytest
from devograph.base_graph import BaseGraph


def test_graph_basic():
    g = BaseGraph()
    g.add_node(1)
    g.add_node(2)
    g.add_edge([1, 2])

    assert g.num_nodes() == 2
    assert g.num_edges() == 1


def test_invalid_edge():
    g = BaseGraph()
    g.add_node(1)

    with pytest.raises(ValueError):
        g.add_edge([1])
