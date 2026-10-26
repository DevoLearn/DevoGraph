import pytest
import numpy as np
from devograph.base_graph import BaseGraph


def test_graph_basic():
    g = BaseGraph()
    g.add_node(1, type="cell", x=1.0)
    g.add_node(2, type="cell", x=2.0)
    g.add_edge([1, 2], weight=0.5)

    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    assert len(g) == 2
    assert repr(g) == "BaseGraph(num_nodes=2, num_edges=1)"
    assert g.has_node(1)
    assert not g.has_node(99)
    assert g.get_node(1) == {"type": "cell", "x": 1.0}
    assert g.get_nodes() == [1, 2]
    assert g.has_edge([1, 2])
    assert g.has_edge([2, 1])
    assert not g.has_edge([1, 3])


def test_invalid_edge():
    g = BaseGraph()
    g.add_node(1)

    with pytest.raises(ValueError):
        g.add_edge([1])

    with pytest.raises(ValueError):
        g.add_edge([1, 2])


def test_duplicate_node():
    g = BaseGraph()
    g.add_node(1)
    with pytest.raises(ValueError):
        g.add_node(1)


def test_neighbors_and_degree():
    g = BaseGraph()
    for i in range(1, 5):
        g.add_node(i)

    g.add_edge([1, 2])
    g.add_edge([1, 3, 4])  # hyperedge with 3 nodes

    assert g.neighbors(1) == {2, 3, 4}
    assert g.neighbors(2) == {1}
    assert g.neighbors(3) == {1, 4}
    assert g.degree(1) == 2
    assert g.degree(2) == 1
    assert g.degree(3) == 1
    assert g.incident_edges(1) == [0, 1]
    assert g.incident_edges(2) == [0]

    with pytest.raises(KeyError):
        g.neighbors(99)

    with pytest.raises(KeyError):
        g.degree(99)


def test_remove_node_and_edge():
    g = BaseGraph()
    for i in range(1, 4):
        g.add_node(i)
    g.add_edge([1, 2])
    g.add_edge([2, 3])

    g.remove_edge(0)
    assert g.num_edges() == 1
    assert g.get_edge(0)["nodes"] == [2, 3]

    with pytest.raises(IndexError):
        g.remove_edge(10)

    # Remove node 2 (should remove incident edge [2, 3])
    g.remove_node(2)
    assert g.num_nodes() == 2
    assert not g.has_node(2)
    assert g.num_edges() == 0

    with pytest.raises(KeyError):
        g.remove_node(99)


def test_to_incidence_matrix():
    g = BaseGraph()
    for i in range(1, 4):
        g.add_node(i)
    g.add_edge([1, 2])
    g.add_edge([1, 2, 3])  # hyperedge

    H = g.to_incidence_matrix()
    assert H.shape == (3, 2)
    # Edge 0 has nodes 1 and 2
    assert H[0, 0] == 1.0
    assert H[1, 0] == 1.0
    assert H[2, 0] == 0.0
    # Edge 1 has nodes 1, 2, 3
    assert H[0, 1] == 1.0
    assert H[1, 1] == 1.0
    assert H[2, 1] == 1.0

    # Empty graph
    empty_g = BaseGraph()
    assert empty_g.to_incidence_matrix().shape == (0, 0)


def test_to_adjacency_matrix():
    g = BaseGraph()
    for i in range(1, 4):
        g.add_node(i)
    g.add_edge([1, 2], weight=2.0)
    g.add_edge([2, 3], weight=1.5)

    A_binary = g.to_adjacency_matrix(weighted=False)
    assert A_binary.shape == (3, 3)
    assert A_binary[0, 1] == 1.0 and A_binary[1, 0] == 1.0
    assert A_binary[1, 2] == 1.0 and A_binary[2, 1] == 1.0
    assert A_binary[0, 2] == 0.0

    A_weighted = g.to_adjacency_matrix(weighted=True)
    assert A_weighted[0, 1] == 2.0
    assert A_weighted[1, 2] == 1.5


def test_to_edge_index():
    g = BaseGraph()
    for i in range(1, 4):
        g.add_node(i)
    g.add_edge([1, 2])

    edge_index = g.to_edge_index(bidirectional=True)
    assert edge_index.shape == (2, 2)
    assert list(edge_index[0]) == [0, 1]
    assert list(edge_index[1]) == [1, 0]

    edge_index_directed = g.to_edge_index(bidirectional=False)
    assert edge_index_directed.shape == (2, 1)

    empty_g = BaseGraph()
    assert empty_g.to_edge_index().shape == (2, 0)


def test_dict_serialization():
    g = BaseGraph()
    g.add_node(10, label="nucleus")
    g.add_node(20, label="membrane")
    g.add_edge([10, 20], relationship="contains")

    data = g.to_dict()
    assert "nodes" in data and "edges" in data
    assert data["nodes"]["10"]["label"] == "nucleus"

    reconstructed = BaseGraph.from_dict(data)
    assert reconstructed.num_nodes() == 2
    assert reconstructed.num_edges() == 1
    assert reconstructed.has_node(10)
    assert reconstructed.has_node(20)
    assert reconstructed.has_edge([10, 20])
    assert reconstructed.get_edge(0)["attrs"]["relationship"] == "contains"
