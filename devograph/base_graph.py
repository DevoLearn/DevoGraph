from typing import Dict, List, Any, Set, Optional
import itertools
import numpy as np


class BaseGraph:
    """
    Base graph class for DevoGraph.
    Supports normal pairwise edges and multi-node hyperedges.
    """

    def __init__(self):
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, node_id: int, **attrs):
        """Add a node to the graph with optional attributes."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        self.nodes[node_id] = attrs

    def has_node(self, node_id: int) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes

    def get_node(self, node_id: int) -> Dict[str, Any]:
        """Return attributes of the given node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist")
        return self.nodes[node_id]

    def get_nodes(self) -> List[int]:
        """Return a list of all node IDs."""
        return list(self.nodes.keys())

    def remove_node(self, node_id: int):
        """Remove a node and any incident edges/hyperedges."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist")
        del self.nodes[node_id]
        # Remove incident edges
        self.edges = [e for e in self.edges if node_id not in e["nodes"]]

    def add_edge(self, node_ids: List[int], **attrs):
        """Add a pairwise edge (2 nodes) or hyperedge (>2 nodes)."""
        if len(node_ids) < 2:
            raise ValueError("Edge must have at least two nodes")

        for nid in node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Node {nid} does not exist")

        self.edges.append({
            "nodes": list(node_ids),
            "attrs": attrs
        })

    def has_edge(self, node_ids: List[int]) -> bool:
        """Check if an edge with the exact set of nodes exists."""
        target_set = set(node_ids)
        for e in self.edges:
            if set(e["nodes"]) == target_set:
                return True
        return False

    def get_edge(self, index: int) -> Dict[str, Any]:
        """Get edge dictionary at the given index."""
        if index < 0 or index >= len(self.edges):
            raise IndexError("Edge index out of range")
        return self.edges[index]

    def get_edges(self) -> List[Dict[str, Any]]:
        """Return all edges in the graph."""
        return self.edges

    def remove_edge(self, index: int):
        """Remove an edge by index."""
        if index < 0 or index >= len(self.edges):
            raise IndexError("Edge index out of range")
        del self.edges[index]

    def neighbors(self, node_id: int) -> Set[int]:
        """Return the set of neighbor node IDs connected via edges or hyperedges."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist")

        nbrs = set()
        for e in self.edges:
            if node_id in e["nodes"]:
                for nid in e["nodes"]:
                    if nid != node_id:
                        nbrs.add(nid)
        return nbrs

    def degree(self, node_id: int) -> int:
        """Return the number of edges/hyperedges incident to node_id."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist")
        return sum(1 for e in self.edges if node_id in e["nodes"])

    def incident_edges(self, node_id: int) -> List[int]:
        """Return indices of all edges/hyperedges incident to node_id."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist")
        return [idx for idx, e in enumerate(self.edges) if node_id in e["nodes"]]

    def num_nodes(self) -> int:
        """Return the total number of nodes in the graph."""
        return len(self.nodes)

    def num_edges(self) -> int:
        """Return the total number of edges/hyperedges in the graph."""
        return len(self.edges)

    def to_incidence_matrix(self) -> np.ndarray:
        """
        Generate incidence matrix H of shape (num_nodes, num_edges).
        H[i, j] = 1 if node i is incident to edge j, otherwise 0.
        Commonly used in Hypergraph Neural Networks (HGNN / Hyper-SAGNN).
        """
        n = self.num_nodes()
        m = self.num_edges()
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float32)

        node_to_idx = {nid: idx for idx, nid in enumerate(self.nodes.keys())}
        H = np.zeros((n, m), dtype=np.float32)

        for edge_idx, edge in enumerate(self.edges):
            for nid in edge["nodes"]:
                H[node_to_idx[nid], edge_idx] = 1.0

        return H

    def to_adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """
        Generate adjacency matrix A of shape (num_nodes, num_nodes).
        For pairwise edges, A[u, v] = 1.
        For hyperedges of size > 2, uses clique expansion to connect all node pairs.
        """
        n = self.num_nodes()
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        node_to_idx = {nid: idx for idx, nid in enumerate(self.nodes.keys())}
        A = np.zeros((n, n), dtype=np.float32)

        for edge in self.edges:
            nodes = edge["nodes"]
            weight = edge["attrs"].get("weight", 1.0) if weighted else 1.0
            for u, v in itertools.combinations(nodes, 2):
                idx_u, idx_v = node_to_idx[u], node_to_idx[v]
                if weighted:
                    A[idx_u, idx_v] += weight
                    A[idx_v, idx_u] += weight
                else:
                    A[idx_u, idx_v] = 1.0
                    A[idx_v, idx_u] = 1.0

        return A

    def to_edge_index(self, bidirectional: bool = True) -> np.ndarray:
        """
        Export edges in PyTorch Geometric (PyG) style edge_index format of shape (2, num_directed_edges).
        Hyperedges are expanded into pairwise clique edges.
        """
        if self.num_edges() == 0 or self.num_nodes() == 0:
            return np.empty((2, 0), dtype=np.int64)

        node_to_idx = {nid: idx for idx, nid in enumerate(self.nodes.keys())}
        src_list = []
        dst_list = []

        for edge in self.edges:
            nodes = edge["nodes"]
            for u, v in itertools.combinations(nodes, 2):
                idx_u, idx_v = node_to_idx[u], node_to_idx[v]
                src_list.append(idx_u)
                dst_list.append(idx_v)
                if bidirectional:
                    src_list.append(idx_v)
                    dst_list.append(idx_u)

        return np.array([src_list, dst_list], dtype=np.int64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to a Python dictionary."""
        return {
            "nodes": {str(nid): attrs for nid, attrs in self.nodes.items()},
            "edges": self.edges
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseGraph":
        """Reconstruct a BaseGraph from a dictionary."""
        g = cls()
        nodes = data.get("nodes", {})
        if isinstance(nodes, dict):
            for nid, attrs in nodes.items():
                g.add_node(int(nid), **attrs)
        elif isinstance(nodes, list):
            for nid in nodes:
                g.add_node(int(nid))

        for edge in data.get("edges", []):
            g.add_edge(edge["nodes"], **edge.get("attrs", {}))

        return g

    def __len__(self) -> int:
        return self.num_nodes()

    def __repr__(self) -> str:
        return f"BaseGraph(num_nodes={self.num_nodes()}, num_edges={self.num_edges()})"
