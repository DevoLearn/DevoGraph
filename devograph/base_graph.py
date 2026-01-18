from typing import Dict, List, Any


class BaseGraph:
    """
    Base graph class for DevoGraph.
    Supports normal edges and hyperedges.
    """

    def __init__(self):
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, node_id: int, **attrs):
        if node_id in self.nodes:
            raise ValueError("Node already exists")
        self.nodes[node_id] = attrs

    def add_edge(self, node_ids: List[int], **attrs):
        if len(node_ids) < 2:
            raise ValueError("Edge must have at least two nodes")

        for nid in node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Node {nid} does not exist")

        self.edges.append({
            "nodes": node_ids,
            "attrs": attrs
        })

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)
