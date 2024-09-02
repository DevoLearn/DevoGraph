import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data


class Graph:
    """
    A class representing a graph with nodes, edges, and labels.
    This class provides methods to manipulate the graph, including adding nodes, edges, and plotting.
    """

    def __init__(self, nodes, edge_dict, labels):
        """
        Initializes a Graph object.

        Parameters:
        - nodes: A tensor containing the node features.
        - edge_dict: A dictionary where keys are node indices and values are lists of adjacent nodes.
        - labels: A dictionary containing labels for each node.
        """

        self.nodes = nodes
        self.edge_dict = edge_dict
        self.labels = labels
        
    def add_edges(self, edge_dict, parent_index):
        """
        Adds edges to the graph, updating the adjacency list.

        Parameters:
        - edge_dict: The current edge dictionary to update.
        - parent_index: The index of the parent node where new edges are to be added.

        Returns:
        - Updated Graph object with new edges.
        """

        for node in edge_dict:
            if parent_index in edge_dict[node]:
                edge_dict[node].remove(parent_index)
                edge_dict[node].append(self.nodes.size(0) - 2)
                edge_dict[node].append(self.nodes.size(0) - 1)
        edge_dict[self.nodes.size(0) - 2] = [parent_index]
        edge_dict[self.nodes.size(0) - 1] = [parent_index]
        self.edge_dict = edge_dict
        return Graph(self.nodes, self.edge_dict, self.labels)


    def add_daughter_cells(self, daughters, parent_index, daughter_labels):
        """
        Adds daughter cells to the graph.

        Parameters:
        - daughters: Tensor containing features of the daughter cells.
        - parent_index: Index of the parent cell that is splitting.
        - daughter_labels: Labels for the new daughter cells.

        Returns:
        - Updated Graph object with new daughter cells.
        """

        print(parent_index)
        self.labels[parent_index] = daughter_labels[0]
        self.labels.update({self.nodes.size(0) : daughter_labels[1]})
        print("labels",self.labels)
        return self.add_nodes(daughters, parent_index)


    def add_nodes(self, new_nodes, parent_index):
        """
        Adds new nodes to the graph at the specified parent index.

        Parameters:
        - new_nodes: Tensor containing features of the new nodes to be added.
        - parent_index: Index of the parent node where the new nodes are inserted.

        Returns:
        - Updated Graph object with new nodes.
        """

        if new_nodes.dim() == 1:
            new_nodes = new_nodes.unsqueeze(0)
        left_nodes = self.nodes[:parent_index]
        right_nodes = self.nodes[parent_index:]
        right_nodes = right_nodes[1:]
        self.nodes = torch.cat([left_nodes, new_nodes, right_nodes])
        print(self.nodes)
        return Graph(self.nodes, self.edge_dict, self.labels)


    def to_data(self):
        """
        Converts the graph into PyTorch Geometric Data format.

        Returns:
        - PyTorch Geometric Data object containing node features and edge indices.
        """

        edges = []
        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
        )


    def plot(self, fig=None, node_colors=None):
        """
        Plots the graph using NetworkX.

        Parameters:
        - fig: Matplotlib figure object for plotting (optional).
        - node_colors: List of colors for nodes (optional).

        Returns:
        - Numpy array representing the image of the plotted graph.
        """
        
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=True)

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")

        if fig is None:
            fig = plt.figure()
        canvas = FigureCanvas(fig)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels=self.labels)

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image
    

    def copy(self):
        """
        Creates a deep copy of the graph.

        Returns:
        - A new Graph object that is a deep copy of the current graph.
        """
        
        nodes = self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device)
        edge_dict = copy.deepcopy(self.edge_dict)
        labels = copy.deepcopy(self.labels)
        return Graph(
            nodes, edge_dict, labels
        )