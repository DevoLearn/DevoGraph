import random
from typing import Optional
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GraphNCA(nn.Module):
    """
    Neural Cellular Automaton (NCA) model for a graph-based structure.

    This model uses a GCN (Graph Convolutional Network) to learn representations
    and update node features, and can grow or modify the graph dynamically based on certain rules.
    """
    
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        """
        Initializes the GraphNCA model.

        Parameters:
        - graph: A graph object containing the initial structure.
        - num_hidden_channels: Number of hidden channels in the neural networks.
        - max_replications: Maximum number of times a cell can replicate.
        """

        super().__init__()
        self.graph = graph

        self.value_idx = 0
        self.replication_idx = 1

        self.operations = [torch.add, torch.subtract, torch.multiply]
        self.activations = [torch.relu, torch.tanh]

        self.replicated_cells = []
        self.num_operations = len(self.operations)
        self.num_activations = len(self.activations)

        self.operation_channels = [2, 4]
        self.activation_channels = [5, 6]

        self.num_hidden_channels = num_hidden_channels
        self.num_channels = self.get_number_of_channels(
            self.num_operations, self.num_activations, self.num_hidden_channels
        )

        self.perception_net = GCNConv(
            self.num_channels, self.num_channels * 3, bias=False
        )
        self.update_net = nn.Sequential(
            nn.Linear(self.num_channels * 3, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels),
        )
        self.split_network = nn.Sequential(        
            nn.Linear(self.num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels * 2),
        )
        self.max_replications = max_replications

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        """
        Calculates the total number of channels based on the number of operations, activations, and hidden channels.

        Parameters:
        - num_operations: Number of operations available (e.g., add, subtract).
        - num_activations: Number of activation functions available (e.g., ReLU, tanh).
        - num_hidden_channels: Number of hidden channels in the neural network.

        Returns:
        - Total number of channels (int).
        """

        return num_operations + num_activations + num_hidden_channels + 2


    def forward(self, x, edge_index):
        """
        Forward pass through the model.

        Parameters:
        - x: Node features (tensor of shape [num_nodes, num_channels]).
        - edge_index: Edge indices (tensor defining connections between nodes).

        Returns:
        - Updated node features after processing (tensor of shape [num_nodes, num_channels]).
        """

        features = self.perception_net(x, edge_index)
        update = self.update_net(features)
        x = x + update
        return x


    def grow(
        self,
        graph,
        parent_index,
        daughter_labels,
    ):
        """
        Grows the graph by replicating cells and adding new connections.

        Parameters:
        - graph: The graph object to grow.
        - parent_index: Index of the parent cell to replicate.
        - daughter_labels: Labels for the new daughter cells.

        Returns:
        - new_graph: The updated graph after growth.
        """
        
        new_graph = graph.copy()

        data = new_graph.to_data()

        x = self.forward(data.x, data.edge_index)

        split = self.split_network(x[parent_index])
        
        daughter1 = split[:self.num_channels]
        daughter2 = split[self.num_channels:]
        daughters = torch.stack([daughter1, daughter2])

        new_graph = new_graph.add_daughter_cells(daughters, parent_index, daughter_labels)

        new_graph = new_graph.add_edges(new_graph.edge_dict, parent_index)
        print("grown graph",new_graph.nodes.shape)

        return new_graph