import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import cma

class GraphNCA(nn.Module):
    def __init__(self, graph, csv_path, num_channels: int = 16):
        """
        Initializes the GraphNCA model.
        
        Args:
            graph (Graph): The graph structure on which the model operates.
            csv_path (str): Path to the CSV file containing true positions.
            num_channels (int): Number of channels or features for each node in the graph (default: 16).
        """
        super().__init__()
        self.graph = graph

        # Initialize distance matrices
        self.predicted_distance_matrix = None
        self.true_distance_matrix = None

        # Load true positions of cells from CSV
        self.true_positions = self.load_positions_from_csv(csv_path)

        self.num_channels = num_channels

        # Perception network based on GCNConv
        self.perception_net = GCNConv(
            self.num_channels, self.num_channels * 3, bias=False
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(self.num_channels * 3, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels),
        )
        
        # Split network for generating daughter cells
        self.split_network = nn.Sequential(        
            nn.Linear(self.num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels * 2),
        )


    def forward(self, xx, edge_index):
        """
        Forward pass for updating node features.

        Args:
            xx (Tensor): Input node features.
            edge_index (Tensor): Edge connectivity information.

        Returns:
            Tensor: Updated node features.
        """
        features = self.perception_net(xx, edge_index)        
        update = self.update_net(features)
        xx = xx.clone() + update

        return xx


    def load_positions_from_csv(self, csv_path):
        """
        Loads the true positions of cells from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            dict: A dictionary mapping cell labels to their 3D positions as tensors.
        """
        df = pd.read_csv(csv_path)
        true_positions = {}
        for _, row in df.iterrows():
            cell_label = row['Parent Cell']
            position = (row['parent_x'], row['parent_y'], row['parent_z'])
            true_positions[cell_label] = torch.tensor(position, dtype=torch.float32)
        return true_positions


    def euclidean_distance_3d(self, node1, node2):
        """
        Calculate Euclidean distance between two nodes in 3D space.

        Args:
            node1 (Tensor): 3D position of the first node.
            node2 (Tensor): 3D position of the second node.

        Returns:
            Tensor: The Euclidean distance between the two nodes.
        """
        diff = node1 - node2
        dist = torch.sqrt(torch.clamp(torch.sum(diff ** 2), min=1e-9, max=1e9))
        return dist


    def objective_function(self, params):
        """
        Objective function for optimization. Calculates the loss based on parameters.

        Args:
            params (list): List of model parameters to be optimized.

        Returns:
            float: The calculated loss.
        """
        # Assign the parameters to the model
        self.set_parameters(params)

        # Update distance matrices
        self.update_distance_matrices()

        # Calculate and return loss
        loss = self.calculate_loss()
        return loss.item()


    def set_parameters(self, params):
        """
        Set the parameters of the model.

        Args:
            params (list): Flattened list of model parameters.
        """
        start = 0
        for p in self.parameters():
            end = start + p.numel()
            p.data.copy_(torch.tensor(params[start:end]).view_as(p))
            start = end


    def optimize_with_cma_es(self, graph):
        """
        Optimize model parameters using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

        Args:
            graph (Graph): The graph object representing the current state of the cells.
        """
        initial_params = torch.cat([p.flatten() for p in self.parameters()]).detach().numpy()

        # Initialize CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(initial_params, 0.5)

        self.graph = graph.copy()

        # Perform optimization loop
        while not es.stop():
            solutions = es.ask()
            losses = [self.objective_function(x) for x in solutions]
            loss_tensor = torch.tensor(losses)
            loss = torch.clamp(loss_tensor, min=-1e6, max=1e6).tolist()
            es.tell(solutions, loss)

        # Retrieve optimized parameters
        optimized_params = es.result.xbest
        self.set_parameters(optimized_params)


    def update_distance_matrices(self):
        """
        Update the predicted and true distance matrices based on node positions.
        """
        data = self.graph.to_data()
        data.x = torch.clamp(data.x, min=-1e6, max=1e6)
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1e6, neginf=-1e6)

        num_nodes = data.x.size(0)

        if self.predicted_distance_matrix is None or self.predicted_distance_matrix.size(0) != num_nodes:
            self.predicted_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            self.true_distance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # Update the predicted distance matrix
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = self.euclidean_distance_3d(data.x[i], data.x[j])
                self.predicted_distance_matrix[i, j] = dist
                self.predicted_distance_matrix[j, i] = dist

        # Update the true distance matrix
        labels = list(self.graph.labels.values())
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                label1 = labels[i]
                label2 = labels[j]
                pos1 = self.true_positions[label1]
                pos2 = self.true_positions[label2]
                dist = self.euclidean_distance_3d(pos1, pos2)
                self.true_distance_matrix[i, j] = dist
                self.true_distance_matrix[j, i] = dist


    def calculate_loss(self):
        """
        Calculate the MSE (Mean Squared Error) loss between the predicted and true distance matrices.

        Returns:
            Tensor: The total loss, including regularization.
        """
        criterion = nn.MSELoss()
        mse_loss = criterion(self.predicted_distance_matrix, self.true_distance_matrix)

        # L2 regularization to avoid overfitting
        l2_lambda = 1e-5
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        total_loss = mse_loss + l2_lambda * l2_norm

        return total_loss


    def grow(self, graph, parent_index, daughter_labels):
        """
        Simulate the growth of cells by splitting a parent cell into two daughter cells.

        Args:
            graph (Graph): The graph structure before growth.
            parent_index (int): Index of the parent cell in the graph.
            daughter_labels (list): List of labels for the daughter cells.

        Returns:
            new_graph (Graph): The graph after growth.
            updated_nodes (list): List of newly added nodes representing the daughter cells.
        """
        new_graph = graph.copy()
        data = new_graph.to_data()

        # Normalize node features
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True) + 1e-6
        data.x = (data.x - mean) / std
    
        # Update node features using the forward pass
        xx = self.forward(data.x, data.edge_index)

        # Split parent cell into two daughter cells
        split = self.split_network(xx[parent_index])
        daughter1 = split[:self.num_channels]
        daughter2 = split[self.num_channels:]
        daughters = torch.stack([daughter1, daughter2])

        # Add daughter cells to the graph
        new_graph, updated_nodes = new_graph.add_daughter_cells(daughters, parent_index, daughter_labels)

        return new_graph, updated_nodes
