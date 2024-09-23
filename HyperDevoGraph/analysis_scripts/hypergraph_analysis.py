# hypergraph_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
import networkx as nx
import hypernetx as hnx


class HypergraphAnalyzer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.ce_data = None
        self.hypergraphs = {}

    def load_data(self):
        """
        Load and preprocess the C. elegans cell data.
        """
        # Load raw data
        ce_raw_data = pd.read_csv(f'{self.data_dir}/CE_raw_data.csv')
        ce_raw_data = ce_raw_data.drop(columns=['Unnamed: 0'])

        # Normalize coordinates
        ce_raw_data['x'] = ce_raw_data['x'] - ce_raw_data['x'].mean()
        ce_raw_data['y'] = ce_raw_data['y'] - ce_raw_data['y'].mean()
        ce_raw_data['z'] = ce_raw_data['z'] - ce_raw_data['z'].mean()

        # Load lineage data
        lineage_data = pd.read_excel(
            f'{self.data_dir}/cell-phenotype-lineage-data.xlsx',
            sheet_name='daughter-of-database',
            engine='openpyxl',
            usecols=['CELL NAME', 'CELL NAME.1']
        )
        lineage_data.rename(
            columns={'CELL NAME': 'cell', 'CELL NAME.1': 'mother'}, inplace=True)

        # Merge datasets
        self.ce_data = ce_raw_data.merge(lineage_data, how='inner', on='cell')

    def create_proximity_hypergraphs(self, threshold=5):
        """
        Create hypergraphs based on spatial proximity.

        Args:
            threshold (float): Distance threshold for grouping cells.
        """
        if self.ce_data is None:
            self.load_data()

        for time_point in range(1, 191, 10):
            sub_ce_data = self.ce_data[self.ce_data['time'] == time_point]
            positions = sub_ce_data[['x', 'y', 'z']].values
            nested_dict = {}

            # Group nodes by proximity
            for i in range(len(positions)):
                nested_dict[i] = [sub_ce_data.iloc[i]['cell']]
                for j in range(i + 1, len(positions)):
                    if np.linalg.norm(positions[i] - positions[j]) < threshold:
                        nested_dict[i].append(sub_ce_data.iloc[j]['cell'])

            H = hnx.Hypergraph(nested_dict)
            self.hypergraphs[time_point] = H

    def create_dbscan_hypergraphs(self, eps=5, min_samples=2, include_noise=True):
        """
        Create hypergraphs using DBSCAN clustering.

        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
            include_noise (bool): Whether to include noise points as separate hyperedges.
        """
        if self.ce_data is None:
            self.load_data()

        for time_point in range(1, 191, 10):
            sub_ce_data = self.ce_data[self.ce_data['time'] == time_point]

            # Combine the cells with the same name, average positions and size
            new_sub_ce_data = sub_ce_data.groupby('cell').agg(
                {'x': 'mean', 'y': 'mean', 'z': 'mean',
                    'size': 'mean', 'mother': 'first'}
            ).reset_index()

            positions = new_sub_ce_data[['x', 'y', 'z']].values
            clustering = DBSCAN(
                eps=eps, min_samples=min_samples).fit(positions)
            labels = clustering.labels_

            list_of_lists = []
            # Add nodes to hyperedges based on clusters
            for cluster_id in set(labels):
                if cluster_id != -1:  # Non-noise points
                    members = new_sub_ce_data['cell'][labels ==
                                                      cluster_id].tolist()
                    list_of_lists.append(members)
                elif include_noise:
                    # Add noise points as individual hyperedges
                    noise_members = new_sub_ce_data['cell'][labels == cluster_id].tolist(
                    )
                    list_of_lists.extend([[member]
                                         for member in noise_members])

            H = hnx.Hypergraph(list_of_lists)
            self.hypergraphs[time_point] = H

    def draw_hypergraphs(self, title_prefix='Hypergraphs'):
        """
        Visualize the hypergraphs.

        Args:
            title_prefix (str): Prefix for plot titles.
        """
        fig, axs = plt.subplots(5, 4, figsize=(20, 20))
        for idx, ax in enumerate(axs.flat):
            time_point = 1 + idx * 10
            if time_point not in self.hypergraphs:
                continue
            hnx.draw(self.hypergraphs[time_point], ax=ax)
            ax.set_title(f'{title_prefix} - Time point {time_point}')
        plt.tight_layout()
        plt.show()


def main():
    analyzer = HypergraphAnalyzer(data_dir='data')
    analyzer.load_data()

    # Create hypergraphs based on proximity threshold
    print("Creating proximity hypergraphs...")
    analyzer.create_proximity_hypergraphs(threshold=5)
    analyzer.draw_hypergraphs(title_prefix='Proximity Hypergraphs')

    # Create hypergraphs using DBSCAN clustering
    print("Creating DBSCAN hypergraphs...")
    analyzer.create_dbscan_hypergraphs(
        eps=5, min_samples=2, include_noise=True)
    analyzer.draw_hypergraphs(title_prefix='DBSCAN Hypergraphs')


if __name__ == '__main__':
    main()
