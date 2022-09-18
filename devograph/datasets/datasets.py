import dgl
import os
import re
import pandas as pd
import torch as th
import numpy as np

from dgl.data import DGLDataset
from dgl.data.utils import download, save_info, save_graphs, load_graphs, load_info

home_dir = os.environ['HOME']+'/'
class CETemporalGraphKNN(DGLDataset):
    '''
    The C. elegans Temporal Graph built on given the csv file using KNN method.
    '''
    home_dir = os.environ['HOME']+'/'
    def __init__(self, name='CETemporalGraph', url=None, raw_dir=f'{home_dir}.CEData/', save_dir=f'{home_dir}.CEData/', 
                 hash_key=(), force_reload=False, verbose=False, transform=None, 
                 knn_k=4, knn_algorithm='bruteforce-blas', knn_dist='euclidean', 
                 time_start=None, time_end=None, columns=[]):
        self.knn_k = knn_k
        self.knn_algorithm=knn_algorithm
        self.knn_dist=knn_dist
        self.time_start = time_start
        self.time_end = time_end
        self.graphs = []
        self.info = {'cell':[]}
        self.batch_graph = None
        self.columns=columns

        super().__init__(name, url, raw_dir, save_dir, hash_key, 
                         force_reload, verbose, transform)
        
        
    def has_cache(self):
        graph_path = os.path.join(self.save_dir,
                                  self.name + '.bin')
        if os.path.exists(graph_path):
            return True

        return False

    def download(self):
        if os.path.exists(self.raw_path):
            return
        file_name = download(self.url, self.raw_path)
        file_name = re.findall(r"([^\/]*\.csv)", file_name)[0]
        print(f'Finish downloading {file_name}')

    def process(self):
        raw_data:pd.DataFrame = pd.read_csv(self.raw_path, usecols=['cell', 'time', 'x', 'y', 'z', 'size'])
        time_start = self.time_start if self.time_start is not None else raw_data.time.min()
        time_end = self.time_end if self.time_end is not None else raw_data.time.max()
        self.info['time_start'] = time_start
        self.info['time_end'] = time_end

        for time in range(time_start, time_end+1):
            _raw_data = raw_data[raw_data.time == time]
            if len(_raw_data) == 0:
                continue
            # TODO: need to average positions of the same node (type)?
            # Why are there multiple cells with the same name?
            pos = th.tensor(_raw_data[['x', 'y', 'z']].to_numpy())
            graph = dgl.knn_graph(pos, self.knn_k, self.knn_algorithm, self.knn_dist)
            graph.ndata['pos'] = pos
            for col in self.columns:
                graph.ndata[col] = th.tensor(_raw_data['size'].to_numpy())
            self.info['cell'].append(_raw_data['cell'].to_list())
            self.graphs.append(graph)
        
    def save(self):
        save_info(f'{self.save_dir}{self.name}_info.pkl', self.info)
        save_graphs(f'{self.save_dir}{self.name}.bin', self.graphs)

    def load(self):
        self.graphs, _ = load_graphs(f'{self.save_dir}{self.name}.bin')
        self.info = load_info(f'{self.save_dir}{self.name}_info.pkl')

    def set_batch_graph(self, batch_graph):
        self.batch_graph = batch_graph

    def set_info(self, new_info:dict):
        self.info.update(new_info)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
        
    @property
    def raw_path(self):
        return f'{self.raw_dir}{self.raw_name}'

    @property
    def raw_name(self):
        return 'CE_raw_data.csv'

def to_temporal_directed(cell_temp_datasets, ce_lineage_path, verbose=False):
    '''
    Given C elegans temporal graph datasets, these API builds a directed graph linking 
    daughter and mother cells between successive graphs. A new node attribute in 'ndata'
    that is the relative timestamp of each graph(frame) will be built for differentiating
    different frames in the temporal directed graph.

    Note that the same cell across successive frames will be connected as well.

    ce_temp_datasets: CETemporalGraphKNN
    ce_lineage_path: pickle path of 
            the C elegans lineage csv file containing "mother" and "daughter" columns, and each
            column contains cell names
    
    return: a directed graph
    '''
    
    info = cell_temp_datasets.info
    lineage = pd.read_csv(ce_lineage_path, usecols=['daughter', 'mother'])
    
    for i, g in enumerate(cell_temp_datasets):
        g.ndata['time'] = th.full((g.number_of_nodes(), 1), i)

    res_g:dgl.DGLGraph = dgl.batch(cell_temp_datasets)
    _batch_node_interval = th.cumsum(res_g.batch_num_nodes(), dim=0)
    batch_node_interval = [(0 if i == 0 else _batch_node_interval[i-1].item(), _batch_node_interval[i].item()) 
            for i in range(len(_batch_node_interval))]

    for i in range(1, len(batch_node_interval)):
        daughter_list, mother_list = info['cell'][i], info['cell'][i-1]
        daughter_interval, mother_interval = batch_node_interval[i], batch_node_interval[i-1]
        daughter_idx:np.ndarray = np.array([])
        mother_idx:np.ndarray = np.array([])

        for j, daughter in enumerate(daughter_list):
            if daughter not in lineage.daughter.values:
                if verbose:
                    print(f'daughter cel {daughter} is not in the daughter column of lineage tree, skip it.')
                continue
            mother = lineage[lineage.daughter == daughter].mother.unique()[0]
            # TODO: the daughter cell in the last frame will be connected to the cell in the next frame
            if mother not in mother_list:
                if verbose:
                    print(f'mother cell {mother} of the daughter cell {daughter} in the {i}th frame '
                        f'is not in the the {i-1}th frame of the input temporal graphs, skip it')
                continue
            _mother_idx = (np.where((np.array(mother_list) == mother) | 
                    (np.array(mother_list) == daughter))[0] + \
                    mother_interval[0]) # left
            mother_idx = np.append(mother_idx, _mother_idx)
            daughter_idx = np.append(daughter_idx, 
                    np.repeat(j + daughter_interval[0], len(_mother_idx)))
        mother_idx = mother_idx.astype(np.int32)
        daughter_idx = daughter_idx.astype(np.int32)
        res_g = dgl.add_edges(res_g, mother_idx, daughter_idx)

    return res_g, batch_node_interval 
    
if __name__ == '__main__':
    datasets = CETemporalGraphKNN(
        time_start=0, time_end=3, columns=['size'],
        url='https://raw.githubusercontent.com/LspongebobJH/DevoGraph/main/data/CE_raw_data.csv?token=GHSAT0AAAAAABMX6RJRRFC2U5QOCZXHNBVYYVL5Y2A')
    res_g, batch_node_interval = to_temporal_directed(datasets, '~/.CEData/CE_lineage_data.csv')
    datasets.set_batch_graph(res_g)
    datasets.set_info({'batch_node_interval': batch_node_interval})
    print("Done testing")
