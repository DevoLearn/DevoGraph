import argparse
import dgl
import torch as th
from kmapper import Cover
from devograph.models.gcn import GCN
from devograph.datasets.datasets import CETemporalGraphKNN, to_temporal_directed

'''TODO:
* mini-batch training has not been implemented
* in frame_pipe training process in incomplete
* in frame_topo_pipe how to define length of interval and fraction of overlapping part 
'''

def frame_gnn_pipe(model, g:dgl.DGLGraph, feat, args):
    gnn_embed = model(g, feat)
    return gnn_embed

def frame_topo_pipe(lens, g:dgl.DGLGraph, topo, args):
    num_cover, perc_overlap = args.num_cover, args.perc_overlap
    axis = lens(g, topo).detach().cpu().numpy()
    cover = Cover(num_cover, perc_overlap)
    cover.fit_transform(axis)



def concat(gnn_embed, topo_embed):
    pass
    
def frame_pipe(g:dgl.DGLGraph, args):
    g.to(args.device)
    if isinstance(args, dict):
        args = argparse.Namespace(args)
    if args.model_name == 'gcn':
        feat = g.ndata['feat']
        model = GCN(g.ndata['feat'].shape[0], args.hid_dim, args.out_dim, dropout=args.droput)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    if args.topo:
        topo = g.ndata['pos']
        lens = GCN(topo.shape[0], args.topo_hid_dim, args.topo_out_dim, dropout=args.droput)
    
    model.train()
    lens.train()
    for epoch in range(args.n_epochs):
        gnn_embed = frame_gnn_pipe(model, g, feat, args)
        topo_embed = frame_gnn_pipe(lens, g, topo, args)
        output_embed = concat(gnn_embed, topo_embed)

if __name__ == '__main__':
    args = {
        'model_name': 'gcn',
        'topo': True,
        'hid_dim': 8,
        'out_dim': 3,
        'topo_hid_dim': 8,
        'topo_out_dim': 3,
        'num_cover': 5,
        'perc_overlap': 0.2
    }

    print("start loading graph")
    datasets = CETemporalGraphKNN(
        time_start=0, time_end=3, columns=['size'],
        url='https://raw.githubusercontent.com/LspongebobJH/DevoGraph/main/data/CE_raw_data.csv?token=GHSAT0AAAAAABMX6RJRRFC2U5QOCZXHNBVYYVL5Y2A')
    res_g, batch_node_interval = to_temporal_directed(datasets, '~/.CEData/CE_lineage_data.csv')
    datasets.set_batch_graph(res_g)
    datasets.set_info({'batch_node_interval': batch_node_interval})
    print("done loading graph")

    frame_pipe(datasets[0], args)
    print("done testing")
