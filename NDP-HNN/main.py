"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
import argparse
import os
from config import Config
from utils import set_seed, get_device, ensure_dir
from data_io import load_dataset
from snapshots import build_snapshots
from model import DynGrowingHNN
from train import train_model
from evaluate import extract_embeddings, save_embeddings

def parse_args():
    p = argparse.ArgumentParser("NDP-HNN Training")
    p.add_argument("--csv", type=str, default=Config.csv_path, help="cells_birth_and_pos.csv")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--k", type=int, default=Config.knn_k)
    p.add_argument("--radius", type=float, default=Config.spatial_radius)
    p.add_argument("--hid", type=int, default=Config.hid_dim)
    p.add_argument("--out", type=int, default=Config.out_dim)
    p.add_argument("--conv", type=str, default=Config.conv_type, choices=["hgcn","hsage","ugnn"])
    p.add_argument("--rnn", type=str, default=Config.rnn_type, choices=["gru","lstm","rnn"])
    p.add_argument("--tf", action="store_true", help="use a tiny transformer encoder")
    p.add_argument("--save_dir", type=str, default=Config.save_dir)
    p.add_argument("--emb", type=str, default=None)
    p.add_argument("--seed", type=int, default=Config.seed)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.save_dir)

    #--- 1. data
    dataset = load_dataset(args.csv)

    #--- 2. hypergraph snapshots
    snaps = build_snapshots(dataset, k=args.k, spatial_radius=args.radius)
    print(f"Built {len(snaps)} snapshots (time 0..{dataset['T_max']})")

    #--- 3. model
    model = DynGrowingHNN(
        in_dim=4, hid_dim=args.hid, out_dim=args.out,
        num_edge_types=2,
        conv_type=args.conv,
        rnn_type=args.rnn,
        use_transformer=args.tf
    ).to(device)

    #--- 4. train
    model = train_model(model, snaps, dataset, epochs=args.epochs, lr=args.lr, device=device)

    #--- 5. embeddings (T, N, D)
    embeds = extract_embeddings(model, snaps, device=device)

    # --- 6. decide where to save
    cfg = Config(conv_type=args.conv, rnn_type=args.rnn, save_dir=args.save_dir)
    out_path = args.emb if args.emb else cfg.embeddings_path

    # --- 7. save
    save_embeddings(embeds, out_path)
    print(f"Embeddings saved to: {out_path}")

if __name__ == "__main__":
    main()
