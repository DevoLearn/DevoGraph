"""Plotting helpers shared by the notebooks.

Every function takes plain pandas/numpy/torch objects and returns a
matplotlib ``Figure`` (notebooks call ``plt.show()`` or just let the figure
display). Kept dependency-light: matplotlib + seaborn + networkx + sklearn
(already required elsewhere in the project).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from .incidence import EDGE_TYPE_NAMES, HyperedgeBatch


def pca_reduce(X, n_components: int = 2) -> np.ndarray:
    """PCA via numpy SVD (avoids the sklearn.decomposition dependency)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


# ---------------------------------------------------------------------------
# Raw data / spatial structure
# ---------------------------------------------------------------------------

def plot_cell_positions_3d(
    positions: pd.DataFrame,
    color: Optional[Sequence] = None,
    title: str = "Cell positions",
    cmap: str = "tab20",
) -> plt.Figure:
    """3D scatter of cell positions. ``positions`` needs columns x, y, z."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(positions["x"], positions["y"], positions["z"], c=color, cmap=cmap, s=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    if color is not None and not isinstance(color, str):
        fig.colorbar(sc, ax=ax, shrink=0.6, label="cluster")
    fig.tight_layout()
    return fig


def plot_dbscan_clusters(positions: pd.DataFrame, labels: np.ndarray, title: str = "DBSCAN clusters") -> plt.Figure:
    """Scatter of (x, y) coloured by DBSCAN cluster id; noise (-1) shown in grey."""
    fig, ax = plt.subplots(figsize=(6, 6))
    is_noise = labels == -1
    ax.scatter(positions.loc[is_noise, "x"], positions.loc[is_noise, "y"], c="lightgrey", s=12, label="noise")
    ax.scatter(
        positions.loc[~is_noise, "x"],
        positions.loc[~is_noise, "y"],
        c=labels[~is_noise],
        cmap="tab20",
        s=18,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hyperedge / incidence structure
# ---------------------------------------------------------------------------

def plot_edge_type_counts(batch: HyperedgeBatch, title: str = "Hyperedges by type") -> plt.Figure:
    """Bar chart of how many hyperedges of each type are in H_aug(t)."""
    counts = [int((batch.edge_type == i).sum().item()) for i in range(len(EDGE_TYPE_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(EDGE_TYPE_NAMES, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("# hyperedges")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_incidence_sparsity(batch: HyperedgeBatch, title: str = "H_aug(t) sparsity") -> plt.Figure:
    """Scatter plot of nonzero (cell, hyperedge) incidences, coloured by edge type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cell_idx = batch.cell_index.numpy()
    edge_idx = batch.edge_index.numpy()
    etype = batch.edge_type[batch.edge_index].numpy()

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, name in enumerate(EDGE_TYPE_NAMES):
        m = etype == i
        ax.scatter(edge_idx[m], cell_idx[m], s=1, color=colors[i], label=name)

    ax.set_xlabel("hyperedge index")
    ax.set_ylabel("cell index")
    ax.set_title(title)
    ax.legend(markerscale=5)
    fig.tight_layout()
    return fig


def plot_hyperedge_size_distribution(batch: HyperedgeBatch, title: str = "Hyperedge size distribution") -> plt.Figure:
    sizes = batch.edge_sizes().numpy()
    etype = batch.edge_type.numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, name in enumerate(EDGE_TYPE_NAMES):
        s = sizes[etype == i]
        if len(s) > 0:
            ax.hist(s, bins=range(2, int(s.max()) + 2), alpha=0.6, label=name, color=colors[i])
    ax.set_xlabel("hyperedge size (# member cells)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_attention_for_cell(
    batch: HyperedgeBatch,
    e2v_attn: torch.Tensor,
    cell_idx: int,
    cell_name: str,
    top_k: int = 15,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of a Hyper-SAGNN layer's E2V attention weights for one cell.

    ``e2v_attn`` is the per-incidence attention tensor returned by
    ``HyperSAGNNLayer(..., return_attention=True)``, aligned with
    ``batch.cell_index`` / ``batch.edge_index``.
    """
    mask = batch.cell_index.numpy() == cell_idx
    edges = batch.edge_index.numpy()[mask]
    weights = e2v_attn.numpy()[mask]
    etypes = batch.edge_type[batch.edge_index].numpy()[mask]

    order = np.argsort(-weights)[:top_k]
    edges, weights, etypes = edges[order], weights[order], etypes[order]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bar_colors = [colors[e] for e in etypes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(weights)), weights, color=bar_colors)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"e{e}\n({EDGE_TYPE_NAMES[t]})" for e, t in zip(edges, etypes)], fontsize=7, rotation=45)
    ax.set_ylabel("E2V attention weight")
    ax.set_title(title or f"Top-{top_k} hyperedge attention weights for cell '{cell_name}'")
    fig.tight_layout()
    return fig


def plot_hypergraph_subgraph(
    batch: HyperedgeBatch,
    cell_names: Sequence[str],
    cell_indices: Sequence[int],
    max_edges: int = 60,
    title: str = "Hypergraph (bipartite view)",
) -> plt.Figure:
    """Bipartite cell<->hyperedge graph for a small subset of cells."""
    keep_cells = set(cell_indices)
    mask = np.isin(batch.cell_index.numpy(), list(keep_cells))
    edge_ids = np.unique(batch.edge_index.numpy()[mask])[:max_edges]

    G = nx.Graph()
    for c in keep_cells:
        G.add_node(("cell", c), bipartite=0, label=cell_names[c])
    for e in edge_ids:
        G.add_node(("edge", int(e)), bipartite=1)

    cell_idx = batch.cell_index.numpy()
    edge_idx = batch.edge_index.numpy()
    for c, e in zip(cell_idx, edge_idx):
        if c in keep_cells and e in edge_ids:
            G.add_edge(("cell", int(c)), ("edge", int(e)))

    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    cell_nodes = [n for n in G.nodes if n[0] == "cell"]
    edge_nodes = [n for n in G.nodes if n[0] == "edge"]
    nx.draw_networkx_nodes(G, pos, nodelist=cell_nodes, node_color="#4C72B0", node_size=300, ax=ax, label="cells")
    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, node_color="#C44E52", node_size=80, ax=ax, label="hyperedges")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_labels(G, pos, labels={n: cell_names[n[1]] for n in cell_nodes}, font_size=7, ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Embeddings / training curves
# ---------------------------------------------------------------------------

def plot_embeddings_pca(
    embeddings: torch.Tensor,
    labels: Sequence,
    label_names: Optional[Sequence[str]] = None,
    title: str = "Node embeddings (PCA)",
) -> plt.Figure:
    emb = embeddings.detach().cpu().numpy()
    if emb.shape[1] > 2:
        emb = pca_reduce(emb, 2)

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if label_names is not None:
        handles, _ = sc.legend_elements()
        ax.legend(handles, label_names, loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_training_curves(history: pd.DataFrame, title: str = "Training curves") -> plt.Figure:
    """``history`` has one row per (epoch, t) step with loss/metric columns."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for col in ["fate"]:
        if col in history.columns:
            axes[0].plot(history[col].to_numpy(), label=col, alpha=0.8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Losses")
    axes[0].legend()

    for col in ["fate_acc"]:
        if col in history.columns:
            axes[1].plot(history[col].to_numpy(), label=col, alpha=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Metrics")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spatial_evolution(
    counts_by_t: pd.DataFrame,
    title: str = "Hypergraph evolution over time",
) -> plt.Figure:
    """``counts_by_t`` indexed by t with columns = edge type names -> counts."""
    fig, ax = plt.subplots(figsize=(9, 4))
    counts_by_t.plot(ax=ax)
    ax.set_xlabel("timepoint t")
    ax.set_ylabel("# hyperedges")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D visualizations (interactive, Plotly)
# ---------------------------------------------------------------------------

def embryo_development_3d(nfb, universe, fate, fate_names, step: int = 5,
                          title: str = "C. elegans embryo development (coloured by founder lineage)"):
    """Animated 3D scatter of the embryo: every cell at its real (x, y, z),
    coloured by founder lineage, with a Play button + slider over timepoints.

    ``nfb`` is a NodeFeatureBuilder (uses ``nfb.raw_positions(t)``); ``fate`` is
    the (N,) founder-class tensor; sampled every ``step`` timepoints.
    """
    import plotly.graph_objects as go

    tps = nfb.timepoints[::step]

    def frame_scatter(t):
        df = nfb.raw_positions(t)
        df = df[df["cell"].isin(universe.names)]
        cls = [int(fate[universe.get(c)].item()) for c in df["cell"]]
        names = [fate_names[c] for c in cls]
        return go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"], mode="markers",
            marker=dict(size=3, color=cls, colorscale="Turbo", cmin=0, cmax=len(fate_names) - 1),
            text=names, hoverinfo="text", name=str(t),
        )

    frames = [go.Frame(name=str(t), data=[frame_scatter(t)]) for t in tps]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05, buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, steps=[dict(method="animate", label=f.name,
                 args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for f in frames])],
    )
    return fig


def plot_embeddings_3d(embeddings, labels, label_names=None,
                       title: str = "Learned embeddings (PCA to 3D)"):
    """Interactive 3D scatter of embeddings reduced to 3 PCA components,
    coloured by (lineage) class. Shows the clusters the coherence metric scores."""
    import plotly.graph_objects as go

    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(labels)
    xyz = pca_reduce(X, 3)
    names = [label_names[int(c)] for c in y] if label_names is not None else [str(int(c)) for c in y]
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=3, color=y, colorscale="Turbo"),
        text=names, hoverinfo="text",
    ))
    fig.update_layout(title=title, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    return fig


def plot_spatial_clusters_3d(positions: pd.DataFrame, cluster_labels,
                             title: str = "Spatial DBSCAN clusters (3D)"):
    """Interactive 3D scatter of cells at real (x, y, z) coloured by spatial
    DBSCAN cluster id; noise (-1) shown in grey. ``positions`` needs x, y, z."""
    import plotly.graph_objects as go

    labels = np.asarray(cluster_labels)
    noise = labels == -1
    traces = []
    if noise.any():
        traces.append(go.Scatter3d(
            x=positions["x"][noise], y=positions["y"][noise], z=positions["z"][noise],
            mode="markers", marker=dict(size=2, color="lightgrey"), name="noise", hoverinfo="skip"))
    keep = ~noise
    traces.append(go.Scatter3d(
        x=positions["x"][keep], y=positions["y"][keep], z=positions["z"][keep],
        mode="markers", marker=dict(size=3, color=labels[keep], colorscale="Turbo"),
        text=[f"cluster {c}" for c in labels[keep]], hoverinfo="text", name="clusters"))
    fig = go.Figure(traces)
    fig.update_layout(title=title, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig


def _animate(frames, title, axis_titles=("x", "y", "z"), scene_range=None):
    """Wrap a list of go.Frame into a Play/Pause + slider 3D figure (shared by
    all animated 3D views so A/B/C behave identically)."""
    import plotly.graph_objects as go

    fig = go.Figure(data=frames[0].data, frames=frames)
    scene = dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2])
    if scene_range is not None:
        (xr, yr, zr) = scene_range
        scene["xaxis"] = dict(title=axis_titles[0], range=xr)
        scene["yaxis"] = dict(title=axis_titles[1], range=yr)
        scene["zaxis"] = dict(title=axis_titles[2], range=zr)
    fig.update_layout(
        title=title, scene=scene,
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05, buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, steps=[dict(method="animate", label=f.name,
                 args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for f in frames])],
    )
    return fig


def embeddings_3d_animation(embeddings_by_t, present_by_t, fate, fate_names, step: int = 5,
                            title: str = "Learned embedding space over time (PCA to 3D)"):
    """Animated 3D scatter of the learned embeddings across timepoints.

    ``embeddings_by_t`` / ``present_by_t`` are dicts keyed by timepoint (from the
    trained model); ``fate`` is the (N,) founder-class tensor. PCA is fit **once**
    on all present embeddings pooled together so the axes stay fixed across frames
    (the clusters move, the coordinate system does not). Play + slider over time.
    """
    import plotly.graph_objects as go

    tps = sorted(embeddings_by_t.keys())[::step]
    pooled = np.concatenate([np.asarray(embeddings_by_t[t][present_by_t[t]], dtype=float) for t in tps], axis=0)
    mean = pooled.mean(axis=0)
    _, _, Vt = np.linalg.svd(pooled - mean, full_matrices=False)
    basis = Vt[:3].T  # (D, 3) stable PCA basis

    def frame_scatter(t):
        p = present_by_t[t]
        emb = np.asarray(embeddings_by_t[t][p], dtype=float)
        xyz = (emb - mean) @ basis
        y = np.asarray(fate[p])
        names = [fate_names[int(c)] for c in y]
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker=dict(size=3, color=y, colorscale="Turbo", cmin=0, cmax=len(fate_names) - 1),
            text=names, hoverinfo="text", name=str(t))

    all_xyz = (pooled - mean) @ basis
    rng = [[float(all_xyz[:, i].min()), float(all_xyz[:, i].max())] for i in range(3)]
    frames = [go.Frame(name=str(t), data=[frame_scatter(t)]) for t in tps]
    return _animate(frames, title, axis_titles=("PC1", "PC2", "PC3"), scene_range=(rng[0], rng[1], rng[2]))


def spatial_clusters_3d_animation(nfb, step: int = 5, eps: float = None, min_samples: int = None,
                                  title: str = "Spatial DBSCAN clusters over time (3D)"):
    """Animated 3D scatter of cells at real (x, y, z) coloured by their spatial
    DBSCAN cluster, across timepoints. Axes are fixed to the global embryo bounds
    so you watch the clusters form and split as the embryo develops. Noise (-1)
    shows as the low end of the colour scale."""
    import plotly.graph_objects as go
    from sklearn.cluster import DBSCAN
    from . import hyperedges

    if eps is None:
        eps = hyperedges.SPATIAL_EPS
    if min_samples is None:
        min_samples = hyperedges.SPATIAL_MIN_SAMPLES

    tps = nfb.timepoints[::step]
    # global bounds for stable axes
    lo = np.array([np.inf, np.inf, np.inf]); hi = -lo
    frames_pos = {}
    for t in tps:
        df = nfb.raw_positions(t).drop_duplicates("cell").reset_index(drop=True)
        frames_pos[t] = df
        if len(df):
            lo = np.minimum(lo, df[["x", "y", "z"]].min().values)
            hi = np.maximum(hi, df[["x", "y", "z"]].max().values)

    def frame_scatter(t):
        df = frames_pos[t]
        if len(df) >= min_samples:
            lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df[["x", "y", "z"]].values)
        else:
            lab = np.full(len(df), -1)
        n_clu = len(set(lab)) - (1 if -1 in lab else 0)
        return go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"], mode="markers",
            marker=dict(size=3, color=lab, colorscale="Turbo"),
            text=[f"t={t}  cluster {c}" for c in lab], hoverinfo="text",
            name=f"{t} ({n_clu} clusters)")

    rng = ([float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])], [float(lo[2]), float(hi[2])])
    frames = [go.Frame(name=str(t), data=[frame_scatter(t)]) for t in tps]
    return _animate(frames, title, axis_titles=("x", "y", "z"), scene_range=rng)


def plot_metrics_summary(fate_acc, emb_sil, emb_knn, raw_sil, raw_knn,
                         title="DHGNN - final metrics"):
    """Simple 2D bar chart of the two headline metrics.

    ``fate_acc`` is the held-out cell-fate accuracy (model only, no baseline).
    The two lineage-coherence scores (silhouette, kNN-purity) are shown as the
    **learned** embeddings vs a **raw-feature** baseline, so the bars directly
    show how much structure the DHGNN adds. Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    cats = ["Fate accuracy\n(held-out)", "Lineage\nsilhouette", "Lineage\nkNN-purity"]
    learned = [fate_acc, emb_sil, emb_knn]
    raw = [np.nan, raw_sil, raw_knn]
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, learned, w, label="learned (DHGNN)", color="#2b8cbe")
    b2 = ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in raw], w,
                label="raw features", color="#bdbdbd")
    b2[0].set_visible(False)  # no raw baseline for supervised fate accuracy
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend()
    for bars in (b1, b2):
        for r in bars:
            if not r.get_visible():
                continue
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return fig
