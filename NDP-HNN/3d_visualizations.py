# -*- coding: utf-8 -*-
"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""

import sys
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataloader import CEGrowingHypergraphDataset


# ---adjust these values to test quickly
dataset = CEGrowingHypergraphDataset("./data/ce_temporal_data.csv", max_timepoints=3, max_cells_per_tp=30)

def snapshot_to_plotly_df(snapshot, time_index):
    features = snapshot['node_features'].numpy()
    cell_ids = [f"cell_{i}" for i in range(len(features))]
    reverse_map = {v: u for u, v in snapshot['lineage_edges']}
    rows = []
    for idx, (depth, x, y, z) in enumerate(features):
        cell = cell_ids[idx]
        mother = cell_ids[reverse_map[idx]] if idx in reverse_map else "P0"
        rows.append({
            "cell": cell,
            "mother": mother,
            "time": time_index,
            "x": x,
            "y": y,
            "z": z
        })
    return pd.DataFrame(rows)

#---combine first 3 timepoints
dfs = []
for t in range(min(3, len(dataset))):
    df_t = snapshot_to_plotly_df(dataset[t], time_index=t+1)
    dfs.append(df_t)

plot_df = pd.concat(dfs, ignore_index=True)
plot_df['node_id'] = plot_df['cell'] + "_" + plot_df['time'].astype(str)
#---saving the .csv file to the data folder for the purpose of future use and cross check visualizations
plot_df.to_csv("./data/ce_hypergraph_subset_plot_ready.csv", index=False)

#---load the dataset and restrict to time points 1, 2, and 3.
df = pd.read_csv("./data/ce_hypergraph_subset_plot_ready.csv")
df_subset = df[df['time'].isin([1, 2, 3])].copy()
df_subset['node_id'] = df_subset['cell'] + "_" + df_subset['time'].astype(str)

#---define color mapping for a node based on its first appearance time.
def node_color(time_val):
    if time_val == 1:
        return "red"
    elif time_val == 2:
        return "green"
    elif time_val == 3:
        return "blue"
    else:
        return "gray"

#-------------------
#---helper functions

def get_node_positions(df):
    """Return a dictionary mapping node_id -> (x, y, z) from the DataFrame."""
    return {row['node_id']: (row['x'], row['y'], row['z']) for _, row in df.iterrows()}

def get_node_colors(df):
    """
    Return a dictionary mapping each node_id to its color,
    according to the 'time' field in the DataFrame.
    """
    return {row['node_id']: node_color(row['time']) for _, row in df.iterrows()}

def get_hyperedges(df):
    """
    Group nodes by the 'mother' attribute.
    Each group (except for mother == "P0") defines a hyperedge.
    """
    hyperedges = defaultdict(set)
    for _, row in df.iterrows():
        hyperedges[row['mother']].add(row['node_id'])
    return hyperedges

def compute_hypernode_positions(df, pos):
    """
    Given a DataFrame (which is assumed to be cumulative up to a time point) and
    a dictionary of node positions, compute a hypernode (parent node) for each mother (≠ "P0").
    The hypernode is defined as the centroid of its daughter nodes.
    Also assign the hypernode a color based on the minimal time among daughters:
      - red if at least one daughter is from time 1,
      - green if no time 1, but at least one from time 2,
      - blue otherwise.
    Returns two dictionaries:
      - hyper_pos: mapping mother -> (x,y,z) centroid.
      - hyper_color: mapping mother -> color.
    """
    hyperedges = get_hyperedges(df)
    hyper_pos = {}
    hyper_color = {}
    # For each mother (except "P0"):
    for mother, nodes in hyperedges.items():
        if mother == "P0" or len(nodes) == 0:
            continue
        coords = np.array([pos[n] for n in nodes])
        centroid = coords.mean(axis=0)
        hyper_pos[mother] = centroid
        # Determine the earliest time among daughter nodes.
        times = []
        for n in nodes:
            # Extract time value from the original df: node_id format "cell_time"
            try:
                t = int(n.split("_")[-1])
                times.append(t)
            except:
                times.append(99)  # fallback
        min_time = min(times)
        hyper_color[mother] = node_color(min_time)
    return hyper_pos, hyper_color

def get_parent_child_edges(df, pos):
    """
    For each daughter (where mother != "P0"), locate a parent (with cell equal to the mother
    and with an earlier time) in the given cumulative DataFrame.
    Returns a list of tuples (parent_node, daughter_node).
    Only those parent nodes that are present in the cumulative set (pos) are included.
    """
    parent_edges = []
    for idx, row in df.iterrows():
        if row['mother'] != "P0":
            possible_parents = df[(df['cell'] == row['mother']) & (df['time'] < row['time'])]
            if not possible_parents.empty:
                parent_row = possible_parents.sort_values(by='time', ascending=False).iloc[0]
                if parent_row['node_id'] in pos:
                    parent_edges.append((parent_row['node_id'], row['node_id']))
    return parent_edges

def save_positions(cum_df, pos, hyper_pos, filename):
    """
    Save the 3D positions of daughter nodes and hypernodes into a CSV file.
    The CSV file will contain: id, x, y, z, and type ("daughter" or "parent").
    """
    data = []
    for node, (x, y, z) in pos.items():
        data.append({'id': node, 'x': x, 'y': y, 'z': z, 'type': 'daughter'})
    for parent, (x, y, z) in hyper_pos.items():
        data.append({'id': parent, 'x': x, 'y': y, 'z': z, 'type': 'parent'})
    df_positions = pd.DataFrame(data)
    df_positions.to_csv(filename, index=False)
    print(f"Saved 3D positions to {filename}")




# ----------------------------------
#---New cumulative plotting function

def plot_cumulative_parent_division(cum_df, title):
    """
    Create Plotly traces for a cumulative DataFrame 'cum_df' (all rows with time <= current time).
    Each daughter node is assigned a color based on its time of appearance.
    Hypernodes (parent centroids) are computed and colored based on the earliest daughter's time.
    Parent-daughter (explicit) edges are also drawn.
    Returns:
      - traces: A list of Plotly traces.
      - pos: Dictionary of daughter node positions.
      - hyper_pos: Dictionary of hypernode positions.
      - hyper_color: Dictionary mapping hypernode to its color.
    """
    pos = get_node_positions(cum_df)
    node_colors = get_node_colors(cum_df)
    hyperedges = get_hyperedges(cum_df)
    hyper_pos, hyper_color = compute_hypernode_positions(cum_df, pos)
    parent_edges = get_parent_child_edges(cum_df, pos)

    traces = []

    #---draw edges from each hypernode to its daughter nodes.
    for mother, nodes in hyperedges.items():
        if mother == "P0" or mother not in hyper_pos:
            continue
        center = hyper_pos[mother]
        col = hyper_color[mother]
        for n in nodes:
            x_vals = [center[0], pos[n][0]]
            y_vals = [center[1], pos[n][1]]
            z_vals = [center[2], pos[n][2]]
            traces.append(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines',
                line=dict(color=col, width=3, dash='dot'),
                hoverinfo='none',
                showlegend=False
            ))

    #---overlay explicit parent-daughter edges.
    for u, v in parent_edges:
        if u in pos and v in pos:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            z_vals = [pos[u][2], pos[v][2]]
            #---use the color of the daughter node v for the edge.
            color_edge = node_colors[v]
            traces.append(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines',
                line=dict(color=color_edge, width=4, dash='dash'),
                hoverinfo='text',
                text=f"Parent: {u} → Daughter: {v}",
                showlegend=False
            ))

    #---plot daughter nodes.
    node_x = [pos[n][0] for n in pos]
    node_y = [pos[n][1] for n in pos]
    node_z = [pos[n][2] for n in pos]
    node_text = list(pos.keys())
    node_marker_color = [node_colors[n] for n in pos]
    traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(size=6, color=node_marker_color, symbol='circle'),
        text=node_text,
        textposition="top center",
        name="Daughter Nodes"
    ))

    #---plot hypernodes as squares.
    if hyper_pos:
        hyper_x = [hyper_pos[m][0] for m in hyper_pos]
        hyper_y = [hyper_pos[m][1] for m in hyper_pos]
        hyper_z = [hyper_pos[m][2] for m in hyper_pos]
        hyper_text = list(hyper_pos.keys())
        hyper_marker_color = [hyper_color[m] for m in hyper_pos]
        traces.append(go.Scatter3d(
            x=hyper_x, y=hyper_y, z=hyper_z,
            mode='markers+text',
            marker=dict(size=10, color=hyper_marker_color, symbol='square'),
            text=hyper_text,
            textposition="bottom center",
            name="Parent Hypernodes"
        ))

    return traces, pos, hyper_pos



# -------------------------------
#---create cumulative data for each time point:
time_points = [1, 2, 3]
cumulative_dfs = {t: df_subset[df_subset['time'] <= t] for t in time_points}
cumulative_titles = {1: "Cumulative Time <= 1", 2: "Cumulative Time <= 2", 3: "Cumulative Time <= 3"}

#---create subplots: one cumulative scene per time point.
fig = make_subplots(rows=1, cols=3,
                    specs=[[{'type': 'scene'}]*3],
                    subplot_titles=[cumulative_titles[t] for t in time_points])
saved_positions = {}

for i, t in enumerate(time_points, start=1):
    cum_df = cumulative_dfs[t]
    traces, pos, hyper_pos = plot_cumulative_parent_division(cum_df, cumulative_titles[t])
    for trace in traces:
        fig.add_trace(trace, row=1, col=i)
    #---save cumulative positions into CSV.
    filename = f"cumulative_3D_positions_time_{t}.csv"
    save_positions(cum_df, pos, hyper_pos, filename)
    saved_positions[t] = {"daughter": pos, "parent": hyper_pos}

fig.update_layout(height=600, width=1800, title_text="Cumulative Parent-Child Division Visualization\n(Red: Time 1; Green: new at Time 2; Blue: new at Time 3)")
fig.show()
