import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def save_cell_division_hypergraph(
    csv_path: str,
    html_save: str = 'full',
    timepoints: list[int] | None = None,
    output_full: str = 'cells_division_hypergraph.html',
    output_partial: str = 'cells_division_windows_hypergraph.html'):
    
  """
    Load cell‐division data and save either:
      - full star‐expansion hypergraph HTML (html_save='full'), or
      - partial (two‐window) hypergraph HTML (html_save='partial').
    """
  
    #--- load data
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'Parent Cell': 'cell',
        'parent_x': 'x',
        'parent_y': 'y',
        'parent_z': 'z',
        'Birth Time': 'birth_time'
    })

    #--- create node dict
    df_nodes = df.drop_duplicates(subset='cell')
    nodes = df_nodes.set_index('cell')[['x','y','z']].to_dict('index')

    #---create hyperedges with centroids
    hyperedges = []
    for _, row in df.iterrows():
        parent = row['cell']
        d1, d2 = row['Daughter 1'], row['Daughter 2']
        if d1 in nodes and d2 in nodes:
            coords = np.vstack([
                [row['x'], row['y'], row['z']],
                [nodes[d1]['x'], nodes[d1]['y'], nodes[d1]['z']],
                [nodes[d2]['x'], nodes[d2]['y'], nodes[d2]['z']]
            ])
            centroid = coords.mean(axis=0)
            hyperedges.append({
                'id': f"he_{parent}",
                'members': [parent, d1, d2],
                'centroid': centroid,
                'birth_time': int(row['birth_time'])
            })

    colors = px.colors.qualitative.Plotly
    color_map = {he['id']: colors[i % len(colors)] for i, he in enumerate(hyperedges)}

    if html_save == 'full':
        #--- full star‐expansion
        fig = go.Figure()
        #--- plot all cells
        fig.add_trace(go.Scatter3d(
            x=[n['x'] for n in nodes.values()],
            y=[n['y'] for n in nodes.values()],
            z=[n['z'] for n in nodes.values()],
            mode='markers',
            marker=dict(size=4, color='lightgray'),
            name='Cells',
            hovertext=list(nodes.keys())
        ))
        #--- plot each hyperedge
        for he in hyperedges:
            cx, cy, cz = he['centroid']
            #--- extra node
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode='markers+text',
                marker=dict(size=9, symbol='diamond', color=color_map[he['id']]),
                text=[he['id']], textposition='top center',
                name=f'Hyperedge {he["id"]}',
                hovertext=[f"Birth Time: {he['birth_time']}"]
            ))
            #--- spokes
            for member in he['members']:
                coord = nodes[member]
                fig.add_trace(go.Scatter3d(
                    x=[cx, coord['x']],
                    y=[cy, coord['y']],
                    z=[cz, coord['z']],
                    mode='lines',
                    line=dict(color=color_map[he['id']], width=2),
                    showlegend=False
                ))
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            title='Cell Division Hypergraph (Full Star‑Expansion)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.write_html(output_full)
        print(f"Saved full hypergraph to {output_full}")

    elif html_save == 'partial':
        if timepoints is None or len(timepoints) != 2:
            raise ValueError("For partial mode, you must provide timepoints=[start, end].")
        start, end = timepoints
        mid = (start + end) / 2
        windows = [(start, mid), (mid, end)]
        titles = [f"{w[0]}–{w[1]}" for w in windows]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type':'scene'}, {'type':'scene'}]],
            subplot_titles=[f"Birth {t}" for t in titles]
        )
        for idx, (w_start, w_end) in enumerate(windows, start=1):
            subset = [he for he in hyperedges if w_start <= he['birth_time'] < w_end]
            involved = {m for he in subset for m in he['members']}
            #--- cells
            fig.add_trace(go.Scatter3d(
                x=[nodes[c]['x'] for c in involved],
                y=[nodes[c]['y'] for c in involved],
                z=[nodes[c]['z'] for c in involved],
                mode='markers',
                marker=dict(size=4, color='lightgray'),
                name='Cells',
                hovertext=list(involved)
            ), row=1, col=idx)
            #--- hyperedges
            for he in subset:
                cx, cy, cz = he['centroid']
                fig.add_trace(go.Scatter3d(
                    x=[cx], y=[cy], z=[cz],
                    mode='markers+text',
                    marker=dict(size=8, symbol='diamond', color=color_map[he['id']]),
                    text=[he['id']], textposition='top center',
                    name=f"Hyperedge {he['id']}",
                    hovertext=[f"Birth Time: {he['birth_time']}"]
                ), row=1, col=idx)
                for m in he['members']:
                    coord = nodes[m]
                    fig.add_trace(go.Scatter3d(
                        x=[cx, coord['x']],
                        y=[cy, coord['y']],
                        z=[cz, coord['z']],
                        mode='lines',
                        line=dict(color=color_map[he['id']], width=2),
                        showlegend=False
                    ), row=1, col=idx)

        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            height=600, width=1000,
            title_text=f"Hypergraph by Birth Time Windows [{start}–{end}]"
        )
        fig.write_html(output_partial)
        print(f"Saved partial hypergraph to {output_partial}")

    else:
        raise ValueError("html_save must be either 'full' or 'partial'.")


# function calling for partial and full data
# save_cell_division_hypergraph('cells_birth_and_pos.csv', html_save='full')
# save_cell_division_hypergraph('cells_birth_and_pos.csv', html_save='partial', timepoints=[0,100])
