import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--- load data
df = pd.read_csv('CE_cell_graph_data.csv')
df_sub = df.iloc[:30].copy()

#--- hyperedges by x‐coordinate quartiles
quartiles = df_sub['x'].quantile([0.25, 0.5, 0.75])
def assign_he(x):
    if x <= quartiles[0.25]:
        return 'he1'
    elif x <= quartiles[0.50]:
        return 'he2'
    elif x <= quartiles[0.75]:
        return 'he3'
    else:
        return 'he4'
df_sub['hyperedge'] = df_sub['x'].apply(assign_he)

#--- hyperedge list (extra‐node/star expansion)
hyperedges = []
for he_id, group in df_sub.groupby('hyperedge'):
    members = group['cell'].tolist()
    coords = group[['x','y','z']].values
    centroid = coords.mean(axis=0)
    seq = ''.join(np.random.choice(list('ATCG'), size=10))
    hyperedges.append({
        'id': he_id,
        'members': members,
        'centroid': tuple(centroid),
        'sequence': seq
    })

#--- plott
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_sub['x'], df_sub['y'], df_sub['z'], s=20)
for he in hyperedges: #--- (extra-)nodes and star edges
    cx, cy, cz = he['centroid']
    ax.scatter(cx, cy, cz, s=100, marker='^')
    ax.text(cx, cy, cz, f"{he['id']}: {he['sequence']}", fontsize=8)
    for cell in he['members']:
        x, y, z = df_sub.loc[df_sub['cell'] == cell, ['x','y','z']].values[0]
        ax.plot([cx, x], [cy, y], [cz, z], alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Star‐Expansion Hypergraph (subset)')
plt.tight_layout()
plt.show()
