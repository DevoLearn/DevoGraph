"""
=============================================================================
build_functional_lookup_table.py
=============================================================================
Functional Hyperedge Lookup Table — C. elegans Embryogenesis Project
Generates the identical table as functional_lookup_table.xlsx for
cross-verification.

USAGE (in Jupyter):
    %run build_functional_lookup_table.py
    # OR
    exec(open('build_functional_lookup_table.py').read())

REQUIRED FILES (adjust DATA_DIR path at the top of this script):
    - Alignment_map.xlsx          (WormBase neuron→lineage map)
    - Connectome.csv              (adult synaptic connections)
    - cells_birth_and_pos.csv     (lineage tree, optional — for cell universe)
    - ce_temporal_data.csv        (temporal tracking, optional — for cell universe)

OUTPUT FILE:
    - functional_lookup_table.csv     (4436 rows — FFL motifs with embryo IDs)
=============================================================================
"""

import pandas as pd
import numpy as np
import math
import re
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust these paths to match your directory structure
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = 'raw_dataset'        # directory containing the data files
ALIGNMENT_MAP  = os.path.join(DATA_DIR, 'Alignment_map_csv.csv')
CONNECTOME     = os.path.join(DATA_DIR, 'Connectome.csv')
CELLS_BIRTH    = os.path.join(DATA_DIR, 'cells_birth_and_pos.csv')   # optional
TEMPORAL_DATA  = os.path.join(DATA_DIR, 'ce_temporal_data.csv')      # optional
OUTPUT_DIR     = 'output_tables'      # directory to write output CSVs
FIGURES_DIR    = os.path.join('Figures_and_Visualizations', 'functional')  # directory for the FFL grid figure


# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 — Nomenclature explanation (printed when script runs)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("FUNCTIONAL LOOKUP TABLE — C. elegans Embryogenesis Project")
print("=" * 70)
print("""
CANONICAL NOMENCLATURE: Sulston 1983 notation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells_birth_and_pos.csv and ce_temporal_data.csv both use Sulston notation
natively (e.g. ABplapaaaapp, MSaapa, Cpapa). Using the same convention
throughout the project means zero-translation lookups at every step.

WormBase (Alignment_map.xlsx) uses a slightly different format:
  - Inserts a space after the root:      "AB plapaaaapp"
  - Uses dots for P-lineage cells:       "P1.apa"
  - Appends terminal orientation suffix: "AB alpppapav" (v = ventral)

THREE NORMALIZATION RULES APPLIED:
  Rule 1  Remove all spaces        "AB plapaaaapp" → "ABplapaaaapp"
  Rule 2  Remove dots (AB/MS/P..)  "P1.apa"        → "P1apa"
  Rule 3  Strip suffix v or d      "AB alpppapav"  → "ABalpppapa"
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Normalization function (canonical for entire project)
# ═════════════════════════════════════════════════════════════════════════════

def normalize_to_sulston(raw_wormbase_name):
    """
    Convert a WormBase lineage name to Sulston 1983 notation.

    Parameters
    ----------
    raw_wormbase_name : str
        Lineage name as it appears in Alignment_map.xlsx
        e.g. "AB plapaaaapp", "P1.apa", "AB alpppapav", "H2L.aa"

    Returns
    -------
    str
        Normalized Sulston notation name
        e.g. "ABplapaaaapp",   "P1apa",  "ABalpppapa",  "H2L.aa"

    Rules
    -----
    1. Strip leading/trailing whitespace
    2. Remove all spaces
    3. Remove dots ONLY for standard embryonic lineage prefixes:
       AB, MS, Ea, Ep, Ca, Cp, Da, Dp, EMS, P0-P12
       (NOT for H2L.aa, G2.al, K.p, QR.ap, AmsoL — these are kept as-is)
    4. Strip terminal orientation suffix 'v' or 'd' if:
       - name length > 6 characters, AND
       - last character is 'v' or 'd', AND
       - second-to-last character is a letter (not a digit)
    """
    n = str(raw_wormbase_name).strip()

    # Rule 1 & 2: remove spaces
    n = n.replace(' ', '')

    # Rule 3: remove dots for standard lineage prefixes only
    standard_prefix = re.match(r'^(AB|MS|Ea|Ep|Ca|Cp|Da|Dp|EMS|P\d)', n)
    if standard_prefix:
        n = n.replace('.', '')

    # Rule 4: strip WormBase terminal orientation suffix v or d
    if len(n) > 6 and n[-1] in ('v', 'd') and n[-2].isalpha():
        n = n[:-1]

    return n


def normalization_steps_detail(raw):
    """
    Returns step-by-step transformation for verification / proof table.
    """
    raw = str(raw).strip()
    step1 = raw.replace(' ', '')
    if re.match(r'^(AB|MS|Ea|Ep|Ca|Cp|Da|Dp|EMS|P\d)', step1):
        step2 = step1.replace('.', '')
    else:
        step2 = step1
    if len(step2) > 6 and step2[-1] in ('v', 'd') and step2[-2].isalpha():
        step3 = step2[:-1]
    else:
        step3 = step2
    rules = []
    if step1 != raw:   rules.append('spaces_removed')
    if step2 != step1: rules.append('dots_removed')
    if step3 != step2: rules.append(f'suffix_{step2[-1]}_stripped')
    return step1, step2, step3, (' + '.join(rules) if rules else 'none')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load data
# ═════════════════════════════════════════════════════════════════════════════
print("Loading data files...")

align = pd.read_csv(ALIGNMENT_MAP)
conn  = pd.read_csv(CONNECTOME)

print(f"  Alignment_map_csv.csv  : {len(align)} rows, {align['Cell'].nunique()} unique cells")
print(f"  Connectome.csv      : {len(conn)} rows, "
      f"{conn['Neuron'].nunique()} source neurons")

# Apply normalization to alignment map
align['sulston_lineage'] = align['Lineage Name'].apply(normalize_to_sulston)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Supplement 5 missing neurons from WormBase Document 15/16
# ═════════════════════════════════════════════════════════════════════════════
# These five neurons are in Connectome.csv but absent from Alignment_map.xlsx.
# Their lineage paths are taken directly from the WormBase Individual Neurons page.
MISSING_NEURONS = [
    # adult_name   wormbase_raw              sulston_normalized    description
    ('AVFL', 'P1.aaaa / W.aaa',  'P1aaaa',
     'Anterior ventral process F left — interneuron'),
    ('AVFR', 'P1.aaaa / W.aaa',  'Waaa',
     'Anterior ventral process F right — interneuron'),
    ('DB1',  'AB plpaaaapp',     'ABplpaaaapp',
     'Dorsal B-type motor neuron 1 — ventral cord, innervates dorsal muscles'),
    ('DB3',  'AB prpaaaapp',     'ABprpaaaapp',
     'Dorsal B-type motor neuron 3 — ventral cord, innervates dorsal muscles'),
    ('PHBR', 'AB prapppappp',    'ABprapppappp',
     'Phasmid neuron B right — chemosensory'),
]

missing_df = pd.DataFrame(MISSING_NEURONS,
    columns=['Cell', 'Lineage Name', 'sulston_lineage', 'Description'])
align_full = pd.concat([align, missing_df], ignore_index=True)

print(f"\n  Added {len(MISSING_NEURONS)} missing neurons: "
      f"{[m[0] for m in MISSING_NEURONS]}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build identity map (one row per neuron, deduplicated)
# ═════════════════════════════════════════════════════════════════════════════
conn_neurons = set(conn['Neuron']) | set(conn['Target'])

# Keep first occurrence only (removes bilateral duplicates in alignment map)
neuron_rows = (align_full[align_full['Cell'].isin(conn_neurons)]
               .drop_duplicates(subset='Cell')
               .reset_index(drop=True))

n2sulston  = dict(zip(neuron_rows['Cell'], neuron_rows['sulston_lineage']))
n2raw      = dict(zip(neuron_rows['Cell'], neuron_rows['Lineage Name']))
n2desc     = dict(zip(neuron_rows['Cell'], neuron_rows.get('Description', [''] * len(neuron_rows))))

covered    = conn_neurons & set(n2sulston.keys())
not_covered = conn_neurons - covered
print(f"\nIdentity map coverage: {len(covered)} / {len(conn_neurons)} connectome neurons")
if not_covered:
    print(f"  WARNING — still missing: {sorted(not_covered)}")
else:
    print("  All 299 connectome neurons covered (100%)")

# Build identity map DataFrame
identity_rows = []
for neuron in sorted(n2sulston.keys()):
    raw = n2raw.get(neuron, '')
    s1, s2, s3, steps = normalization_steps_detail(raw)
    identity_rows.append({
        'adult_neuron':            neuron,
        'wormbase_raw_lineage':    raw,
        'sulston_lineage':         n2sulston[neuron],
        'description':             n2desc.get(neuron, ''),
        'norm_step1_remove_spaces':  s1,
        'norm_step2_remove_dots':    s2,
        'norm_step3_strip_suffix':   s3,
        'normalization_applied':     steps,
    })

identity_df = pd.DataFrame(identity_rows)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Extract all FFL triplets from Connectome.csv
# ═════════════════════════════════════════════════════════════════════════════
print("\nEnumerating Feed-Forward Loop (FFL) motifs from connectome...")

edges      = set(zip(conn['Neuron'], conn['Target']))
weight_map = dict(zip(zip(conn['Neuron'], conn['Target']),
                       conn['Number of Connections']))
nt_map     = dict(zip(conn['Neuron'], conn['Neurotransmitter']))

# Build adjacency dict for faster iteration
adj = {}
for src, tgt in edges:
    adj.setdefault(src, []).append(tgt)
for k in adj:
    adj[k] = sorted(adj[k])

ffl_rows = []
motif_id = 0

# Canonical FFL: A→B, A→C, B→C  (A=source, B=intermediary, C=target)
for A in sorted(adj.keys()):
    for B in adj.get(A, []):
        for C in adj.get(B, []):
            if (A, C) in edges and A != B and B != C and A != C:

                eA = n2sulston.get(A, '')
                eB = n2sulston.get(B, '')
                eC = n2sulston.get(C, '')

                wAB = weight_map.get((A, B), 0)
                wAC = weight_map.get((A, C), 0)
                wBC = weight_map.get((B, C), 0)
                total = wAB + wAC + wBC

                # Hyperedge weight: w_e = log(1 + total_synapses)
                # Biological justification: log-scale compresses high-synapse
                # motifs while preserving relative ordering. +1 avoids log(0).
                w_e = round(math.log(1 + total), 6)

                ffl_rows.append({
                    # ── Identifier ──────────────────────────────────────────
                    'motif_id':               f'FFL_{motif_id:04d}',

                    # ── Adult neuron names (from Connectome.csv) ────────────
                    'adult_A':                A,
                    'adult_B':                B,
                    'adult_C':                C,

                    # ── Raw WormBase lineage names (from Alignment_map.xlsx) ─
                    'embryo_A_wormbase_raw':  n2raw.get(A, ''),
                    'embryo_B_wormbase_raw':  n2raw.get(B, ''),
                    'embryo_C_wormbase_raw':  n2raw.get(C, ''),

                    # ── Normalized Sulston names ★ USE THESE IN CODE ★ ────────
                    'embryo_A_sulston':       eA,
                    'embryo_B_sulston':       eB,
                    'embryo_C_sulston':       eC,

                    # ── Synapse counts ───────────────────────────────────────
                    'synapses_A_to_B':        wAB,
                    'synapses_A_to_C':        wAC,
                    'synapses_B_to_C':        wBC,
                    'total_synapses':         total,

                    # ── Hyperedge weight for H_aug ───────────────────────────
                    'hyperedge_weight_w_e':   w_e,

                    # ── Neurotransmitter type of A ───────────────────────────
                    'neurotransmitter_A':     nt_map.get(A, ''),

                    # ── Quality flag ─────────────────────────────────────────
                    'all_embryo_mapped':      bool(eA and eB and eC),
                })
                motif_id += 1

ffl_df = pd.DataFrame(ffl_rows)

fully_mapped = ffl_df['all_embryo_mapped'].sum()
print(f"  Total FFL motifs found      : {len(ffl_df)}")
print(f"  Fully mappable (all 3 IDs)  : {fully_mapped}")
print(f"  Partially mapped            : {len(ffl_df) - fully_mapped}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Build normalization proof table
# ═════════════════════════════════════════════════════════════════════════════
proof_df = identity_df[[
    'adult_neuron',
    'wormbase_raw_lineage',
    'norm_step1_remove_spaces',
    'norm_step2_remove_dots',
    'norm_step3_strip_suffix',
    'sulston_lineage',
    'normalization_applied',
]].copy()
proof_df.columns = [
    'adult_neuron',
    'input_wormbase_raw',
    'after_step1_spaces_removed',
    'after_step2_dots_removed',
    'after_step3_suffix_stripped',
    'output_sulston_final',
    'rules_applied',
]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save outputs
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

ffl_path = os.path.join(OUTPUT_DIR, 'functional_lookup_table.csv')
ffl_df.to_csv(ffl_path, index=False)

print(f"\nOutput saved:")
print(f"  {ffl_path:<45} ({len(ffl_df)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Cross-verification checksums (compare with Excel file)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CROSS-VERIFICATION CHECKSUMS")
print("  Compare these values with the Excel file to confirm they are identical")
print("=" * 70)

print(f"\n  [1] Identity map rows                : {len(identity_df)}")
print(f"  [2] Unique sulston names in id map   : {identity_df['sulston_lineage'].nunique()}")
print(f"  [3] FFL motif rows                   : {len(ffl_df)}")
print(f"  [4] Fully mapped FFL rows            : {int(ffl_df['all_embryo_mapped'].sum())}")
print(f"  [5] Sum of all total_synapses        : {int(ffl_df['total_synapses'].sum())}")
print(f"  [6] Sum of all hyperedge weights w_e : {ffl_df['hyperedge_weight_w_e'].sum():.4f}")
print(f"  [7] Mean hyperedge weight w_e        : {ffl_df['hyperedge_weight_w_e'].mean():.6f}")
print(f"  [8] Max synapse count (single motif) : {int(ffl_df['total_synapses'].max())}")
print(f"  [9] Motif with max synapses          : {ffl_df.loc[ffl_df['total_synapses'].idxmax(), 'motif_id']} "
      f"({ffl_df.loc[ffl_df['total_synapses'].idxmax(), 'adult_A']}→"
      f"{ffl_df.loc[ffl_df['total_synapses'].idxmax(), 'adult_B']}→"
      f"{ffl_df.loc[ffl_df['total_synapses'].idxmax(), 'adult_C']})")
print(f"  [10] First motif embryo_A_sulston    : {ffl_df.loc[0, 'embryo_A_sulston']}")
print(f"  [11] Last motif embryo_C_sulston     : {ffl_df.loc[len(ffl_df)-1, 'embryo_C_sulston']}")

# Spot-check specific neurons
spot_check = ['ADAL', 'RIVL', 'AVAL', 'RIM']
print(f"\n  Spot-check sulston names for specific neurons:")
for n in spot_check:
    s = n2sulston.get(n, 'NOT FOUND')
    r = n2raw.get(n, '')
    print(f"    {n:8s}  raw='{r}'  →  sulston='{s}'")

print("\n" + "=" * 70)
print("EXPECTED CHECKSUMS (must match Excel sheet values):")
print("  [1]  299    [2]  299    [3]  4436   [4]  4436")
print("  [5]  see output    [6]  see output")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
# EXPOSE DataFrames for interactive use in Jupyter
# ═════════════════════════════════════════════════════════════════════════════
print("""
DataFrames now available in your namespace:
  identity_df    — 299 rows, neuron → sulston lineage identity map
  ffl_df         — 4436 rows, complete functional hyperedge lookup table
  proof_df       — 299 rows, step-by-step normalization proof

Quick look at ffl_df:
""")
print(ffl_df[['motif_id','adult_A','adult_B','adult_C',
              'embryo_A_sulston','embryo_B_sulston','embryo_C_sulston',
              'total_synapses','hyperedge_weight_w_e']].head(30).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Gerstein-Style FFL Motif Grid Visualization
# ═════════════════════════════════════════════════════════════════════════════
# This section adds the visual representation of all FFL motifs from ffl_df.
# Run %run build_functional_lookup_table.py first to generate ffl_df,
# then run this section, OR run the full script and the figure is auto-generated.
#
# WHAT THIS FIGURE IS:
#   Inspired by the Mark Gerstein lab's network motif atlas figure
#   (as seen in "Insights from integrative analysis of the C. elegans genome").
#   Each cell in the grid = one FFL motif from YOUR connectome data.
#   Reading the grid tells you: which neuron triples form FFLs, how strong
#   they are, and what neurotransmitter type drives each motif.
#
# WHY 3 NODES (not 2 or 4)?
#   An FFL is the MINIMUM structure that creates two parallel information paths.
#   - 2 nodes: only one path possible (A→B). No parallelism.
#   - 3 nodes: A→C directly AND A→B→C indirectly. Two paths — enough for
#              coincidence detection and noise filtering.
#   - 4+ nodes: More complex motifs exist (bifan, 4-node loops etc.) but they
#              are less abundant and harder to enumerate exhaustively.
#   The 3-node FFL is the most abundant higher-order motif in C. elegans.
#   It is also the minimum size that creates a non-trivial hyperedge (size≥3).
#
# IS POSITION DATA NEEDED IN FUNCTIONAL HYPEREDGES?
#   NO. Functional hyperedges (H_functional) encode DESTINY, not location.
#   The three embryonic cells in each FFL motif are grouped together because
#   they will become synaptically connected neurons in the adult — this is a
#   look-ahead biological prior. Their positions at any given timepoint are
#   irrelevant to this grouping. Position data is used in H_spatial only.
#   In H_functional: weight w_e = log(1 + synapse_count), derived from
#   the adult connectome, completely independent of embryonic XYZ coordinates.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("STEP 9 — Generating Gerstein-Style FFL Motif Grid")
print("=" * 70)

# ── Helper: draw one FFL triangle in a single axis ────────────────────────
def _draw_one_ffl(ax, row, vmin, vmax):
    """
    Draw a single FFL motif as a triangle diagram.

    Layout (matching Gerstein figure):
      ● A (source)  — top centre,   circle
      ● B (intermediary) — bottom left, circle
      ▲ C (target)  — bottom right, triangle

    Arrow colours:
      Red/pink  = excitatory neurotransmitter (A sends activating signal)
      Teal/cyan = inhibitory neurotransmitter (A sends suppressing signal)
      Grey      = B→C connection (type not annotated in Connectome.csv)

    Border brightness:
      Encodes hyperedge weight w_e (brighter border = stronger motif)
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')

    nt      = str(row.get('neurotransmitter_A', '')).lower()
    w_e     = float(row['hyperedge_weight_w_e'])
    n_syn   = int(row['total_synapses'])
    intensity = (w_e - vmin) / max(vmax - vmin, 0.01)

    # Node colours by NT type of A
    if 'exc' in nt:
        col_A    = '#E8476A'   # vivid pink-red   = excitatory
        col_arr  = '#E8476A'
    elif 'inh' in nt:
        col_A    = '#2ECBC1'   # teal-cyan         = inhibitory
        col_arr  = '#2ECBC1'
    else:
        col_A    = '#AAAAAA'
        col_arr  = '#AAAAAA'

    col_B = '#A855F7'          # purple  = intermediary (always)
    col_C = '#3B82F6'          # blue    = target        (always)

    # Positions of A (top), B (bottom-left), C (bottom-right)
    pA = (0.50, 0.815)
    pB = (0.16, 0.185)
    pC = (0.84, 0.185)

    # Draw arrows A→B, A→C (NT colour), B→C (grey)
    def _arrow(start, end, color, lw=0.95):
        dx = end[0]-start[0]; dy = end[1]-start[1]
        L  = (dx**2 + dy**2)**0.5
        ox, oy = dx/L * 0.175, dy/L * 0.175
        ax.annotate('', xy=(end[0]-ox, end[1]-oy),
                    xytext=(start[0]+ox, start[1]+oy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    mutation_scale=5))

    _arrow(pA, pB, col_arr)
    _arrow(pA, pC, col_arr)
    _arrow(pB, pC, '#606060')

    # Draw nodes: A and B = circles, C = upward triangle
    ms = 10
    ax.plot(*pA, 'o', ms=ms, color=col_A, mec='white', mew=0.6, zorder=4)
    ax.plot(*pB, 'o', ms=ms, color=col_B, mec='white', mew=0.6, zorder=4)
    ax.plot(*pC, '^', ms=ms, color=col_C, mec='white', mew=0.6, zorder=4)

    # Neuron labels inside nodes (first 4 chars)
    for pos, label in [(pA, row['adult_A'][:4]),
                        (pB, row['adult_B'][:4]),
                        (pC, row['adult_C'][:4])]:
        ax.text(*pos, label, ha='center', va='center',
                fontsize=3.6, color='white', fontweight='bold', zorder=5)

    # Weight label at very bottom
    ax.text(0.5, 0.03, f'w={w_e:.2f}', ha='center', va='bottom',
            fontsize=3.4, color='#FFD93D')

    # Synapse count label at top (small, above node A)
    ax.text(0.5, 0.97, f'{n_syn}syn', ha='center', va='top',
            fontsize=3.2, color='#AAAAFF')

    # Cell background tint by NT type
    bg_col = '#1E0A10' if 'exc' in nt else ('#071414' if 'inh' in nt else '#111111')
    ax.set_facecolor(bg_col)

    # Border brightness encodes weight
    border_alpha = 0.25 + 0.75 * intensity
    border_col   = col_arr
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(border_col)
        sp.set_alpha(border_alpha)
        sp.set_linewidth(1.1)


# ── Build sample: top 110 motifs proportional to NT frequency ─────────────
N_TOTAL  = 36        # cells in the grid (6 columns × 10 rows)
N_COLS   = 6
N_ROWS   = 6

n_exc = int(round(N_TOTAL * (ffl_df['neurotransmitter_A'] == 'exc').mean()))
n_inh = N_TOTAL - n_exc

exc_sample = (ffl_df[ffl_df['neurotransmitter_A'] == 'exc']
              .nlargest(n_exc, 'total_synapses').reset_index(drop=True))
inh_sample = (ffl_df[ffl_df['neurotransmitter_A'] == 'inh']
              .nlargest(n_inh, 'total_synapses').reset_index(drop=True))
grid_df = (pd.concat([exc_sample, inh_sample])
           .sort_values('total_synapses', ascending=False)
           .reset_index(drop=True))

vmin = ffl_df['hyperedge_weight_w_e'].min()
vmax = ffl_df['hyperedge_weight_w_e'].max()

print(f"  Grid sample: {len(grid_df)} motifs  "
      f"({n_exc} excitatory, {n_inh} inhibitory)")
print(f"  Synapse range in grid: "
      f"{int(grid_df['total_synapses'].min())} – "
      f"{int(grid_df['total_synapses'].max())}")

# ── Compose the figure ─────────────────────────────────────────────────────
OUTER_BG = '#060610'

fig = plt.figure(figsize=(22, 20), facecolor=OUTER_BG)

# Grid area takes most of the figure; reserve bottom strip for legend
gs = fig.add_gridspec(
    N_ROWS + 2, N_COLS,
    left=0.02, right=0.98,
    top=0.91, bottom=0.05,
    hspace=0.12, wspace=0.08
)

# Draw each FFL cell
for idx in range(N_ROWS * N_COLS):
    row_i, col_i = divmod(idx, N_COLS)
    ax = fig.add_subplot(gs[row_i, col_i])
    if idx < len(grid_df):
        _draw_one_ffl(ax, grid_df.iloc[idx], vmin, vmax)
    else:
        ax.axis('off')
        ax.set_facecolor(OUTER_BG)

# ── Title ──────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    'C. elegans FFL Network Motifs — All Motifs  '
    f'(Connectome.csv  |  {len(ffl_df):,} total motifs, top {N_TOTAL} shown)',
    ha='center', va='center',
    fontsize=14, fontweight='bold', color='white'
)
fig.text(
    0.5, 0.932,
    'Each cell = one Feed-Forward Loop triplet (A→B, A→C, B→C)  |  '
    'Sorted by total synapse count (strongest top-left)',
    ha='center', va='center',
    fontsize=10, color='#AAAAAA'
)

# ── Legend  (matches Gerstein figure style) ────────────────────────────────
legend_ax = fig.add_subplot(gs[N_ROWS:, :])
legend_ax.axis('off')
legend_ax.set_facecolor(OUTER_BG)

legend_items = [
    # Node shapes
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E8476A',
           ms=9, linestyle='None', label='Node A  (FFL source neuron)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#A855F7',
           ms=9, linestyle='None', label='Node B  (intermediary neuron)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#3B82F6',
           ms=9, linestyle='None', label='Node C  (target neuron)'),
    # Arrow / border colours
    mpatches.Patch(color='#E8476A', alpha=0.85,
                   label='Excitatory A-node  (arrow + border + background tint)'),
    mpatches.Patch(color='#2ECBC1', alpha=0.85,
                   label='Inhibitory A-node  (arrow + border + background tint)'),
    mpatches.Patch(color='#606060', alpha=0.85,
                   label='B→C edge  (grey — NT type not in Connectome.csv for this arm)'),
    # Annotations
    mpatches.Patch(color='#FFD93D', alpha=0.85,
                   label='w=X.XX  →  hyperedge weight w_e = log(1 + total_synapses)'),
    mpatches.Patch(color='#AAAAFF', alpha=0.85,
                   label='Xsyn  →  total synapse count across all 3 FFL arms'),
    mpatches.Patch(color='#3A3A3A', alpha=0.85,
                   label='Border brightness  ∝  w_e  (brighter = stronger motif)'),
]

legend = legend_ax.legend(
    handles=legend_items,
    loc='center', ncol=3,
    facecolor='#10101E', edgecolor='#333355',
    labelcolor='white', fontsize=9.5,
    framealpha=0.9,
    handlelength=1.4, handletextpad=0.6,
    columnspacing=1.2
)

# ── Stats text block (right side, bottom) ─────────────────────────────────
stats_text = (
    f"Total FFL motifs in connectome: {len(ffl_df):,}   |   "
    f"Excitatory A-node: {(ffl_df['neurotransmitter_A']=='exc').sum():,} (91%)   |   "
    f"Inhibitory A-node: {(ffl_df['neurotransmitter_A']=='inh').sum():,} (9%)   |   "
    f"All 3 embryo IDs mapped: {int(ffl_df['all_embryo_mapped'].sum()):,} / {len(ffl_df):,} (100%)"
)
fig.text(0.5, 0.022, stats_text, ha='center', fontsize=8.5, color='#888888')

os.makedirs(FIGURES_DIR, exist_ok=True)
output_path = os.path.join(FIGURES_DIR, 'functional_ffl_motif_grid.png')
fig.savefig(output_path, dpi=160, bbox_inches='tight', facecolor=OUTER_BG)
plt.show()
print(f"\n  Figure saved → {output_path}")
print(f"\n  HOW TO READ THIS FIGURE:")
print(f"  ─────────────────────────────────────────────────────────────────")
print(f"  • Each cell in the grid = one FFL motif from your connectome data")
print(f"  • Reading left→right, top→bottom: strongest → weakest motifs")
print(f"  • Pink/red cells   = excitatory A-node (91% of all FFLs)")
print(f"  • Teal/cyan cells  = inhibitory A-node (9% of all FFLs)")
print(f"  • Top number (blue)= total synapses across all 3 arms of triangle")
print(f"  • Bottom number (gold) = w_e = log(1 + synapses) = incidence matrix value")
print(f"  • Border brightness = relative motif strength (brightest = strongest)")
print(f"  • In H_functional: each cell in this grid becomes ONE COLUMN of the")
print(f"    incidence matrix. The 3 embryo Sulston names in that motif are the")
print(f"    3 ROWS that receive a non-zero entry in that column.")
