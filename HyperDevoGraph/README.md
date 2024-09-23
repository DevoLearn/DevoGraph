# HyperDevoGraph: Modeling C. elegans Development with Hypergraph

## Introduction

This project analyzes raw CE data and cell-phenotype-lineage data from _Caenorhabditis elegans_ (_C. elegans_), focusing on spatial characteristics and developmental stages of cells. The data includes 786 unique cells tracked over time from -16 to 190 time points, detailing their spatial coordinates and size changes. 

The goal is to model and visualize complex relationships between cells during development, capturing both spatial and lineage-based interactions using hypergraphs, a framework representing higher-order relationships where an edge can connect more than two nodes.

## Data Sources

### 1. CE-Raw Data:
- **Source**: `CE_raw_data.csv`
- **Description**: Contains raw cell data with spatial coordinates, cell identifiers, and attributes such as size.
- **Time Points**: From -16 (pre-division stages) to 190, covering key development stages in _C. elegans_ lifecycle.
- **Coordinates**: 3D space (x, y, z) tracking physical development of each cell.
- **Cell Size**: Sizes range from 24 to 103 units, indicating growth and division events.

### 2. Cell-Phenotype-Lineage Data:
- **Source**: `cell-phenotype-lineage-data.xlsx`
- **Description**: Contains lineage information, with cell names and their corresponding mother cells.
- **Mother Cell**: Indicates the parent cell from which the current cell was derived.

## Data Fields
- `cell`: Unique identifier for each cell.
- `time`: Time point of observation.
- `x, y, z`: Spatial coordinates in 3D space.
- `size`: Size of the cell at each time point.
- `mother`: Identifier of the mother cell.

## Data Preparation
- Spatial coordinates are normalized to center around 10 for easier analysis.
- Cells with missing mother data include P1, P2, P3, P4, Z2, and Z3, representing special states in the lineage.
- Mean coordinates and size of each cell are calculated across all time points.

## Analysis

### 1. **Graph Creation**
- **Static Fully-Connected Spatial Graph**: Graph created using mean coordinates of each cell, visualized with the Fruchterman-Reingold layout algorithm.

### 2. **Spatial Hypergraph Creation**
- **Distance Threshold-based Approach**: Cells are connected within a defined spatial proximity (distance threshold). 
- **Clustering-based Hypergraph (DBSCAN)**: Uses DBSCAN clustering to group cells by proximity.

### 3. **Lineage Hypergraph Creation**
- **Lineage-based Hypergraph**: Connects cells based on their common mother links, visualized using the Fruchterman-Reingold layout.

### 4. **Dynamic Hypergraph**
- Conceptual model that dynamically represents changes in cellular relationships over time.

## HypergraphAnalyzer Class

### Initialization:
```python
analyzer = HypergraphAnalyzer(data_dir='data')

## Methods

- `load_data()`: Loads and preprocesses the raw data.
- `create_proximity_hypergraphs(threshold=5)`: Creates hypergraphs based on a spatial distance threshold.
- `create_dbscan_hypergraphs(eps=5, min_samples=2, include_noise=True)`: Creates DBSCAN clustering hypergraphs based on spatial proximity.
- `draw_hypergraphs(title_prefix='Hypergraphs')`: Visualizes the created hypergraphs.

## Workflow

### 1. Data Loading:

```python
analyzer.load_data()
```

### 2. Hypergraph Creation:

- **Proximity Hypergraphs**:

```python
analyzer.create_proximity_hypergraphs(threshold=5)
```
### DBSCAN Hypergraphs:

```python
analyzer.create_dbscan_hypergraphs(eps=5, min_samples=2, include_noise=True)
```
### 3. Visualization:
```python
analyzer.draw_hypergraphs(title_prefix='Proximity Hypergraphs')
analyzer.draw_hypergraphs(title_prefix='DBSCAN Hypergraphs')
```


Python 3.11+ with the following packages:

    numpy
    pandas
    matplotlib
    scipy
    scikit-learn
    HyerNetX


## Data Files

**CE_raw_data.csv:** Contains raw cell data (spatial coordinates, identifiers, etc.).

**cell-phenotype-lineage-data.xlsx:** Contains lineage information (cell names, mother cells).

Ensure that these files are placed in a data directory relative to where the script is run.

### How to Run

1. Place the required data files in a directory named data.
2. Install the required dependencies.
3. Run the script:
```python
python hypergraph_analysis.py
```

## Notes
- You can modify the parameters in create_proximity_hypergraphs and create_dbscan_hypergraphs to experiment with different grouping criteria.
- Ensure that the data files are correctly formatted and located in the data directory.
- The script generates plots using matplotlib. Make sure your environment supports GUI operations if running locally.






