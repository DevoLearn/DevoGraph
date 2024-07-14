# Devoworm CE-Raw & Cell-Phenotype-Lineage Data Analysis

## Introduction
This project explores the raw CE data and cell-phenotype-lineage data from C. elegans, focusing on the spatial characterstics and development stages of cells. The data encompasses 786 unique cells tracked over time from -16 to 190 time points, detailing their spatial coordinates and size changes.

## Data Source
The spatial data is derived from the "CE-Raw" sheet. Access the data sheet here.

The data is derived from the "Daughter-of-Database" sheet, a comprehensive repository of developmental biology data. Access the data sheet here.

## Data Description
CE-Raw Data
    Cell Names: 786 unique identifiers for cells.
    Time Points: Range from -16 (indicative of pre-division stages in C. elegans) to 190. The negative time points correspond to the P1 stage of the C. elegans lifecycle, where cell division has yet to commence.
    Coordinates: Each cell is tracked in a 3D space (x, y, z), providing insights into the physical development of the organism over time.
    Cell Size: Observed sizes range from 24 to 103 units, indicating growth and division events.

Database-of-Daughter Data
    Cell Names: Corresponding to theunique cell identifiers in the CE-Raw data.
    Mother Cell: Indicates the parent cell from which the current cell was derived.

Data Fields

    cell: Identifier for each cell.
    time: Time point of observation.
    x, y, z: Spatial coordinates.
    size: Size of the cell at each time point.

## Data Preparation
Coordinates have been scaled to center around the value 10 to facilitate easier analysis and visualization. Most cells have associated 'mother' data, except for six cells (P1, P2, P3, P4, Z2, Z3) that represent initial or special states in the lineage.

Each cell has been obtained from the CE-Raw data and matched with its corresponding mother cell from the Daughter-of-Database data. This allows for the tracking of cell lineage and developmental stages.

Every single cell has been tracked to obtain its mean coordinates and size across time. The features of each cell include the mean x, y, z coordinates and mean size, and the mother cell from which it originated.

## Analysis

The analysis is structured to decipher the developmental patterns of C. elegans through both static graphs and dynamic hypergraphs:

### 1. Graph Creation

- **Static Fully-Connected Spatial Graph**: Created using the mean coordinates of each cell across different time points. The graph is visualized with the Fruchterman-Reingold layout algorithm to emphasize the spatial relationships between cells. Due to the dense nature of the graph, it can be challenging to interpret.

### 2. Spatial Hypergraph Creation

This approach utilizes hypergraphs to provide a more intuitive representation of spatial relationships:

- **a. Distance Threshold-based Approach**: Hypergraphs are constructed by connecting cells within a specific spatial proximity defined by a distance threshold. This method allows adjustments to the threshold to enhance the hypergraph's interpretability. Visualizations can be found in  [Hypergraph Visualizations Folder](../visualizations/).

- **b. Clustering-based Hypergraph**: Leveraging the DBSCAN clustering algorithm, this method groups cells based on spatial proximity, resulting in fewer hyperedges and cleaner visualizations. The hypergraphs are visualized using the Fruchterman-Reingold layout to highlight spatial relationships. Visualization can be found in [DBSCAN Hypergraph Folder](../visualizations/).

### 3. Lineage Hypergraph Creation 

- **Lineage-based Hypergraph**: Proposed to connect cells based on their common lineage or mother links. This model aims to visualize lineage relationships using the Fruchterman-Reingold layout algorithm, potentially providing insights into the generational development of cells. Visualization can be found in [DBSCAN Hypergraph Folder](../visualizations/).

### 4. Dynamic Hypergraph *(Ideation)*

- **Time-Series Hypergraph**: A conceptual model to use hypergraphs constructed from the mean coordinates of cells across time, aiming to dynamically represent changes and developments in cellular relationships as they occur over time.

Each analytical approach is designed to enhance our understanding of cellular dynamics and developmental biology in C. elegans, utilizing advanced graphical and hypergraphical methods to uncover new insights.

Requirements

Python 3.11+ with the following packages:

    numpy
    pandas
    matplotlib
    scipy
    HyerNetX
