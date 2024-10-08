# Topological Data Analysis of C. elegans Using Kepler Mapper

This project leverages the **Kepler Mapper** algorithm to visualize the graph of cell connectivity in *Caenorhabditis elegans* (C. elegans), providing insights into its neural and cellular networks through Topological Data Analysis (TDA).

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Visualization](#visualization)


## Introduction

Topological Data Analysis (TDA) is a method for understanding the shape and structure of data. In this project, we apply TDA to the connectome of *C. elegans*, a model organism in neuroscience and developmental biology.

The **Kepler Mapper** algorithm is a versatile tool that helps in creating visualizations of high-dimensional data. By using this algorithm, we aim to construct a visual representation of the cell connectivity in *C. elegans*, providing an intuitive understanding of its complex neural network.


## Installation

To get started, clone this repository:

```bash
git clone https://github.com/DevoLearn/DevoGraph.git
cd DevoGraph/stage_2/TDA_using_K-Mapper
```


## Usage

Follow these steps to run the analysis:

1. **Prepare the Data**: 
   - Use the provided `cell-by-cell-data-v2.xlsx` file in the same folder, or
   - Upload your own data and update paths in the `kmapper.ipynb` file accordingly.

2. **Run Kepler Mapper**: 
   ```bash
   jupyter notebook kmapper.ipynb
   ```

3. **Visualize the Results**: 
   - Navigate to the `output` folder
   - Open the generated HTML files in a web browser to explore the topological network

**Note**: This `kmapper.ipynb` file was originally run in a Colab environment. If running locally, please update the file paths accordingly.


## Data

The dataset used in this project contains information about the connectivity of cells in C. elegans. The data is provided in an Excel file with multiple sheets, each detailing different aspects of cell connectivity and lineage, out of which `daughter-of-database` sheet has been used.

#### Sheet overview:
- **daughter-of-database**: Contains information about the lineage and connections between cells.

    | CELL NAME | LOCATION | OBJECT     | RELATION    | CELL NAME | LOCATION | 
    |-----------|----------|------------|-------------|-----------|----------|
    | AB	    | nucleus  | subClassOf	| daughter of | P0	      | nucleus  |
    | ABa	    | nucleus  | subClassOf	| daughter of | AB	      | nucleus  |
    | ...       | ...      | ...        | ...         | ...       | ...      |


## Visualization

The visualization generated by Kepler Mapper will be an interactive HTML file. It allows you to explore the clusters and connections within the data, providing insights into the structure of the C. elegans neural network.




