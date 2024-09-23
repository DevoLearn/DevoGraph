# Growing Graph Neural Network (GNN) for Developmental Biology

This project introduces a novel approach to modeling dynamic biological systems using Growing Graph Neural Networks (GNNs). This implementation simulates cellular development by dynamically adding nodes to the graph based on birth times, mirroring the process of cell division. The network not only grows temporally but also learns spatial relationships, positioning new "daughter cells" accurately within the graph structure.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Visualization](#visualization)

## Introduction

By combining temporal growth with spatial learning, this Growing GNN offers a powerful tool for studying complex developmental processes, potentially revolutionizing our understanding of embryogenesis and tissue formation. This project was developed as part of a Google Summer of Code (GSoC) 2024 project with DevoLearn.

## Features

- Dynamic node addition based on cell birth times
- Spatial learning for accurate positioning of new cells
- Visualization of graph growth over time
- Application to C. elegans developmental data

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/DevoLearn/DevoGraph.git
cd DevoGraph/Growing-GNN
```

Create a virtual environment and install the necessary packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

Follow these steps to run the analysis:

1. **Prepare the Data**: Use the provided `cells_birth_and_pos.csv` file or upload your own data and update paths in the `GrowingGNN.ipynb` file accordingly.

2. **Run the Model**: Execute the `GrowingGNN.ipynb` notebook:
   ```bash
   jupyter notebook GrowingGNN.ipynb
   ```

3. **Results**: 
   - A gif of the generated graph will be saved as `growing_graph.gif`. 
   - The final state of the graph is saved as an image `final_state.png`.

**Note**: If running in Google Colab or other platforms, please update file paths accordingly.

## Data

The dataset contains information about cells in C. elegans, including:
- 3D position coordinates (x, y, z)
- Parent and daughter cell relationships
- Birth times of daughter cells (division time of parent cells)

- Data is provided as a CSV file with the following structure:


    | Parent Cell |   parent_x   |   parent_y   |   parent_z   | Daughter 1 | Daughter 2 | Birth Time |
    |-------------|--------------|--------------|--------------|------------|------------|------------|
    | P0	      | 422.07777778 | 248.31666666	| 14.326666666 | AB         | P1         | 0          |
    | AB	      | 317.785053825| 251.80042565	| 14.700365336 | ABa	    | ABp	     | 17         |
    | P1	      | 445.3393565  | 253.86056338	| 15.238196116 | EMS	    | P2	     | 18         |
    | ...         | ...          | ...          | ...          | ...        | ...        | ...        |


## Visualization

The project generates two main visualizations:
1. `growing_graph.gif`: An animated gif showing the growth of the graph over time.
2. `final_state.png`: A static image of the final graph state.

These visualizations help in understanding the temporal and spatial development of the cellular structure.