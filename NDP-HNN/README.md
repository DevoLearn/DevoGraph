# MODELING NEURAL DEVELOPMENTAL PROGRAMS OF C. elegans USING GROWING HYPERGRAPH NEURAL NETWORKS 

## NDP-HNN Environment Setup

This guide explains how to create a new conda environment named `ghnn` and install the required Python packages using the provided `requirements.txt`. It is expected to have conda/anaconda distribution already installed in your device or server. 


## Steps

1. Create the conda environment with Python 3.9 (or another version of your choice):

   ```bash
   conda create -n ghnn python=3.9 -y
   ````

2. Activate the environment:

   ```bash
   conda activate ghnn
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Verify the installation:

   ```bash
   python -c "import torch, tqdm, torch_geometric, hypernetx, networkx, pandas, matplotlib, seaborn, umap"
   echo "All packages imported successfully!"
   ```

## Under Construction

> Further details of code and implementation will be provided soon.

