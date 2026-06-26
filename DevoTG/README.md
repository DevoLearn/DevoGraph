# DevoTG: Developmental Temporal Graph Networks

## Preprint Alert!
[DevoTG: Temporal Graph Neural Networks for Modeling C. elegans Developmental Connectomics](https://arxiv.org/abs/2606.21940). _arXiv_, 2606.21940.

## 🧬 Project Overview

DevoTG is a comprehensive framework for analyzing C. elegans cell division patterns and connectome development using temporal graph neural networks. This repository combines advanced visualization techniques with state-of-the-art graph neural networks to understand developmental biology through the lens of graph theory and network science.

The project focuses on:
- **Cell Division Analysis**: Understanding spatiotemporal patterns in C. elegans development
- **Connectome Development**: Analyzing neural network formation across developmental stages
- **Temporal Graph Networks**: Applying TGN models to developmental biology data
- **Interactive Visualization**: Creating 3D animations and interactive plots for biological insights
- **Network Analysis**: Comprehensive analysis of temporal network topology and dynamics

## 🚀 Key Features

### 🔬 Data Analysis & Processing
- Comprehensive cell division dataset analysis
- Connectome [dataset](https://github.com/openworm/ConnectomeToolbox/blob/main/cect/data) loading and processing from [ConnectomeToolbox](https://openworm.org/ConnectomeToolbox/Witvliet_2021) given by [Witvliet et al. 2021](https://doi.org/10.1038/s41586-021-03778-8)
- Temporal graph construction (CTDG/DTDG formats)
- Statistical analysis of developmental patterns
- Connection stability analysis (stable, developmental, variable)

### 🧠 Machine Learning
- Temporal Graph Neural Network (TGN) implementation
- Link prediction for cell division events and synaptic connections
- Memory-based graph attention mechanisms
- Training and evaluation pipelines

### 🎨 Advanced Visualization Tools
- Static high-resolution plots for publication
- Interactive 3D visualizations with Plotly
- Sophisticated neural network animations with:
  - Play/pause controls and timeline sliders
  - Detailed hover information for nodes and edges
  - Directional arrows for asymmetric connections
  - Edge width proportional to synaptic weight
  - Bidirectional connection detection
- Network analysis dashboards
- Node importance heatmaps over time
- Connection stability visualizations

### 📊 Network Analysis
- Growth metrics analysis across development
- Centrality measures (degree, betweenness, closeness)
- Network topology metrics (density, clustering, path lengths)
- Community detection algorithms
- Network efficiency analysis

## 📁 Repository Structure

```
DevoTG/
├── README.md                          # This file
├── requirements.txt                   # Package dependencies
├── environment.yml                    # Conda environment file
├── .gitignore                         # Git ignore rules
├── config.yaml                        # Configuration parameters
│
├── devotg/                           # Main package
│   ├── __init__.py
│   ├── data/                           # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset_loader.py         # Cell division dataset loading
│   │   ├── connectome_loader.py      # Connectome dataset processing
│   │   └── temporal_graph_builder.py # CTDG/DTDG construction
│   │
│   ├── models/                       # Neural network models
│   │   ├── __init__.py
│   │   └── tgn_model.py             # TGN implementation for link prediction
│   │
│   ├── visualization/                # Visualization components
│   │   ├── __init__.py
│   │   ├── cell_visualizer.py       # Cell division visualizations
│   │   ├── connectome_visualizer.py # Connectome visualizations
│   │   ├── neural_animator.py       # Advanced network animations
│   │   └── lineage_animator.py      # Lineage tree animations
│   │
│   ├── analysis/                     # Analysis utilities
│   │   ├── __init__.py
│   │   ├── statistics.py            # Statistical analysis
│   │   └── network_analysis.py      # Network topology analysis
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       └── thresholds.py            # Threshold calculations
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_cell_lineage_data_analysis.ipynb       # Dataset understanding
│   ├── 02_cell_lineage_visualization.ipynb       # Visualization examples
│   ├── 03_tgn_training.ipynb                     # Model training
│   └── 04_connectome_development_analysis.ipynb  # Comprehensive connectome analysis
│
├── data/                             # Data directory
│   ├── connectome_datasets/         # Downloaded connectome data
│   ├── cell_lineage_datasets/       # cell lineage data
│   └── processed_datasets/          # Processed data files
│
├── outputs/                          # Generated outputs
│   ├── connectome_analysis/         # Connectome analysis results
│   ├── cell_lineage_analysis/       # Cell lineage analysis results
│   └── models/                      # Saved model analysis results and weights
│
└── scripts/                         # Utility scripts
    ├── run_analysis.py              # Cell division analysis pipeline
    ├── run_connectome_analysis.py   # Connectome analysis pipeline
    └── train_tgn.py                 # Model training script
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended for TGN training)
- FFmpeg (for animation export)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/DevoLearn/DevoGraph.git
cd DevoGraph/DevoTG

# Option 1: Create environment from environment.yml (recommended)
conda env create -f environment.yml
conda activate devotg

# Option 2: Reproduce exact environment (platform-specific)
conda create --name devotg --file requirements.txt
conda activate devotg

# Install package in editable mode
pip install -e .
```

### Development Setup

```bash
# Create a fresh conda environment
conda create -n devotg python=3.12
conda activate devotg

# Install dependencies
pip install -r requirements.txt   # or use environment.yml if you want full reproducibility

# Install in development mode (with dev extras)
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Cell Division Analysis

```python
from devotg.data import DatasetLoader
from devotg.visualization import CellDivisionVisualizer

# Load dataset
loader = DatasetLoader()
df = loader.load_csv("data/cell_lineage_datasets/cells_birth_and_pos.csv")

# Create visualizer
visualizer = CellDivisionVisualizer(df)

# Generate interactive plot
fig = visualizer.create_interactive_plot(color_by='generation')
fig.show()
```

### 2. Connectome Development Analysis

```python
from devotg.data import load_connectome_datasets
from devotg.analysis import analyze_connectome_network
from devotg.visualization import create_neural_network_animation

# Download and process connectome datasets
loader = load_connectome_datasets()

# Perform comprehensive network analysis
analyzer = analyze_connectome_network(
    edges_csv_path="data/processed_datasets/dtdg_edges_temporal.csv",
    nodes_csv_path="data/processed_datasets/dtdg_nodes.csv"
)

# Create interactive animation
fig = create_neural_network_animation(
    nodes_csv_path="data/processed_datasets/dtdg_nodes.csv",
    edges_csv_path="data/processed_datasets/dtdg_edges_temporal.csv", 
    summary_csv_path="data/processed_datasets/dtdg_summary_statistics.csv"
)
fig.show()
```

### 3. Complete Analysis Pipeline

```bash
# Run the complete connectome analysis pipeline
python scripts/run_connectome_analysis.py

# This will:
1. Download connectome datasets
2. Process into temporal graphs
3. Perform network analysis
4. Generate comprehensive visualizations
5. Create interactive animations
```

## 📊 Example Outputs

The repository generates various types of visualizations and analyses:

### Static Visualizations
- **Network Growth Metrics**: Development of connections over time
- **Connection Type Evolution**: Chemical vs electrical synapses
- **Centrality Analysis**: Important neurons at each developmental stage
- **Network Topology**: Directed and weighted network graphs

### Interactive Visualizations
- **Neural Network Development Animation**: Temporal evolution with play/pause controls
- **Network Analysis Dashboard**: Interactive metrics and statistics
- **Node Importance Heatmaps**: Centrality measures across time
- **Network Explorer**: Filterable and interactive network topology

### Analysis Results
- **Network Growth Analysis**: Quantitative developmental metrics
- **Connection Stability**: Classification of connections by temporal patterns
- **Node Importance Evolution**: Centrality measures over development
- **Comprehensive Reports**: JSON format with all analysis results

## 🧪 Advanced Features

### Connectome Analysis
- Processes Witvliet et al. 2021 C. elegans connectome datasets
- Temporal network construction with 8 developmental timepoints
- Neuron classification by functional type (sensory, motor, interneuron, etc.)
- Connection stability analysis (stable, developmental, variable patterns)

### Interactive Animations
- Advanced Plotly-based animations with smooth transitions
- Detailed hover information for nodes and edges
- Directional arrows for asymmetric connections
- Real-time statistics display during animation
- Export capabilities for presentations and publications

### Network Analysis
- Comprehensive topology metrics (density, clustering, path lengths)
- Centrality evolution tracking (degree, betweenness, closeness)
- Community detection and modular structure analysis
- Growth pattern identification and statistical testing

## 📧 Configuration

The `config.yaml` file contains configurable parameters:

```yaml
data:
  feature_dim: 172
  csv_path: "data/cell_lineage_datasets/cells_birth_and_pos.csv"

model:
  memory_dim: 100
  time_dim: 100
  embedding_dim: 100
  learning_rate: 0.001

visualization:
  size_threshold_small: 50.0
  size_threshold_large: 200.0
  export:
    dpi: 300
    format: "png"

training:
  epochs: 20
  batch_size: 200
  val_ratio: 0.15
  test_ratio: 0.15
```

## 📈 Performance

### Model Performance
- **Link Prediction AUC**: ~0.83 on test set
- **Average Precision**: ~0.78 on validation set
- **Training Time**: ~1 minute per 20 epochs on GPU

### Visualization Performance
- **Interactive Plots**: Real-time interaction with 1000+ cells and neurons
- **Animation Export**: HD video generation in <5 minutes
- **Memory Usage**: <2GB RAM for typical datasets
- **Network Layout**: Optimized spring layout algorithms for stability

### Analysis Performance
- **Network Metrics**: Comprehensive analysis of 8 timepoints in <1 minute
- **Connection Stability**: Classification of thousands of connections in seconds
- **Centrality Calculations**: Efficient NetworkX-based computations

## 🔬 Scientific Applications

### Research Use Cases
- **Developmental Biology**: Understanding neural circuit formation
- **Network Neuroscience**: Temporal network analysis of brain development
- **Computational Biology**: Graph-based modeling of biological systems
- **Systems Biology**: Multi-scale analysis of developmental processes

### Educational Applications
- **Interactive Learning**: Visualizations for teaching network neuroscience
- **Research Training**: Complete pipeline for students and researchers
- **Publication Ready**: High-quality figures and animations for papers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Preserve all existing functionality when making changes
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📊 Data Sources

### Primary Datasets
- **Witvliet et al. 2021**: "Connectomes across development reveal principles of brain maturation" - Temporal connectome data
- **C. elegans Cell Division Data**: Spatiotemporal cell division patterns
- **ConnectomeToolbox**: Integration with existing C. elegans data resources

### Data Formats
- **Input**: CSV, Excel, JSON formats
- **Processing**: Temporal graph formats (CTDG/DTDG)
- **Output**: CSV, JSON, HTML, PNG, MP4, SVG formats

## 🏆 Key Features Summary

### Data Processing
- Automatic dataset download and processing
- Multiple data format support
- Robust error handling and validation
- Scalable processing pipelines

### Network Analysis
- Comprehensive topology metrics
- Temporal evolution tracking
- Statistical significance testing
- Connection pattern classification

### Visualization
- Publication-ready static plots
- Interactive web-based visualizations
- Advanced animations with controls
- Customizable styling and themes

### Machine Learning
- Temporal graph neural networks
- Link prediction capabilities
- Memory-based attention mechanisms
- Comprehensive evaluation metrics

## 📚 Citation

If you use DevoTG in your research, please cite:

```bibtex
@software{devotg2025,
  publisher={GitHub},
  title={DevoTG: Developmental Temporal Graph Networks},
  author={Jayadratha Gayen, Mehul Arora, and Bradly Alicea},
  year={2025},
  url={https://github.com/DevoLearn/DevoGraph/tree/main/DevoTG}
}
```

### Related Publications
Please also cite the original data sources:
- Witvliet, Daniel, et al. "Connectomes across development reveal principles of brain maturation." Nature 596.7871 (2021): 257-261.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENCE.md) file for details.

## 🙏 Acknowledgments

- My GSoC 2025 mentors Mehul Arora, Bradly Alicea, Sarrah Bastawala and Jesse Parent for their guidance and support
- The DevoLearn organization for hosting this project
- The Google Open Source Programs team for supporting this project
- The C. elegans developmental biology community 
- Witvliet et al. for providing the connectome datasets
- PyTorch Geometric team for temporal graph tools
- Plotly team for interactive visualization capabilities
- NetworkX developers for network analysis tools
- OpenWorm project for data integration support

## 📞 Contact

- **Email**: jayadratha.gayen@research.iiit.ac.in
- **GitHub**: [@Jayadratha](https://github.com/Jayadratha)
- **Project Link**: [https://github.com/DevoLearn/DevoGraph/tree/main/DevoTG](https://github.com/DevoLearn/DevoGraph/tree/main/DevoTG)

## 🚀 Getting Started Checklist

- [ ] Clone repository and install dependencies
- [ ] Run `python scripts/run_cell_lineage_analysis.py` for cell division analysis
- [ ] Run `python scripts/train_tgn.py` for model training
- [ ] Run `python scripts/run_connectome_analysis.py --skip-download` for connectome DTDG analysis
- [ ] Open generated HTML files in web browser for interactive exploration
- [ ] Review Jupyter notebooks for detailed analysis examples
- [ ] Explore configuration options in `config.yaml`
- [ ] Check `outputs/` directory for all generated files

---

*Built for advancing developmental biology research through computational methods and interactive visualization*
