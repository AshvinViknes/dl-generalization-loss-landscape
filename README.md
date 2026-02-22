# NNMHA: Neural Network Loss Landscape Analysis with Label Corruption

## Project Information
- **Group:** F
- **Project:** 12
- **Subject:**  NNMHA WS 2025/26 
- **Topic:** Understanding Generalization of Deep Learning via Loss Landscape
- **Institution:** TU Dresden

## Overview

This project analyzes how neural network loss landscapes change under label corruption. Specifically, it investigates the relationship between:

1. **Label corruption probability** (`p`) - the fraction of training labels randomly corrupted
2. **Basin width** - the "flatness" of the loss landscape around minima
3. **Model generalization** - test accuracy and robustness

The key insight is understanding how noisy labels affect the geometric properties of loss basins and model performance.

## Architecture & Components

### Core Modules

#### 1. **Model** (`model.py`)
Defines the neural network architecture used for training:
- CNN/MLP implementation for MNIST classification
- Configurable layer sizes and activation functions
- Cross-entropy loss for multi-class classification

#### 2. **Main** (`main.py`)
The training pipeline orchestrator:
- Handles data loading with label corruption injection
- Trains models for different corruption levels
- Saves checkpoints and training metrics
- Exports loss landscape data for analysis

**Key workflow:**
```
1. Load MNIST dataset
2. For each corruption level p ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}:
   - Corrupt labels: flip p fraction of training labels randomly
   - Train model from scratch
   - Save trained model and loss values
3. Extract 1D loss landscape around minima
4. Generate CSV files with alpha-loss pairs
```

#### 3. **Landscape** (`landscape.py`)
Loss landscape extraction and analysis:
- Computes 1D slices through the loss surface
- Extracts loss values along random directions
- Fits parabolic approximations
- Generates perturbation data for visualization

#### 4. **Plotter** (`plotter.py`)
Visualization and statistical analysis (see detailed docs below)
A class-based Python module for analyzing and visualizing parabolic loss basin data:

- **Automated CSV loading** - Load and fit quadratic parabolas to loss curves
- **Batch visualization** - Generate multiple views of loss landscapes
- **Statistical analysis** - Compute basin widths at fixed loss levels
- **Flexible output** - Save plots to configurable directories with logging

#### 5. **Utils** (`utils.py`)
Helper functions:
- Data loading and preprocessing
- Metric computations
- Logging utilities
- File I/O operations

## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for full dependencies

### Setup

1. Navigate to the respective folder:
```bash
cd path/to/group_f
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run the Code and what happens behind the screen

Run the main training pipeline by using 
```bash
python main.py
```

### Train Models (Generate Loss Landscapes)

To create models with different label corruption levels:

**What happens:**
- Downloads/loads MNIST dataset
- Trains models for each corruption level: $p \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$
- Saves trained models to `results/`
- Exports loss landscape data to `results/1d_csvs/`
- Generates training logs to `logs/events.jsonl`

**Estimated time:** 5-15 minutes per corruption value (depending on hardware)

### Analyze Loss Landscapes

**What happens:**
- Loads all CSV files from `results/1d_csvs/`
- Fits parabolas to loss curves
- Computes basin widths and intersection heights
- Generates publication-quality plots
- Saves visualizations to `results/plots/`


## Project Structure

```
group_f/
├── model.py                     # Neural network model definitions
├── main.py                      # Training script
├── plotter.py                   # Plotting utilities
├── landscape.py                 # Loss landscape analysis
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   └── MNIST/
│       └── raw/                 # Raw MNIST dataset
│
├── results/
│   ├── summary.csv              # Summary statistics
│   ├── 1d_csvs/                 # Loss curve data
│   │   ├── plot_data_p0.0.csv
│   │   ├── plot_data_p0.2.csv
│   │   ├── plot_data_p0.4.csv
│   │   ├── plot_data_p0.6.csv
│   │   ├── plot_data_p0.8.csv
│   │   └── plot_data_p1.0.csv
│   │
│   └── plots/                   # Generated visualizations
│       ├── parabola_basin_comparison_visible.png
│       ├── parabola_with_alpha0_line.png
│       ├── L_vs_p.png
│       ├── width_vs_p_L1e-07.png
│       ├── loss_line_parabola_p*.png
│       └── ...
│
└── logs/
    └── events.jsonl             # Training logs
```

## Key Analyses

### 1. Parabolic Basin Fitting

Each loss curve is fit to a quadratic:
$$L(\alpha) = A\alpha^2 + B\alpha + C$$

Where:
- $\alpha$ is a learning rate or perturbation parameter
- $(A, B, C)$ are fit parameters
- The vertex gives the basin minimum

### 2. Basin Width Computation

For a fixed loss level $L_0$, the basin width is:
$$W = 2\sqrt{\frac{L_0}{A}}$$

This measures "flatness" - larger widths indicate flatter minima.

### 3. Intersection Heights

At a fixed normalized displacement $\alpha_0$, we compute $L(p)$ for each corruption level $p$.

This reveals how basin geometry changes with label noise.

## Main Findings

The analysis reveals key relationships between label corruption and loss landscape geometry:

1. **Label Corruption Narrows Basins (Hypothesis Refuted)**
   - **Inverse relationship:** Higher label corruption $(p)$ → Narrower basins (smaller $W$)
   - **Implication:** Noisy labels lead to sharper, less robust minima
   - This suggests that clean data produces flatter, more generalized solutions

2. **Clean Labels Produce Wider Basins (Better Generalization)**
   - With minimal label noise $(p \approx 0.0)$: Basins are widest
   - Models trained on clean data find flatter minima, which generalize better
   - As corruption increases: Basin sharpness increases (models memorize noise)

3. **Model Robustness vs. Landscape Flatness**
   - Sharp minima (high corruption): Poor generalization, overfitting
   - Flat minima (low corruption): Better generalization, noise robustness
   - Test accuracy inversely correlates with basin sharpness

4. **Emergence of Loss Landscape Topology Changes**
   - As label corruption increases, the loss landscape structure becomes increasingly fragmented
   - Models struggle to find globally optimal solutions
   - Local minima proliferate with increased noise

### CSV Format

Expected format for `plot_data_p*.csv`:
```
alpha,loss
-0.5,0.0234
-0.4,0.0187
...
0.0,0.0001
...
0.5,0.0234
```

## Data

The project uses **MNIST** dataset with synthetic label corruption:
- Clean labels: $p=0.0$
- Corrupted labels: $p \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$

Each model is trained separately, and loss landscapes are extracted via 1D perturbations.

## References

Related concepts:
- Loss landscape geometry and generalization
- Sharp vs. flat minima (Keskar et al., 2017)
- Label noise robustness in deep learning

## Authors / Contributors

**Group F, Project 12** - TU Dresden - NNMHA WS 2025/26 
Ashvinviknes Saravanaperumal
Giriharan Ravichandran
Omar Sarzoon Mahmood Rahmathullah

To access the presentation , please click on the topic name [Understanding Generalization of Deep Learning via Loss Landscape](https://docs.google.com/presentation/d/14PUad-z1T1AczTItBZmnida7P8tocFck2DBcktp0k0Y/edit?usp=sharing)

## Notes

- Plots are saved with DPI=300 for publication quality
- All calculations use double precision (float64)
- The common window is the smallest range available across all curves
- Log messages provide diagnostic information for debugging
- Default output directory is `results/plots/`
