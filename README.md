# Charged Higgs Boson Search Using Low-Level Particle Analysis with Machine Learning

## Project Overview

This repository contains the implementation of advanced machine learning techniques for searching charged Higgs bosons at the ATLAS experiment using low-level particle analysis. The project focuses on the $H^+ \rightarrow W^+h \rightarrow W^+b\bar{b}$ decay channel, comparing the performance of Convolutional Neural Networks (CNNs) and Transformer-based models for discriminating signal from background events.

## Physics Context

The Standard Model of particle physics has been highly successful but is believed to be incomplete. Many well-motivated theoretical frameworks that extend the Standard Model, such as Two-Higgs-Doublet Models (2HDMs) and the Minimal Supersymmetric Standard Model (MSSM), predict the existence of additional Higgs bosons, including charged states (H±). The discovery of such particles would provide definitive evidence for physics beyond the Standard Model.

### Decay Channels

We analyze two decay modes of the W boson from the charged Higgs decay:
- **Leptonic decay (lvbb)**: $H^+ \rightarrow W^+h \rightarrow \ell^+\nu_\ell b\bar{b}$ (selection categories 0, 8, 10)
- **Hadronic decay (qqbb)**: $H^+ \rightarrow W^+h \rightarrow q\bar{q}'b\bar{b}$ (selection categories 3, 9)

## Data Description

The dataset consists of simulated events from the ATLAS experiment, including:
- **Signal samples**: Charged Higgs boson production with masses ranging from 800 GeV to 3000 GeV (DSIDs 510115-510124)
- **Background samples**: Various Standard Model processes that can mimic the signal signature

Each event contains low-level particle information:
- **Four-momentum components**: px, py, pz, E
- **Particle type**: Encoded as integers (-1: padding, 0: electron, 1: muon, 2: neutrino, 3: large-radius jet, 4: small-radius jet)

> **Note on Data Access**: The raw data files are not included in this repository due to size constraints. The data was generated through ATLAS simulation for research purposes. For access to similar datasets or to learn more about ATLAS public datasets, please visit the [ATLAS Open Data Portal](https://opendata.atlas.cern).

## Repository Structure

```
├── data20250214/         # Raw data files in ROOT format
├── figures/              # Output figures from visualization
├── models/               # Trained model files
├── preprocessed/         # Preprocessed data files
├── results/              # Results from model evaluation
├── model_definitions.py  # Neural network architecture definitions
├── preprocess_data.py    # Data preprocessing pipeline
├── read_low_level.py     # Functions for reading low-level particle data
├── train_models.py       # Model training and evaluation
├── training_utils.py     # Utilities for training and evaluation
├── visualization.ipynb   # Visualization scripts for physics interpretations
└── README.md
```

## Machine Learning Approach

### Feature Representation

Rather than using derived high-level features, we train neural networks directly on the basic kinematic properties of particles (four-momenta and particle types), allowing the models to discover discriminative patterns that might be missed in traditional analyses.

### Convolutional Neural Network (CNN)

The CNN architecture processes the particle sequence through convolutional layers followed by an attention mechanism that identifies the most relevant particles:

```python
class ParticleClassifier(nn.Module):
    def __init__(self, input_shape=(5, 30)):
        super(ParticleClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```

### Transformer Model

The Transformer model leverages multi-head self-attention to capture global relationships between particles regardless of their positions in the sequence:

```python
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim=5, max_particles=30, embed_dim=64, num_heads=4, 
                 ff_dim=128, num_layers=3, dropout=0.2, particle_embedding_dim=16):
        super(ParticleTransformer, self).__init__()
        
        # Particle type embedding
        # We shift the particle types from range (-1,0,1,2,3,4) to (0,1,2,3,4,5)
        # where -1 is padding, 0 is electron, 1 is muon, 2 is neutrino,
        # 3 is large-radius jet, and 4 is small-radius jet
        self.particle_embedding = nn.Embedding(
            num_embeddings=7,  # Add extra index for potential out-of-range values
            embedding_dim=particle_embedding_dim
        )        

        # Momentum feature embedding
        self.momentum_embedding = nn.Linear(self.input_dim, embed_dim - particle_embedding_dim)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```

## Usage

### Data Preprocessing

```bash
python preprocess_data.py
```

This script:
1. Reads ROOT files containing low-level particle data
2. Applies selection criteria (including MET > 30 GeV)
3. Separates events into lvbb and qqbb channels
4. Normalizes momentum components
5. Applies appropriate class weighting
6. Saves preprocessed data for model training

### Training Models

```bash
python train_models.py --channel lvbb  # Train models for lvbb channel
python train_models.py --channel qqbb  # Train models for qqbb channel
python train_models.py --channel both  # Train models for both channels
```

The training procedure incorporates:
- Weighted binary cross-entropy loss
- AdamW optimizer with learning rate scheduling
- Early stopping based on validation loss
- Gradient clipping for training stability
- Label smoothing for the Transformer model

### Visualization

The [`visualization.ipynb`](./visualization.ipynb) notebook contains comprehensive visualizations including:
- Learning curves for model training
- ROC curves for performance evaluation
- Confusion matrices for classification accuracy
- Prediction score distributions for signal and background
- Performance comparisons between architectures and channels

These visualizations provide valuable insights into model performance and physical interpretations of the results.

## Results

### Performance Metrics

| Metric | LVBB CNN | LVBB Transformer | QQBB CNN | QQBB Transformer |
|--------|----------|------------------|----------|------------------|
| Accuracy | 0.9041 | 0.9153 | 0.9310 | 0.9301 |
| ROC AUC | 0.9672 | 0.9729 | 0.9656 | 0.9643 |
| Precision | 0.8974 | 0.9091 | 0.9440 | 0.9428 |
| Recall | 0.9129 | 0.9232 | 0.9612 | 0.9613 |
| F1 Score | 0.9051 | 0.9161 | 0.9525 | 0.9520 |

### Key Findings

- All models achieved exceptional AUC values exceeding 0.96, confirming that both neural network architectures effectively learn meaningful patterns from low-level particle information.
- The Transformer model performed optimally for the leptonic channel (AUC of 0.973, 92.32% signal efficiency), while CNNs excelled in the hadronic channel (96.12% signal efficiency).
- The hadronic decay mode demonstrated consistently higher signal efficiency across both architectures (approximately 96% vs. 91-92% for lvbb), suggesting it may be more promising for charged Higgs discovery.
- The leptonic channel provided superior background rejection (approximately 90% vs. 85% for qqbb), making it valuable for setting stringent limits.
- The complementary strengths of the two channels suggest that a combined analysis would maximize discovery potential across the charged Higgs mass range.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Awkward Array
- Uproot
- Vector

### Installation

You can install all required dependencies with:

```bash
pip install -r requirements.txt
```

The content of `requirements.txt`:

```
torch>=1.7.0
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
awkward>=1.0.0
uproot>=4.0.0
vector>=0.8.0
tqdm>=4.50.0
seaborn>=0.11.0
```

## Author

Volodymyr Drobot

## Acknowledgments

Special thanks to Professor Ulla Blumenschein for guidance and to Sid for providing the dataset and valuable insights into the analysis techniques used in the ATLAS experiment.