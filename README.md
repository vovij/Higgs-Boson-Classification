# Charged Higgs Boson Search Using Low-Level Particle Analysis

This repository contains the implementation of machine learning models for analyzing low-level particle information in the search for charged Higgs bosons at the ATLAS experiment.

## Project Overview

This project focuses on developing machine learning techniques to identify charged Higgs boson events in the $H^+ \to W^+h \to W^+b\bar{b}$ decay channel, comparing the performance of different neural network architectures (Convolutional Neural Networks and Transformers) for discriminating signal from background events. The analysis utilizes raw particle four-vectors and particle type information without relying on high-level engineered features.

### Physics Context

The Standard Model of particle physics has been remarkably successful but is believed to be incomplete. Many extensions of the Standard Model, such as Two-Higgs Doublet Models (2HDM) and the Minimal Supersymmetric Standard Model (MSSM), predict the existence of additional Higgs bosons, including charged states ($H^\pm$). The discovery of such particles would provide definitive evidence for physics beyond the Standard Model.

### Decay Channels

We focus on two decay channels of the $W$ boson from the charged Higgs decay:
- **Leptonic decay (lvbb)**: $W^+ \to \ell^+ \nu_\ell$ (selection categories 0, 8, 10)
- **Hadronic decay (qqbb)**: $W^+ \to q\bar{q}$ (selection categories 3, 9)

## Repository Structure

```
├── model_definitions.py    # Neural network architecture definitions
├── preprocess_data.py      # Data preprocessing pipeline
├── read_low_level.py       # Functions for reading low-level particle data
├── train_models.py         # Model training and evaluation
├── training_utils.py       # Utilities for training and evaluation
├── visualization.py        # Visualization scripts for physics interpretations
└── README.md
```

## Data Description

The data consists of simulated events from the ATLAS experiment, including both signal and background processes:

- **Signal samples**: Charged Higgs boson production and decay, with masses ranging from 800 GeV to 3000 GeV (DSIDs 510115-510124)
- **Background samples**: Various Standard Model processes that can mimic the signal signature

Each event contains low-level particle information represented by:
- `ll_particle_px`: x-component of particle momentum
- `ll_particle_py`: y-component of particle momentum
- `ll_particle_pz`: z-component of particle momentum
- `ll_particle_e`: energy of the particle
- `ll_particle_type`: type of the particle

## Models

### Convolutional Neural Network (CNN)

The CNN architecture leverages convolutional layers to extract features from particle sequences, combined with an attention mechanism to focus on the most relevant particles:

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

The Transformer architecture uses multi-head self-attention to capture relationships between particles regardless of their positions in the sequence:

```python
class ParticleTransformer(nn.Module):
    def __init__(self, input_dim=5, max_particles=30, embed_dim=64, num_heads=4, 
                 ff_dim=128, num_layers=3, dropout=0.2, particle_embedding_dim=16):
        super(ParticleTransformer, self).__init__()
        
        # Embedding layers
        self.input_dim = input_dim - 1  # Momentum features (excluding particle type)
        self.particle_embedding = nn.Embedding(num_embeddings=7, embedding_dim=particle_embedding_dim)  
        self.momentum_embedding = nn.Linear(self.input_dim, embed_dim - particle_embedding_dim)
        
        # Transformer encoder
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

This script reads ROOT files, applies selection criteria, and preprocesses the data for both lvbb and qqbb channels.

### Training Models

```bash
python train_models.py --channel lvbb  # Train models for lvbb channel
python train_models.py --channel qqbb  # Train models for qqbb channel
python train_models.py --channel both  # Train models for both channels
```

### Visualization

```bash
python visualization.py
```

This script generates visualizations for model performance analysis and physics interpretation.

## Results

The project demonstrates that neural networks trained on low-level particle information can effectively discriminate between charged Higgs signal events and background processes:

- The CNN model with an attention mechanism achieved strong performance, with AUC values of 0.85-0.87 for the lvbb and qqbb channels.
- The qqbb channel generally showed better performance across most metrics, particularly in terms of precision and overall significance improvement.
- Maximum performance was achieved around 1200-1400 GeV charged Higgs mass, with an estimated significance improvement factor of approximately 3.2-3.4.

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

## Author

Volodymyr Drobot

## Acknowledgments

Special thanks to Professor Blumenschein for guidance and to Sid for providing the dataset and valuable insights into the analysis techniques used in the ATLAS experiment.
