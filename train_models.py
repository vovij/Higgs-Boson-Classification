import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import argparse

# Import model and training functions
from model_definitions import ParticleClassifier, ParticleTransformer, ParticleDataset
from training_utils import train_model, evaluate_model, visualize_model_comparison

class ParticleDataset(Dataset):
    def __init__(self, x_data, y_data, weights=None):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.x_data[idx], self.y_data[idx], self.weights[idx]
        return self.x_data[idx], self.y_data[idx]

def load_data(channel):
    """Load preprocessed data for a specific channel"""
    print(f"Loading {channel.upper()} data...")
    
    # Load data from preprocessed file
    data = np.load(f"preprocessed/{channel}_data.npz")
    x = data['x_particles']
    y = data['y_labels']
    weights = data['weights']
    
    print(f"Data loaded - X shape: {x.shape}, Y shape: {y.shape}")
    return x, y, weights

def prepare_data_loaders(x, y, weights, batch_size=32):
    """Prepare train, validation, and test data loaders"""
    # Split data into train, validation, and test sets
    test_size = 0.15
    val_size = 0.15
    
    # Calculate adjusted validation size
    val_size_adjusted = val_size / (1 - test_size)
    
    # Split data
    x_temp, x_test, y_temp, y_test, w_temp, w_test = train_test_split(
        x, y, weights, test_size=test_size, random_state=42, stratify=y
    )
    
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x_temp, y_temp, w_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = ParticleDataset(x_train, y_train, w_train)
    val_dataset = ParticleDataset(x_val, y_val, w_val)
    test_dataset = ParticleDataset(x_test, y_test, w_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Dataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_channel_models(channel):
    """Train and evaluate models for a specific channel"""
    print(f"\n==== Processing {channel.upper()} Channel ====")
    
    # Load data
    x, y, weights = load_data(channel)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(x, y, weights)
    
    # Model directory
    os.makedirs("models", exist_ok=True)
    
    # Define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CNN parameters
    cnn_params = {'input_shape': (5, 30)}
    
    # Transformer parameters
    transformer_params = {
        'input_dim': 5,
        'max_particles': 30,
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_layers': 3,
        'dropout': 0.2
    }
    
    # Different training parameters for each channel
    if channel == 'lvbb':
        # Standard LVBB training parameters
        cnn_lr = 1e-3
        transformer_lr = 5e-5
        smoothing = 0.05
        transformer_epochs = 15
    else:  # qqbb
        # Enhanced QQBB training parameters with balanced data
        cnn_lr = 5e-4
        transformer_lr = 2e-5
        smoothing = 0.1
        transformer_epochs = 20
    
    # Initialize and train CNN model
    print(f"\n==== Training {channel.upper()} CNN Model ====")
    cnn_model = ParticleClassifier(**cnn_params).to(device)
    
    # Check if model exists
    model_path = f"models/{channel}_cnn.pth"
    if os.path.exists(model_path):
        print(f"Loading existing CNN model from {model_path}")
        cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new CNN model")
        cnn_model, _ = train_model(
            model=cnn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            channel=channel, 
            model_type="cnn",
            num_epochs=3,
            patience=3,
            lr=cnn_lr
        )
    
    # Evaluate CNN model
    cnn_metrics = evaluate_model(cnn_model, test_loader, f"{channel}_cnn")

    # Initialize and train Transformer model
    print(f"\n==== Training {channel.upper()} Transformer Model ====")
    transformer_model = ParticleTransformer(**transformer_params).to(device)
    
    # Check if model exists
    model_path = f"models/{channel}_transformer.pth"
    if os.path.exists(model_path):
        print(f"Loading existing Transformer model from {model_path}")
        transformer_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new Transformer model")
        transformer_model, _ = train_model(
            model=transformer_model,
            train_loader=train_loader,
            val_loader=val_loader,
            channel=channel,
            model_type="transformer",
            num_epochs=transformer_epochs,
            patience=5,
            lr=transformer_lr,
            label_smoothing=smoothing
        )
    
    # Evaluate Transformer model
    transformer_metrics = evaluate_model(transformer_model, test_loader, f"{channel}_transformer")
    
    # Compare models
    visualize_model_comparison(cnn_metrics, transformer_metrics, channel)
    
    return cnn_metrics, transformer_metrics

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train particle physics models')
    parser.add_argument('--channel', type=str, choices=['lvbb', 'qqbb', 'both'], 
                        default='both', help='Channel to process')
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Store all metrics
    all_metrics = {}
    
    # Process requested channel(s)
    if args.channel in ['lvbb', 'both']:
        all_metrics['lvbb'] = train_channel_models('lvbb')
    
    if args.channel in ['qqbb', 'both']:
        all_metrics['qqbb'] = train_channel_models('qqbb')
    
    # If both channels were processed, compare them
    if len(all_metrics) > 1:
        print("\n==== Channel Performance Comparison ====")
        print(f"{'Metric':<20} {'LVBB CNN':<15} {'LVBB Trans.':<15} {'QQBB CNN':<15} {'QQBB Trans.':<15}")
        print("-" * 80)
        
        metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            lvbb_cnn = all_metrics['lvbb'][0][metric]
            lvbb_trans = all_metrics['lvbb'][1][metric]
            qqbb_cnn = all_metrics['qqbb'][0][metric]
            qqbb_trans = all_metrics['qqbb'][1][metric]
            
            print(f"{metric.capitalize():<20} {lvbb_cnn:.4f}{'':<10} {lvbb_trans:.4f}{'':<10} {qqbb_cnn:.4f}{'':<10} {qqbb_trans:.4f}")
    
    print("\nTraining and evaluation complete!")