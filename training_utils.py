import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def train_model(model, train_loader, val_loader, channel, model_type, num_epochs=10, 
                patience=3, lr=1e-4, label_smoothing=0.05):
    """Train model with early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Model name for saving
    model_name = f"{channel}_{model_type}"
    
    # Loss function with label smoothing
    def smoothed_bce_loss(pred, target, weights, smoothing=label_smoothing):
        # Apply label smoothing
        smoothed_target = target * (1 - smoothing) + 0.5 * smoothing
        
        # Ensure compatible shapes
        if pred.dim() > target.dim():
            pred = pred.squeeze()
        
        # BCE loss
        loss = F.binary_cross_entropy(pred, smoothed_target, reduction='none')
        
        # Apply weights
        weighted_loss = (loss * weights).mean()
        return weighted_loss
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # LR scheduler with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.2 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # History for learning curves
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = -1
    no_improvement = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_data in train_pbar:
            batch_x, batch_y, batch_weights = [b.to(device) for b in batch_data]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Calculate loss
            loss = smoothed_bce_loss(outputs.squeeze(), batch_y, batch_weights)
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * len(batch_x)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader.dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_data in val_pbar:
                batch_x, batch_y, batch_weights = [b.to(device) for b in batch_data]
                
                outputs = model(batch_x)
                loss = smoothed_bce_loss(outputs.squeeze(), batch_y, batch_weights)
                
                val_loss += loss.item() * len(batch_x)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader.dataset)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            
            # Save the best model
            torch.save(model.state_dict(), f'models/{model_name}.pth')
            print(f"Model saved at epoch {epoch+1}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(f'models/{model_name}.pth'))
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title(f'{model_name} Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_learning_curve.png')
    plt.close()
        
    return model, history

def evaluate_model(model, test_loader, model_name):
    """Evaluate model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and labels
    all_preds = []
    all_targets = []
    all_weights = []
    
    # Get predictions
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            batch_x, batch_y, batch_weights = [b.to(device) for b in batch_data]
            
            outputs = model(batch_x).squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_weights.extend(batch_weights.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_weights = np.array(all_weights)
    
    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, binary_preds, sample_weight=all_weights)
    
    # Add small noise to predictions to break ties
    np.random.seed(42)
    preds_with_noise = all_preds + np.random.normal(0, 1e-7, size=all_preds.shape)
    preds_with_noise = np.clip(preds_with_noise, 1e-10, 1.0 - 1e-10)
    
    # ROC curve calculation
    try:
        fpr, tpr, _ = roc_curve(all_targets, preds_with_noise, sample_weight=all_weights)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"Warning: ROC calculation failed: {e}")
        fpr, tpr = [0, 1], [0, 1]
        roc_auc = 0.5
    
    # Calculate other metrics
    precision = precision_score(all_targets, binary_preds, sample_weight=all_weights, zero_division=0)
    recall = recall_score(all_targets, binary_preds, sample_weight=all_weights, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, sample_weight=all_weights, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, binary_preds, sample_weight=all_weights)
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets,
        'weights': all_weights
    }
    
    # Print metrics
    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Visualizations
    visualize_predictions(metrics, model_name)
    visualize_confusion_matrix(metrics, model_name)
    visualize_roc_curve(metrics, model_name)
    
    return metrics

# Visualization functions
def visualize_predictions(metrics, model_name):
    """Visualize prediction distribution"""
    plt.figure(figsize=(10, 6))
    
    # Get signal and background predictions
    signal_idx = metrics['targets'] == 1
    bg_idx = metrics['targets'] == 0
    
    signal_preds = metrics['predictions'][signal_idx]
    signal_weights = metrics['weights'][signal_idx]
    
    bg_preds = metrics['predictions'][bg_idx]
    bg_weights = metrics['weights'][bg_idx]
    
    # Plot histograms
    plt.hist(signal_preds, bins=50, alpha=0.6, weights=signal_weights, 
             density=True, color='blue', label='Signal')
    plt.hist(bg_preds, bins=50, alpha=0.6, weights=bg_weights, 
             density=True, color='red', label='Background')
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title(f'{model_name} Prediction Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{model_name}_prediction_dist.png')
    plt.close()
    
def visualize_confusion_matrix(metrics, model_name):
    """Visualize confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Background', 'Signal'])
    plt.yticks([0.5, 1.5], ['Background', 'Signal'])
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()
    
def visualize_roc_curve(metrics, model_name):
    """Visualize ROC curve"""
    plt.figure(figsize=(8, 8))
    plt.plot(metrics['fpr'], metrics['tpr'], 
             label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})',
             color='blue', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{model_name}_roc_curve.png')
    plt.close()
    
def visualize_model_comparison(cnn_metrics, transformer_metrics, channel):
    """Compare CNN and Transformer models for a channel"""
    # ROC Curve comparison
    plt.figure(figsize=(10, 8))
    
    plt.plot(cnn_metrics['fpr'], cnn_metrics['tpr'],
            label=f'CNN (AUC = {cnn_metrics["roc_auc"]:.3f})',
            color='blue', linestyle='-', linewidth=2)
    
    plt.plot(transformer_metrics['fpr'], transformer_metrics['tpr'],
            label=f'Transformer (AUC = {transformer_metrics["roc_auc"]:.3f})',
            color='red', linestyle='--', linewidth=2)
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison - {channel.upper()}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{channel}_roc_comparison.png', dpi=300)
    plt.close()
        
    # Prediction Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # CNN predictions
    signal_idx = cnn_metrics['targets'] == 1
    bg_idx = cnn_metrics['targets'] == 0
    
    signal_preds = cnn_metrics['predictions'][signal_idx]
    signal_weights = cnn_metrics['weights'][signal_idx]
    
    bg_preds = cnn_metrics['predictions'][bg_idx]
    bg_weights = cnn_metrics['weights'][bg_idx]
    
    ax1.hist(signal_preds, bins=50, alpha=0.6, color='blue', 
            weights=signal_weights, label='Signal', density=True)
    ax1.hist(bg_preds, bins=50, alpha=0.6, color='red', 
            weights=bg_weights, label='Background', density=True)
    
    ax1.set_title(f'CNN {channel.upper()} Predictions', fontsize=12)
    ax1.set_xlabel('Prediction Score', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Transformer predictions
    signal_idx = transformer_metrics['targets'] == 1
    bg_idx = transformer_metrics['targets'] == 0
    
    signal_preds = transformer_metrics['predictions'][signal_idx]
    signal_weights = transformer_metrics['weights'][signal_idx]
    
    bg_preds = transformer_metrics['predictions'][bg_idx]
    bg_weights = transformer_metrics['weights'][bg_idx]
    
    ax2.hist(signal_preds, bins=50, alpha=0.6, color='blue', 
            weights=signal_weights, label='Signal', density=True)
    ax2.hist(bg_preds, bins=50, alpha=0.6, color='red', 
            weights=bg_weights, label='Background', density=True)
    
    ax2.set_title(f'Transformer {channel.upper()} Predictions', fontsize=12)
    ax2.set_xlabel('Prediction Score', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{channel}_prediction_comparison.png', dpi=300)
    plt.close()
        
    # Print metrics comparison
    print(f"\n===== {channel.upper()} Model Comparison =====")
    print(f"{'Metric':<20} {'CNN':<15} {'Transformer':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {cnn_metrics['accuracy']:.4f}{'':<10} {transformer_metrics['accuracy']:.4f}")
    print(f"{'ROC AUC':<20} {cnn_metrics['roc_auc']:.4f}{'':<10} {transformer_metrics['roc_auc']:.4f}")
    print(f"{'Precision':<20} {cnn_metrics['precision']:.4f}{'':<10} {transformer_metrics['precision']:.4f}")
    print(f"{'Recall':<20} {cnn_metrics['recall']:.4f}{'':<10} {transformer_metrics['recall']:.4f}")
    print(f"{'F1 Score':<20} {cnn_metrics['f1_score']:.4f}{'':<10} {transformer_metrics['f1_score']:.4f}")