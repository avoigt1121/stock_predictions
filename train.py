"""
Training script for financial time series models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import os

from model import create_model
from data_loader import prepare_financial_data, create_data_loaders
import config


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    """
    Train the model
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def plot_training_history(train_losses, val_losses, save_path='training_history.png', show_plot=False):
    """
    Plot training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close figure to free memory


def save_model(model, filepath, train_losses, val_losses, model_config):
    """
    Save model and training information
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': model_config,
        'training_date': datetime.now().isoformat()
    }, filepath)
    
    print(f"Model saved to {filepath}")


def main():
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing financial data...")
    features_df = prepare_financial_data()
    train_loader, val_loader, scaler = create_data_loaders(
        features_df, 
        train_split=config.TRAIN_SPLIT,
        batch_size=config.BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # Create model
    input_size = features_df.shape[1]  # Number of features
    model_config = {
        'model_type': 'lstm',
        'input_size': input_size,
        'hidden_size': config.HIDDEN_SIZE,
        'num_layers': config.NUM_LAYERS,
        'output_size': config.PREDICTION_DAYS,
        'dropout': config.DROPOUT
    }
    
    model = create_model(**model_config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=str(device)
    )
    
    # Plot results
    plot_training_history(train_losses, val_losses)
    
    # Save model
    save_model(model, config.MODEL_PATH, train_losses, val_losses, model_config)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
