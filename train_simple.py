"""
Simple improved training script that works with limited data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import create_model
from enhanced_features import prepare_enhanced_financial_data
from custom_losses import get_loss_function
from data_loader import FinancialDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def create_simple_data_loaders():
    """Create data loaders with minimal preprocessing to avoid data loss"""
    print("Preparing financial data...")
    
    # Get enhanced features but with minimal preprocessing
    features_df = prepare_enhanced_financial_data()
    print(f"Initial features shape: {features_df.shape}")
    
    # Use a smaller sequence length for limited data
    sequence_length = 15  # Reduced from 30
    
    # Calculate how many sequences we can create
    available_sequences = len(features_df) - sequence_length
    print(f"Available sequences with length {sequence_length}: {available_sequences}")
    
    if available_sequences < 20:
        print("Not enough data for training. Consider:")
        print("1. Using a longer date range")
        print("2. Using simpler features")
        print("3. Reducing sequence length further")
        return None, None, None, None
    
    # Simple feature selection - keep top 10 most correlated features
    target_col = 'returns_1d'
    correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.head(11).index.tolist()  # 10 features + target
    features_df = features_df[top_features]
    
    print(f"Selected features: {top_features}")
    print(f"Final features shape: {features_df.shape}")
    
    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df.values)
    scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns, index=features_df.index)
    
    # Create train/validation split manually
    train_size = int(0.8 * len(scaled_df))
    train_data = scaled_df.iloc[:train_size]
    val_data = scaled_df.iloc[train_size:]
    
    print(f"Train data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create datasets
    train_dataset = FinancialDataset(train_data, sequence_length)
    val_dataset = FinancialDataset(val_data, sequence_length)
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    
    # Use smaller batch size for small datasets
    batch_size = min(8, len(train_dataset) // 2)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler, features_df


def train_simple_model():
    """Train model with simple, robust approach"""
    print("=== Simple Improved Training ===")
    
    # Import pandas here since we need it
    
    device = torch.device('cpu')  # Use CPU for simplicity
    print(f"Using device: {device}")
    
    # Create data loaders
    result = create_simple_data_loaders()
    if result[0] is None:
        print("Training failed - not enough data")
        return
    
    train_loader, val_loader, scaler, features_df = result
    
    # Create smaller model for limited data
    input_size = features_df.shape[1]
    model = create_model(
        'lstm',
        input_size=input_size,
        hidden_size=32,  # Smaller model
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use directional loss for better financial prediction
    criterion = get_loss_function('directional', mse_weight=0.7, directional_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
    
    # Training loop
    num_epochs = 50  # Fewer epochs for small dataset
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'features': features_df.columns.tolist(),
                'scaler': scaler
            }, 'models/simple_improved_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Simple Improved Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('simple_improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved to simple_improved_training_history.png")
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    print("Model saved to models/simple_improved_model.pth")
    
    # Quick evaluation
    print("\n=== Quick Evaluation ===")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.numpy())
    
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    directional_accuracy = np.mean(pred_direction == target_direction)
    
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.1f}%)")
    
    if directional_accuracy > 0.5:
        print("✅ Model is better than random at predicting direction!")
    else:
        print("❌ Model is not better than random. Consider:")
        print("   - More data")
        print("   - Different features")
        print("   - Different model architecture")


if __name__ == "__main__":
    train_simple_model()
