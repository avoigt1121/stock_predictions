"""
Classification approach - predict up/down instead of exact returns
Often works better for financial data
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
from data_loader import FinancialDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class ClassificationDataset(FinancialDataset):
    """Dataset for classification (up/down prediction)"""
    
    def __getitem__(self, idx):
        # Input: sequence_length days of features
        x = self.data[idx:idx + self.sequence_length]
        
        # Target: 1 if next day return > 0, 0 otherwise
        next_return = self.data[idx + self.sequence_length, 0]  # First column is returns
        y = 1.0 if next_return > 0 else 0.0
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


def train_classification_model():
    """Train a classification model for up/down prediction"""
    print("=== Classification Training (Up/Down Prediction) ===")
    
    device = torch.device('cpu')
    
    # Get enhanced features
    features_df = prepare_enhanced_financial_data()
    print(f"Data shape: {features_df.shape}")
    
    # Use top correlated features
    target_col = 'returns_1d'
    correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.head(11).index.tolist()
    features_df = features_df[top_features]
    
    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df.values)
    scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns, index=features_df.index)
    
    # Create train/validation split
    train_size = int(0.8 * len(scaled_df))
    train_data = scaled_df.iloc[:train_size]
    val_data = scaled_df.iloc[train_size:]
    
    # Create classification datasets
    sequence_length = 15
    train_dataset = ClassificationDataset(train_data, sequence_length)
    val_dataset = ClassificationDataset(val_data, sequence_length)
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    
    # Check class balance
    train_labels = []
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        train_labels.append(y.item())
    
    up_ratio = np.mean(train_labels)
    print(f"Up days ratio: {up_ratio:.3f} (balanced is 0.5)")
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (output 1 for binary classification)
    input_size = features_df.shape[1]
    model = create_model(
        'lstm',
        input_size=input_size,
        hidden_size=32,
        num_layers=2,
        output_size=1,  # Single output for binary classification
        dropout=0.3
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use BCE loss for classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Starting classification training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'features': features_df.columns.tolist(),
                'scaler': scaler
            }, 'models/classification_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Classification Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy', alpha=0.8)
    ax2.plot(val_accuracies, label='Validation Accuracy', alpha=0.8)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print("Model saved to models/classification_model.pth")
    
    if best_val_acc > 0.55:
        print("Model better than random")
    elif best_val_acc > 0.52:
        print("Model at random")
    else:
        print("Model not better than random")
    
    return best_val_acc


if __name__ == "__main__":
    train_classification_model()
