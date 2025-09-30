"""
Prediction script for financial time series models
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from model import create_model
from data_loader import prepare_financial_data, create_data_loaders
import config


def load_model(filepath, device='cpu'):
    """
    Load trained model from file
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    model = create_model(**model_config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def make_predictions(model, data_loader, device='cpu', scaler=None):
    """
    Make predictions on data
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return predictions, actuals


def calculate_metrics(predictions, actuals):
    """
    Calculate prediction metrics
    """
    # Flatten arrays if needed
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate directional accuracy (for financial data)
    direction_pred = np.sign(predictions)
    direction_actual = np.sign(actuals)
    directional_accuracy = np.mean(direction_pred == direction_actual)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Directional Accuracy': directional_accuracy
    }


def plot_predictions(predictions, actuals, title="Predictions vs Actual", save_path=None):
    """
    Plot predictions against actual values
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time series plot
    ax1.plot(actuals, label='Actual', alpha=0.8)
    ax1.plot(predictions, label='Predicted', alpha=0.8)
    ax1.set_title(f'{title} - Time Series')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(actuals, predictions, alpha=0.6)
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'{title} - Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def predict_future(model, recent_data, num_predictions=5, device='cpu'):
    """
    Predict future values using the most recent data
    
    Args:
        model: Trained model
        recent_data: Recent sequence of data (sequence_length x features)
        num_predictions: Number of future steps to predict
        device: Device to run on
    """
    model.eval()
    predictions = []
    
    # Convert to tensor and add batch dimension
    current_sequence = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(num_predictions):
            # Make prediction
            pred = model(current_sequence)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            # Update sequence for next prediction
            # This is a simplified approach - you might want to include the prediction
            # in the next input sequence depending on your model design
            # For now, we'll just shift the window
            if len(recent_data) > 1:
                # Shift sequence and add prediction as new data point
                new_row = current_sequence[0, -1, :].clone()
                new_row[0] = pred  # Assuming first feature is the target
                
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_row.unsqueeze(0).unsqueeze(0)
                ], dim=1)
    
    return np.array(predictions)


def main():
    """
    Main prediction function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load trained model
        print("Loading trained model...")
        model, checkpoint = load_model(config.MODEL_PATH, str(device))
        print(f"Model loaded successfully. Training date: {checkpoint.get('training_date', 'Unknown')}")
        
        # Prepare data (same as training)
        print("Preparing data...")
        features_df = prepare_financial_data()
        train_loader, val_loader, scaler = create_data_loaders(
            features_df,
            train_split=config.TRAIN_SPLIT,
            batch_size=config.BATCH_SIZE,
            sequence_length=config.SEQUENCE_LENGTH
        )
        
        # Make predictions on validation set
        print("Making predictions...")
        predictions, actuals = make_predictions(model, val_loader, str(device), scaler)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, actuals)
        print("\nPrediction Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        # Plot results
        plot_predictions(predictions.flatten(), actuals.flatten(), 
                        "Validation Set Predictions", "validation_predictions.png")
        
        # Predict future values
        print("\nPredicting future values...")
        recent_data = features_df.iloc[-config.SEQUENCE_LENGTH:].values
        
        # Normalize recent data using the same scaler
        recent_data_scaled = scaler.transform(recent_data)
        
        future_predictions = predict_future(model, recent_data_scaled, num_predictions=5, device=str(device))
        
        print("Future predictions (next 5 days):")
        for i, pred in enumerate(future_predictions, 1):
            print(f"Day +{i}: {pred:.6f}")
        
        # Plot training history if available
        if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
            plt.figure(figsize=(10, 6))
            plt.plot(checkpoint['train_losses'], label='Training Loss', alpha=0.8)
            plt.plot(checkpoint['val_losses'], label='Validation Loss', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History (from saved model)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('saved_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    except FileNotFoundError:
        print(f"Model file not found at {config.MODEL_PATH}")
        print("Please train a model first using train.py")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
