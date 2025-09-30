"""
Data loading and preprocessing for financial time series analysis
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from api import api_call
import config


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series data
    """
    
    def __init__(self, data, sequence_length, prediction_days=1):
        """
        Args:
            data: DataFrame with financial features
            sequence_length: Number of time steps to look back
            prediction_days: Number of days ahead to predict
        """
        self.data = data.values  # Convert to numpy array
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_days + 1
    
    def __getitem__(self, idx):
        # Input: sequence_length days of features
        x = self.data[idx:idx + self.sequence_length]
        
        # Target: next prediction_days of target feature (e.g., returns)
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_days, 0]  # Assuming first column is target
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def prepare_financial_data(tickers=["AAPL", "MSFT", "TSLA"], start="2025-01-01", end="2025-09-30"):
    """
    Prepare financial data similar for PyTorch
    """
    # Get the data using API
    df = api_call(tickers=tickers, start=start, end=end)
    # Calculate features 
    returns = df.pct_change()
    # Create feature matrix
    features_df = pd.DataFrame()
    # Add returns for primary ticker (AAPL)
    features_df['returns'] = returns['AAPL']
    # Add volatility (20-day rolling std)
    features_df['volatility'] = returns['AAPL'].rolling(20).std()
    # Add correlation with another stock
    features_df['correlation'] = returns['AAPL'].rolling(20).corr(returns['MSFT'])
    # Add more features as needed
    features_df['price_momentum'] = df['AAPL'].pct_change(5)  # 5-day momentum
    features_df['relative_strength'] = returns['AAPL'] / returns['MSFT']
    # Drop NaN values
    features_df = features_df.dropna()
    return features_df


def create_data_loaders(features_df, train_split=0.8, batch_size=32, sequence_length=20):
    """
    Create training and validation data loaders
    """
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df.values)
    scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns, index=features_df.index)
    # Split into train and validation
    split_idx = int(len(scaled_df) * train_split)
    train_data = scaled_df.iloc[:split_idx]
    val_data = scaled_df.iloc[split_idx:]
    # Create datasets
    train_dataset = FinancialDataset(train_data, sequence_length)
    val_dataset = FinancialDataset(val_data, sequence_length)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler


if __name__ == "__main__":
    # Test the data loading
    print("Preparing financial data...")
    features_df = prepare_financial_data()
    print(f"Features shape: {features_df.shape}")
    print(f"Features: {features_df.columns.tolist()}")
    print(f"Data range: {features_df.index[0]} to {features_df.index[-1]}")
    print("\nCreating data loaders...")
    train_loader, val_loader, scaler = create_data_loaders(features_df)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        print(f"Batch input shape: {batch_x.shape}")  # [batch_size, sequence_length, num_features]
        print(f"Batch target shape: {batch_y.shape}")  # [batch_size, prediction_days]
        break
