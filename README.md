# Financial Data Analysis with PyTorch

A complete PyTorch-based toolkit for financial time series analysis and stock price prediction using machine learning.

## Overview

This project implements neural networks (LSTM and Transformer) to predict stock market movements using technical indicators and cross-asset features. The system achieved **55.6% directional accuracy** in predicting whether stock prices will go up or down.

## Project Structure

```
financial_data/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── api.py                             # Stock data fetching (yfinance)
├── analysis.py                        # Basic financial analysis
├── sparse_time_series.py              # Time series utilities
│
├──  Core PyTorch Implementation
├── config.py                          # Basic configuration
├── config_improved.py                 # Advanced hyperparameters
├── model.py                           # Neural network architectures
├── data_loader.py                     # PyTorch dataset handling
├── custom_losses.py                   # Financial-specific loss functions
├── enhanced_features.py               # Technical indicator calculation
│
├──  Training Scripts
├── train.py                           # Basic training script
├── train_simple.py                    # Regression approach (exact returns)
├── train_classification.py            # Classification approach (up/down) 
│
├──  Analysis & Prediction
├── predict.py                         # Make predictions with trained models
├── analyze_prediction_strategies.py   # Data analysis and recommendations
│
└──  models/                         # Saved PyTorch models
    ├── financial_model.pth            # Basic model
    ├── simple_improved_model.pth      # Simple improved model
    └── classification_model.pth       # Best performing model 
```

##  Features & Data (What I used)

### Stock Data Sources
- **Primary ticker**: AAPL (Apple Inc.)
- **Comparison tickers**: MSFT (Microsoft), TSLA (Tesla)
- **Data range**: January 2025 - September 2025 (8 months) (I recommend longer time-frame)
- **Frequency**: Daily prices

### Technical Indicators (25 features)
- **Returns**: 1-day, 3-day, 5-day, 10-day momentum
- **Volatility**: Rolling standard deviation (5d, 10d, 20d windows)
- **Moving Averages**: SMA(5), SMA(10), SMA(20) and price ratios
- **RSI**: Relative Strength Index (14-day)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Position within bands
- **Cross-asset**: Correlations with MSFT and TSLA
- **Lag features**: Previous day's values

### Data Preprocessing
1. **Feature Engineering**: Calculate 25 technical indicators
2. **Feature Selection**: Keep top 11 most correlated with returns
3. **Normalization**: StandardScaler (mean=0, std=1)
4. **Sequence Creation**: 15-day windows for LSTM input
5. **Train/Test Split**: 80% training, 20% validation (time-based) -- Need more data, often ran out of data after removing bad data

##  Model Architectures

### LSTM Model (Primary)
```python
# Configuration
Input Size: 11 features
Hidden Size: 32 units
Layers: 2 LSTM layers
Dropout: 0.3
Output: Binary classification (up/down)
Parameters: ~14,241
```

### Transformer Model (Alternative)
- Multi-head attention mechanism
- Positional encoding for time series
- More complex but potentially better for long sequences

## Performance Results

### Classification Model (Best)
- **Validation Accuracy**: 55.6%
- **Baseline (Random)**: 50.0%
- **Improvement**: +5.6 percentage points
- **Training Time**: ~17 epochs with early stopping

### Regression Models
- **Simple Model RMSE**: 0.80
- **Improved Model RMSE**: 0.85
- **Directional Accuracy**: ~50% (random level)

### Custom Loss Functions
```python
# DirectionalLoss: Penalizes wrong direction more than magnitude
DirectionalLoss(mse_weight=0.7, directional_weight=0.3)

# HuberLoss: Robust to outliers (market crashes)
HuberLoss(delta=1.0)

# QuantileLoss: For confidence intervals
QuantileLoss(quantile=0.5)
```

### Data Handling
- **Memory**: All data loaded in RAM (~32KB for current dataset)
- **Scaling**: Automatic handling for larger datasets
- **Real-time**: Can be extended for live trading

### Improvements:
 **Fix Data Leakage**:
```python
# Normalize only on training data
scaler.fit(train_data)
val_scaled = scaler.transform(val_data)
```

## Usage Examples

### Basic Prediction
```python
from enhanced_features import prepare_enhanced_financial_data
import torch

# Load data
features_df = prepare_enhanced_financial_data()

# Load trained model
model = torch.load('models/classification_model.pth')

# Make prediction
prediction = model(recent_data)
print(f"Tomorrow's direction: {'UP' if prediction > 0.5 else 'DOWN'}")
```

### Custom Training
```python
# Modify hyperparameters in config_improved.py
HIDDEN_SIZE = 64          # Larger model
SEQUENCE_LENGTH = 30      # Longer memory
LEARNING_RATE = 0.0001    # Slower learning

# Run training
python train_improved.py
```


