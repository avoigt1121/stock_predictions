"""
Configuration file for PyTorch financial data analysis
"""
import torch

# Data parameters
SEQUENCE_LENGTH = 20  # Number of days to look back for prediction
PREDICTION_DAYS = 1   # Number of days to predict ahead
TRAIN_SPLIT = 0.8     # Percentage of data for training

# Model parameters
HIDDEN_SIZE = 64      # LSTM hidden layer size
NUM_LAYERS = 2        # Number of LSTM layers
DROPOUT = 0.2         # Dropout rate for regularization

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Features to use (you can modify this based on your analysis.py)
FEATURES = ['returns', 'volatility', 'correlation']
TARGET = 'returns'  # What we want to predict

# Model save path
MODEL_PATH = 'models/financial_model.pth'
