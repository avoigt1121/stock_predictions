"""
Improved configuration for better model performance
"""
import torch

# Data parameters
SEQUENCE_LENGTH = 30  # Increased from 20 - more historical context
PREDICTION_DAYS = 1   
TRAIN_SPLIT = 0.8     

# Model parameters - Larger model for more complex patterns
HIDDEN_SIZE = 128     # Increased from 64
NUM_LAYERS = 3        # Increased from 2
DROPOUT = 0.3         # Increased dropout for regularization

# Training parameters - More careful training
BATCH_SIZE = 16       # Smaller batch size for better gradients
LEARNING_RATE = 0.0005  # Lower learning rate for stability
NUM_EPOCHS = 200      # More epochs with early stopping
WEIGHT_DECAY = 1e-5   # L2 regularization

# Advanced training techniques
GRADIENT_CLIP = 1.0   # Gradient clipping to prevent exploding gradients
SCHEDULER_PATIENCE = 10  # Learning rate scheduler patience
SCHEDULER_FACTOR = 0.5   # Learning rate reduction factor

# Early stopping
EARLY_STOPPING_PATIENCE = 25  # More patience for early stopping
EARLY_STOPPING_MIN_DELTA = 1e-6

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss function options
LOSS_FUNCTION = 'mse'  # Options: 'mse', 'mae', 'huber', 'directional'

# Data preprocessing
FEATURE_SCALING = 'standard'  # Options: 'standard', 'minmax', 'robust'
OUTLIER_THRESHOLD = 3.0  # Remove outliers beyond 3 std devs

# Model ensemble (optional)
ENSEMBLE_SIZE = 3  # Train multiple models and average predictions

# Features to use
TARGET_FEATURE = 'returns_1d'  # What we want to predict
USE_ENHANCED_FEATURES = True    # Use enhanced_features.py instead of basic features

# Model save paths
MODEL_PATH = 'models/enhanced_financial_model.pth'
ENSEMBLE_PATH = 'models/ensemble_models/'

# Validation strategy
VALIDATION_STRATEGY = 'time_split'  # Options: 'time_split', 'walk_forward'
WALK_FORWARD_STEPS = 5  # For walk-forward validation

# Feature selection
FEATURE_SELECTION = True
MAX_FEATURES = 15  # Maximum number of features to use
FEATURE_SELECTION_METHOD = 'correlation'  # Options: 'correlation', 'mutual_info'
