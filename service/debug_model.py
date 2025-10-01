#!/usr/bin/env python3
"""
Debug script to test model loading
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import FinancialLSTM
    import config
    print("✅ Successfully imported model and config")
    print(f"Config features: {config.FEATURES}")
    print(f"Config hidden size: {config.HIDDEN_SIZE}")
    print(f"Config num layers: {config.NUM_LAYERS}")
    print(f"Config prediction days: {config.PREDICTION_DAYS}")
    print(f"Config dropout: {config.DROPOUT}")
except Exception as e:
    print(f"❌ Error importing: {e}")
    sys.exit(1)

# Try to load the model
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'financial_model.pth')
    print(f"Model path: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    # Load the checkpoint to get the model configuration
    checkpoint = torch.load(model_path, map_location='cpu')
    print("✅ Checkpoint loaded successfully")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get model configuration from checkpoint
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        input_size = model_config['input_size']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        output_size = model_config['output_size']
        dropout = model_config['dropout']
        print(f"✅ Using model config from checkpoint: input_size={input_size}")
    else:
        # Fallback to config.py values
        input_size = len(config.FEATURES)
        hidden_size = config.HIDDEN_SIZE
        num_layers = config.NUM_LAYERS
        output_size = config.PREDICTION_DAYS
        dropout = config.DROPOUT
        print(f"✅ Using fallback config: input_size={input_size}")
    
    # Initialize model with the correct parameters
    model = FinancialLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )
    print("✅ Model initialized successfully with correct input size")
    
    # Extract the actual model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found model_state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("✅ Using checkpoint as state_dict directly")
    
    print(f"State dict keys: {list(state_dict.keys())}")
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded and set to eval mode successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
