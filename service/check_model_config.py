#!/usr/bin/env python3
"""
Check what features the saved model was trained with
"""

import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'financial_model.pth')
checkpoint = torch.load(model_path, map_location='cpu')

print("Checkpoint contents:")
for key, value in checkpoint.items():
    if key == 'model_config':
        print(f"{key}: {value}")
    elif key in ['train_losses', 'val_losses']:
        print(f"{key}: {len(value)} values")
    else:
        print(f"{key}: {type(value)}")

if 'model_config' in checkpoint:
    config = checkpoint['model_config']
    print(f"\nOriginal model input size: {config.get('input_size', 'not found')}")
    print(f"Original features used: {config.get('features', 'not found')}")
    
# Check the actual weight shapes
state_dict = checkpoint['model_state_dict']
print(f"\nActual input weight shape: {state_dict['lstm.weight_ih_l0'].shape}")
print(f"This means the model expects {state_dict['lstm.weight_ih_l0'].shape[1]} input features")
