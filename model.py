"""
PyTorch neural network models for financial time series prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialLSTM(nn.Module):
    """
    LSTM model for financial time series prediction
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output predictions
            dropout: Dropout rate for regularization
        """
        super(FinancialLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )   
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        # Apply dropout
        dropped = self.dropout(last_output)
        # Final prediction
        output = self.fc(dropped)
        return output


class FinancialTransformer(nn.Module):
    """
    Transformer model for financial time series prediction
    (More advanced alternative to LSTM)
    """
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(FinancialTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take the last time step and project to output
        output = self.output_projection(x[:, -1, :])
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_model(model_type='lstm', input_size=5, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'lstm' or 'transformer'
        input_size: Number of input features
        **kwargs: Additional model parameters
    """
    if model_type.lower() == 'lstm':
        return FinancialLSTM(
            input_size=input_size,
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            output_size=kwargs.get('output_size', 1),
            dropout=kwargs.get('dropout', 0.2)
        )
    elif model_type.lower() == 'transformer':
        return FinancialTransformer(
            input_size=input_size,
            d_model=kwargs.get('d_model', 64),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 2),
            output_size=kwargs.get('output_size', 1),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size, sequence_length, input_size = 32, 20, 5
    
    # Test LSTM
    print("Testing LSTM model...")
    lstm_model = create_model('lstm', input_size=input_size)
    test_input = torch.randn(batch_size, sequence_length, input_size)
    lstm_output = lstm_model(test_input)
    print(f"LSTM input shape: {test_input.shape}")
    print(f"LSTM output shape: {lstm_output.shape}")
    
    # Test Transformer
    print("\nTesting Transformer model...")
    transformer_model = create_model('transformer', input_size=input_size, d_model=64, nhead=8)
    transformer_output = transformer_model(test_input)
    print(f"Transformer input shape: {test_input.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Count parameters
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"\nLSTM parameters: {lstm_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
