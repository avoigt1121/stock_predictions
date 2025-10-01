from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import torch
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import FinancialLSTM
from data_loader import prepare_financial_data
import config

app = FastAPI(
    title="Stock Prediction API",
    description="PyTorch-based stock prediction service for n8n integration",
    version="1.0.0"
)

# Input schemas
class TickerRequest(BaseModel):
    ticker: str
    days_ahead: Optional[int] = 1
    period: Optional[str] = "1y"  # yfinance period format

class MultiTickerRequest(BaseModel):
    tickers: List[str]
    days_ahead: Optional[int] = 1
    period: Optional[str] = "1y"

class PredictionResponse(BaseModel):
    ticker: str
    prediction: List[float]
    confidence: Optional[float] = None
    timestamp: str
    model_used: str

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        # Try to load the financial model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'financial_model.pth')
        
        # Load the checkpoint to get the model configuration
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model configuration from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            input_size = model_config['input_size']
            hidden_size = model_config['hidden_size']
            num_layers = model_config['num_layers']
            output_size = model_config['output_size']
            dropout = model_config['dropout']
        else:
            # Fallback to config.py values
            input_size = len(config.FEATURES)
            hidden_size = config.HIDDEN_SIZE
            num_layers = config.NUM_LAYERS
            output_size = config.PREDICTION_DAYS
            dropout = config.DROPOUT
        
        # Initialize model with the correct parameters
        model = FinancialLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
        
        # Extract the actual model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        print(f"Model expects {input_size} input features")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None

def preprocess_data(ticker: str, period: str = "1y"):
    """Preprocess data for prediction"""
    try:
        # Download data
        data = yf.download(ticker, period=period, interval="1d")
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Calculate additional features to match the 5 features the model expects
        # Feature 1: returns
        data['returns'] = data['Close'].pct_change()
        
        # Feature 2: volatility (rolling standard deviation)
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Feature 3: correlation with market (using SPY as proxy)
        spy_data = yf.download("SPY", period=period, interval="1d")
        spy_returns = spy_data['Close'].pct_change()
        data['correlation'] = data['returns'].rolling(window=20).corr(spy_returns)
        
        # Feature 4: RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Feature 5: moving average ratio
        data['ma_ratio'] = data['Close'] / data['Close'].rolling(window=20).mean()
        
        # Fill NaN values
        data = data.ffill().bfill()
        
        # Select the 5 features in the correct order
        feature_columns = ['returns', 'volatility', 'correlation', 'rsi', 'ma_ratio']
        features_data = data[feature_columns].values
        
        # Normalize features (simple standardization)
        features_data = (features_data - np.mean(features_data, axis=0)) / (np.std(features_data, axis=0) + 1e-8)
        
        return features_data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data for {ticker}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Stock Prediction API is running",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Detailed health check for n8n monitoring"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_stock(request: TickerRequest):
    """Predict stock price movement for a single ticker"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        period = request.period or "1y"
        features = preprocess_data(request.ticker, period)
        
        # Ensure we have enough data
        if len(features) < config.SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data for {request.ticker}. Need at least {config.SEQUENCE_LENGTH} days."
            )
        
        # Get the last sequence for prediction
        last_sequence = features[-config.SEQUENCE_LENGTH:]
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.cpu().numpy().flatten()
        
        return PredictionResponse(
            ticker=request.ticker,
            prediction=prediction.tolist(),
            timestamp=datetime.now().isoformat(),
            model_used="FinancialLSTM"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_multiple_stocks(request: MultiTickerRequest):
    """Predict stock price movements for multiple tickers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = []
    
    for ticker in request.tickers:
        try:
            single_request = TickerRequest(
                ticker=ticker,
                days_ahead=request.days_ahead,
                period=request.period
            )
            prediction = predict_stock(single_request)
            results.append(prediction)
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
    
    return {
        "predictions": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }
