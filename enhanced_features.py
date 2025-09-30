"""
Improved feature engineering for better model performance
"""

import numpy as np
import pandas as pd
from api import api_call


def calculate_technical_indicators(df, price_col='AAPL'):
    """
    Calculate comprehensive technical indicators
    """
    features = pd.DataFrame(index=df.index)
    prices = df[price_col]
    
    # Returns (multiple timeframes)
    features['returns_1d'] = prices.pct_change(1)
    features['returns_3d'] = prices.pct_change(3)
    features['returns_5d'] = prices.pct_change(5)
    features['returns_10d'] = prices.pct_change(10)
    
    # Volatility (multiple windows)
    features['volatility_5d'] = features['returns_1d'].rolling(5).std()
    features['volatility_10d'] = features['returns_1d'].rolling(10).std()
    features['volatility_20d'] = features['returns_1d'].rolling(20).std()
    
    # Moving averages and ratios
    features['sma_5'] = prices.rolling(5).mean()
    features['sma_10'] = prices.rolling(10).mean()
    features['sma_20'] = prices.rolling(20).mean()
    
    # Price relative to moving averages
    features['price_to_sma5'] = prices / features['sma_5'] - 1
    features['price_to_sma10'] = prices / features['sma_10'] - 1
    features['price_to_sma20'] = prices / features['sma_20'] - 1
    
    # RSI (Relative Strength Index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma_20 = prices.rolling(20).mean()
    std_20 = prices.rolling(20).std()
    features['bb_upper'] = sma_20 + (2 * std_20)
    features['bb_lower'] = sma_20 - (2 * std_20)
    features['bb_position'] = (prices - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # MACD
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Volume-based features (if volume data available)
    # Note: yfinance provides volume, but we're not using it yet
    
    return features


def prepare_enhanced_financial_data(tickers=["AAPL", "MSFT", "TSLA"], start="2025-01-01", end="2025-09-30"):
    """
    Enhanced feature preparation with more predictive indicators
    """
    # Get raw data
    df = api_call(tickers=tickers, start=start, end=end)
    
    # Calculate enhanced features for primary ticker
    features_df = calculate_technical_indicators(df, 'AAPL')
    
    # Cross-stock features
    returns = df.pct_change()
    
    # Correlation features
    features_df['corr_msft_5d'] = returns['AAPL'].rolling(5).corr(returns['MSFT'])
    features_df['corr_msft_10d'] = returns['AAPL'].rolling(10).corr(returns['MSFT'])
    features_df['corr_tsla_5d'] = returns['AAPL'].rolling(5).corr(returns['TSLA'])
    
    # Market breadth (how AAPL performs vs other stocks)
    features_df['outperformance_msft'] = returns['AAPL'] - returns['MSFT']
    features_df['outperformance_tsla'] = returns['AAPL'] - returns['TSLA']
    
    # Momentum indicators
    features_df['momentum_3d'] = features_df['returns_1d'].rolling(3).sum()
    features_df['momentum_5d'] = features_df['returns_1d'].rolling(5).sum()
    
    # Volatility regime (high/low volatility periods)
    vol_median = features_df['volatility_20d'].median()
    features_df['vol_regime'] = (features_df['volatility_20d'] > vol_median).astype(int)
    
    # Lag features (yesterday's values)
    features_df['returns_1d_lag1'] = features_df['returns_1d'].shift(1)
    features_df['volatility_20d_lag1'] = features_df['volatility_20d'].shift(1)
    features_df['rsi_lag1'] = features_df['rsi'].shift(1)
    
    # Drop unnecessary columns (absolute prices, moving averages)
    cols_to_drop = ['sma_5', 'sma_10', 'sma_20', 'bb_upper', 'bb_lower']
    features_df = features_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Clean data
    features_df = features_df.dropna()
    
    # Select most important features (you can experiment with this)
    important_features = [
        'returns_1d',  # Target (should be first)
        'returns_3d', 'returns_5d', 'returns_10d',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'price_to_sma5', 'price_to_sma10', 'price_to_sma20',
        'rsi', 'bb_position',
        'macd', 'macd_signal', 'macd_histogram',
        'corr_msft_5d', 'corr_msft_10d',
        'outperformance_msft', 'outperformance_tsla',
        'momentum_3d', 'momentum_5d',
        'vol_regime',
        'returns_1d_lag1', 'volatility_20d_lag1', 'rsi_lag1'
    ]
    
    # Only keep features that exist
    available_features = [f for f in important_features if f in features_df.columns]
    features_df = features_df[available_features]
    
    print(f"Enhanced features: {features_df.columns.tolist()}")
    print(f"Feature correlation with target:")
    correlations = features_df.corr()['returns_1d'].abs().sort_values(ascending=False)
    print(correlations.head(10))
    
    return features_df


if __name__ == "__main__":
    # Test enhanced features
    enhanced_df = prepare_enhanced_financial_data()
    print(f"Enhanced data shape: {enhanced_df.shape}")
    print(f"Data range: {enhanced_df.index[0]} to {enhanced_df.index[-1]}")
    print(f"First few rows:")
    print(enhanced_df.head())
