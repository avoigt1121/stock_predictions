import yfinance as yf
import pandas as pd


def api_call(tickers = ["AAPL", "MSFT", "TSLA"], start ="2025-01-01", end="2025-09-30"):

# Download historical data
    data = yf.download(tickers, start, end, group_by='ticker')

# Initialize an empty DataFrame to store adjusted close prices
    adj_close_df = pd.DataFrame()

    # Extract adjusted close for each ticker
    for ticker in tickers:
        if 'Adj Close' in data[ticker].columns:
            adj_close_df[ticker] = data[ticker]['Adj Close']
        else:
            # fallback if 'Adj Close' is missing
            adj_close_df[ticker] = data[ticker]['Close']
    return(adj_close_df)
'''
    # Display first few rows
    #print(adj_close_df.head())

# Example: calculate daily returns
    returns = adj_close_df.pct_change()
    #print(returns.head())

    # Example: 5-day rolling mean
    rolling_mean = adj_close_df.rolling(5).mean()
    print(rolling_mean.head())
'''