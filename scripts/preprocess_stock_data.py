import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from utils.data_processing import calculate_log_returns, scale_data

def main():
    raw_data_path = os.path.join('data', 'raw', 'stock_data.csv')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw stock data not found at {raw_data_path}. Please run download_stock_data.py first.")
    
    stock_data = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)
    print(f"Loaded stock data with shape: {stock_data.shape}")
    
    log_returns = calculate_log_returns(stock_data)
    print(f"Calculated log returns with shape: {log_returns.shape}")
    
    log_returns = log_returns.dropna()
    
    scaled_log_returns, scaler_stocks = scale_data(log_returns)
    print(f"Scaled log returns with shape: {scaled_log_returns.shape}")
    
    scaled_log_returns_df = pd.DataFrame(scaled_log_returns, index=log_returns.index, columns=log_returns.columns)
    
    processed_data_dir = os.path.join('data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    
    scaled_log_returns_df.to_csv(os.path.join(processed_data_dir, 'log_returns.csv'))
    print(f'\nPreprocessed log returns saved to {os.path.join(processed_data_dir, "log_returns.csv")}')
    
    import joblib
    joblib.dump(scaler_stocks, os.path.join(processed_data_dir, 'scaler_stocks.pkl'))
    print(f"Scaler saved to {os.path.join(processed_data_dir, 'scaler_stocks.pkl')}")
    
if __name__ == '__main__':
    main()
