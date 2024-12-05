import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def scale_data(stock_data):
    scaler = MinMaxScaler()
    scaled_stock_returns = scaler.fit_transform(stock_data)
    return scaled_stock_returns, scaler

def convert_to_tensor(scaled_stock_returns, device):
    scaled_stock_returns = torch.tensor(scaled_stock_returns, dtype=torch.float32).to(device)
    return scaled_stock_returns
