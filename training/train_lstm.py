# training/train_lstm.py

import torch
from torch import nn
from models.lstm import LSTMModel
from utils.data_processing import calculate_log_returns, scale_data, convert_to_tensor
from utils.plotting import plot_predictions
import pandas as pd
import os
import joblib

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    processed_data_path = os.path.join('data', 'processed', 'log_returns.csv')
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Please run preprocess_stock_data.py first.")
    
    log_returns = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
    print(f"Loaded log returns with shape: {log_returns.shape}")
    
    train_size = int(0.8 * len(log_returns))
    train_X = log_returns.iloc[:train_size].values[:, :]
    train_y = log_returns.iloc[:train_size].values[:, :]
    test_X = log_returns.iloc[train_size:].values[:, :]
    test_y = log_returns.iloc[train_size:].values[:, :]
    
    input_dim = train_X.shape[1]
    hidden_dim = 64
    num_layers = 2
    output_dim = train_y.shape[1]
    
    lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, device).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
  
    scaled_train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, train_size, input_dim)
    scaled_train_y = torch.tensor(train_y, dtype=torch.float32).to(device)  # (train_size, output_dim)
    
    scaled_test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(0).to(device)
    scaled_test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
    
    num_epochs = 200
    for epoch in range(num_epochs):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(scaled_train_X)
        loss = criterion(output, scaled_train_y[-1].unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')
    
    lstm_model.eval()
    with torch.no_grad():
        test_pred = lstm_model(scaled_test_X)
        test_loss = criterion(test_pred, scaled_test_y[-1].unsqueeze(0))
        print(f'\nLSTM Test Loss (MSE): {test_loss.item():.6f}')
    
    predictions_lstm = pd.DataFrame(test_pred.cpu().numpy(), columns=log_returns.columns, index=log_returns.index[train_size:])
    predictions_lstm.to_csv(os.path.join('data', 'processed', 'predictions_lstm.csv'))
    print(f"LSTM predictions saved to {os.path.join('data', 'processed', 'predictions_lstm.csv')}")
    
    from utils.plotting import plot_predictions
    plot_predictions(test_y[:,0], test_pred.cpu().numpy()[0,0], log_returns.columns[0], 'LSTM', 'purple')
    
if __name__ == '__main__':
    main()
