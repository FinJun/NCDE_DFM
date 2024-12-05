import torch
from torch import nn
from models.neural_cde import NeuralFactorCDE
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
  
    scaled_train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, train_size, input_dim)
    scaled_train_y = torch.tensor(train_y, dtype=torch.float32).to(device)  # (train_size, output_dim)
    
    scaled_test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(0).to(device)
    scaled_test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
    
    hidden_dim = 32
    input_dim = scaled_train_X.shape[2]
    factor_dim = 10
    output_dim = scaled_train_y.shape[1]

    model = NeuralFactorCDE(hidden_dim, input_dim, factor_dim, output_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    time_steps = train_size
    time_tensor = torch.linspace(0., 10., time_steps).to(device)
    time_tensor_test = torch.linspace(10.0001, 10., len(test_X)).to(device)
  
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output, factors = model(scaled_train_X, time_tensor)
        loss = criterion(output, scaled_train_y[-1].unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')
    
    model.eval()
    with torch.no_grad():
        test_pred, _ = model(scaled_test_X, time_tensor_test)
        test_loss = criterion(test_pred, scaled_test_y[-1].unsqueeze(0))
        print(f'\nNeural Factor CDE Test Loss (MSE): {test_loss.item():.6f}')
    
    predictions_neural_cde = pd.DataFrame(test_pred.cpu().numpy(), columns=log_returns.columns, index=log_returns.index[train_size:])
    predictions_neural_cde.to_csv(os.path.join('data', 'processed', 'predictions_neural_cde.csv'))
    print(f"Neural Factor CDE predictions saved to {os.path.join('data', 'processed', 'predictions_neural_cde.csv')}")
    
    actual_path = os.path.join('data', 'processed', 'actual.csv')
    if not os.path.exists(actual_path):
        actual = pd.DataFrame(test_y, columns=log_returns.columns, index=log_returns.index[train_size:])
        actual.to_csv(actual_path)
        print(f"Actual values saved to {actual_path}")
    
    from utils.plotting import plot_predictions
    plot_predictions(test_y[:,0], test_pred.cpu().numpy()[0,0], log_returns.columns[0], 'Neural Factor CDE', 'red')
    
if __name__ == '__main__':
    main()
