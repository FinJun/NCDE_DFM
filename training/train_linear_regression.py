import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from models.linear_regression import LinearRegressionModel
from utils.plotting import plot_predictions

def main():
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
    
    lr_model = LinearRegressionModel()
    train_X_df = pd.DataFrame(train_X, columns=log_returns.columns)
    train_y_df = pd.DataFrame(train_y, columns=log_returns.columns)
    lr_model.train(train_X_df, train_y_df)
    print("Linear Regression models trained.")
    
    test_X_df = pd.DataFrame(test_X, columns=log_returns.columns)
    predictions = lr_model.predict(test_X_df)

    predictions.to_csv(os.path.join('data', 'processed', 'predictions_linear_regression.csv'))
    print(f"Linear Regression predictions saved to {os.path.join('data', 'processed', 'predictions_linear_regression.csv')}")
    
    mse_lr = mean_squared_error(test_y, predictions)
    print(f'Linear Regression Model MSE: {mse_lr:.6f}')
    
    plot_predictions(test_y[:,0], predictions.iloc[-1,0], log_returns.columns[0], 'Linear Regression', 'green')
    
if __name__ == '__main__':
    main()
