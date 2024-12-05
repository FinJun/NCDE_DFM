import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from utils.plotting import plot_predictions
import matplotlib.pyplot as plt

def main():
    predictions_dir = os.path.join('data', 'processed')
    predictions_neural_cde_path = os.path.join(predictions_dir, 'predictions_neural_cde.csv')
    predictions_linear_regression_path = os.path.join(predictions_dir, 'predictions_linear_regression.csv')
    predictions_lstm_path = os.path.join(predictions_dir, 'predictions_lstm.csv')
    actual_path = os.path.join(predictions_dir, 'actual.csv')
    
    if not all([os.path.exists(p) for p in [predictions_neural_cde_path, predictions_linear_regression_path, predictions_lstm_path, actual_path]]):
        raise FileNotFoundError("One or more prediction files or actual.csv not found. Please ensure all models have saved their predictions and actual values.")
    
    predictions_neural_cde = pd.read_csv(predictions_neural_cde_path, index_col='Date', parse_dates=True)
    predictions_linear_regression = pd.read_csv(predictions_linear_regression_path, index_col='Date', parse_dates=True)
    predictions_lstm = pd.read_csv(predictions_lstm_path, index_col='Date', parse_dates=True)
    actual = pd.read_csv(actual_path, index_col='Date', parse_dates=True)
    
    mse_neural_cde = mean_squared_error(actual, predictions_neural_cde)
    mse_linear_regression = mean_squared_error(actual, predictions_linear_regression)
    mse_lstm = mean_squared_error(actual, predictions_lstm)
    
    print(f'Neural Factor CDE MSE: {mse_neural_cde:.6f}')
    print(f'Linear Regression MSE: {mse_linear_regression:.6f}')
    print(f'LSTM MSE: {mse_lstm:.6f}')
    
    performance = pd.DataFrame({
        'Model': ['Neural Factor CDE', 'Linear Regression', 'LSTM'],
        'MSE': [mse_neural_cde, mse_linear_regression, mse_lstm]
    })
    
    print(performance)

    plt.figure(figsize=(10,6))
    plt.bar(performance['Model'], performance['MSE'], color=['blue', 'green', 'orange'])
    plt.title('Model Performance Comparison (MSE)')
    plt.ylabel('Mean Squared Error')
    plt.show()

    ticker = 'AAPL'
    plt.figure(figsize=(6,6))
    plt.scatter(actual[ticker], predictions_neural_cde[ticker], color='red', label='Neural Factor CDE Predicted', alpha=0.5)
    plt.scatter(actual[ticker], predictions_linear_regression[ticker], color='green', label='Linear Regression Predicted', alpha=0.5)
    plt.scatter(actual[ticker], predictions_lstm[ticker], color='purple', label='LSTM Predicted', alpha=0.5)
    plt.plot([actual[ticker].min(), actual[ticker].max()], [actual[ticker].min(), actual[ticker].max()], 'k--', label='Ideal')  # 대각선
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Model Comparison - {ticker} Actual vs Predicted Next Day Return')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
