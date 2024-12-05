import matplotlib.pyplot as plt

def plot_scaled_data(combined_data, scaled_stock_returns, selected_stocks):
    plt.figure(figsize=(20, 10))
    for i, ticker in enumerate(selected_stocks):
        plt.plot(combined_data.index, scaled_stock_returns.cpu().numpy()[:, i], label=ticker)
    plt.title('Scaled Stock Log Returns (Monthly)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Value')
    plt.legend(fontsize=8, ncol=5)
    plt.tight_layout()
    plt.show()

def plot_predictions(actual, predicted, ticker, model_name, color):
    plt.figure(figsize=(6,6))
    plt.scatter(actual, predicted, color=color, label=f'{model_name} Predicted', alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', label='Ideal')  # 대각선
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - {ticker} Actual vs Predicted Next Day Return')
    plt.legend()
    plt.show()
