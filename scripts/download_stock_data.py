import yfinance as yf
import pandas as pd
import os
import time

def download_stock_data(tickers, start_date, end_date, interval='1mo', max_retries=3, sleep_time=5):
    data = pd.DataFrame()
    
    for ticker in tickers:
        success = False
        attempts = 0
        while not success and attempts < max_retries:
            try:
                etf_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)['Adj Close']
                data[ticker] = etf_data
                success = True
                print(f'Successfully downloaded {ticker}')
            except Exception as e:
                attempts += 1
                print(f'Failed to download {ticker} (Attempt {attempts}/{max_retries}). Error: {e}')
                time.sleep(sleep_time)
        if not success:
            print(f'Failed to download {ticker} after {max_retries} attempts. Excluding from dataset.')
    
    return data

def main():
    stock_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'DIS', 'NVDA', 'MA', 'PYPL', 'BAC', 'VZ', 'ADBE',
        'CMCSA', 'NFLX', 'KO', 'INTC', 'T', 'PFE', 'XOM', 'PEP', 'CSCO', 'ABT',
        'CRM', 'CVX', 'NKE', 'MRK', 'WMT', 'LLY', 'TMO', 'COST', 'ORCL', 'MCD',
        'MDT', 'DHR', 'WFC', 'BMY', 'NEE', 'TXN', 'QCOM', 'AMGN', 'UNP', 'LOW'
    ]
    
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    interval = '1mo'
    
    stock_data = download_stock_data(stock_tickers, start_date, end_date, interval)
    
    raw_data_dir = os.path.join('data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    stock_data.to_csv(os.path.join(raw_data_dir, 'stock_data.csv'))
    print(f'\nAll data downloaded and saved to {os.path.join(raw_data_dir, "stock_data.csv")}')
    
if __name__ == '__main__':
    main()
