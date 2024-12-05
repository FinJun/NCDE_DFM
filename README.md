# Stock Prediction using Neural CDE, Linear Regression, and LSTM

This project aims to predict stock returns using three models:
1. **Neural Controlled Differential Equations (Neural CDE)**
2. **Linear Regression**
3. **Long Short-Term Memory (LSTM)**

The dataset consists of historical stock price data. The project includes data preprocessing, model training, and performance evaluation. The code is modular, enabling flexibility and easy adaptation.

---

## Project Structure

```
stock_prediction/
├── data/
│   ├── raw/                # Raw data files (e.g., downloaded stock data)
│   ├── processed/          # Processed data files (e.g., log returns, scaled data)
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── neural_cde.py       # Neural CDE model definition
│   ├── linear_regression.py # Linear Regression model definition
│   ├── lstm.py             # LSTM model definition
├── training/               # Model training scripts
│   ├── __init__.py
│   ├── train_neural_cde.py  # Train Neural CDE
│   ├── train_linear_regression.py # Train Linear Regression
│   ├── train_lstm.py        # Train LSTM
├── evaluation/             # Model evaluation scripts
│   ├── __init__.py
│   ├── evaluate_models.py   # Evaluate and compare model performance
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_processing.py   # Data preprocessing utilities
│   ├── plotting.py          # Plotting and visualization utilities
├── scripts/                # Data download and preprocessing scripts
│   ├── download_stock_data.py # Download stock data
│   ├── preprocess_stock_data.py # Preprocess stock data
├── requirements.txt        # Python dependencies
├── main.py                 # Main script to run the full pipeline
└── README.md               # Project documentation
```

---

## Features

- **Data Collection:** Downloads historical stock price data using `yfinance`.
- **Data Preprocessing:** Calculates log returns and scales the data for modeling.
- **Neural CDE Model:** Implements a dynamic factor model using Neural CDE.
- **Linear Regression Model:** Baseline model for stock prediction.
- **LSTM Model:** Captures temporal dependencies for stock price predictions.
- **Evaluation and Comparison:** Compares the performance of all models using MSE and visualizations.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/stock_prediction.git
cd stock_prediction
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Unix-based systems
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Download Stock Data
Run the script to download stock price data:
```bash
python scripts/download_stock_data.py
```

### 2. Preprocess Data
Process the downloaded stock data to compute log returns:
```bash
python scripts/preprocess_stock_data.py
```

### 3. Train Models
Train each of the models:

#### Neural CDE
```bash
python training/train_neural_cde.py
```

#### Linear Regression
```bash
python training/train_linear_regression.py
```

#### LSTM
```bash
python training/train_lstm.py
```

### 4. Evaluate Models
Evaluate and compare the performance of all models:
```bash
python evaluation/evaluate_models.py
```

### 5. Full Pipeline
Run the full pipeline from data download to evaluation:
```bash
python main.py
```
