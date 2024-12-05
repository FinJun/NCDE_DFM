from sklearn.linear_model import LinearRegression
import pandas as pd

class LinearRegressionModel:
    def __init__(self):
        self.models = {}
    
    def train(self, train_X, train_y):
        for ticker in train_y.columns:
            model = LinearRegression()
            model.fit(train_X, train_y[ticker])
            self.models[ticker] = model
    
    def predict(self, test_X):
        predictions = pd.DataFrame(index=range(test_X.shape[0]), columns=self.models.keys())
        for ticker, model in self.models.items():
            predictions[ticker] = model.predict(test_X)
        return predictions
