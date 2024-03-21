import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union

"""
    Class that creates a simple linear regression model
"""
class myLinearRegression():
   
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.weight = 1
        self.bias = 1
        self.cost_history = []
        self.iterations = 0

    """
        Split dataset in 4 pieces: training features, testing features, training
        targets and testing targets
    """
    def _train_test_split(
                self,
                X: pd.core.frame.DataFrame,
                y: pd.core.frame.DataFrame,
                test_size=0.2
            ) -> Union[ 
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame
                    ]:
        N = len(X)
        N_train = int(N * (1 - test_size))
        X_train = X.iloc[:N_train]
        X_test = X.iloc[N_train:]
        y_train = y.iloc[:N_train]
        y_test = y.iloc[N_train:]
        return X_train, X_test, y_train, y_test

    """
        Fit data into the model
    """
    def fit(self,
                X: pd.core.frame.DataFrame, 
                y: pd.core.frame.DataFrame
            ) -> Union[ 
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame
                    ]:
        self.weight = 1
        self.bias = 1
        # X is a vector, y is a vector
        # split data in 80% training data and 20% in testing data
        return self._train_test_split(X, y)

    """
        Prediction function
    """
    def prediction(self, 
                    X: pd.core.frame.DataFrame
                ) -> pd.core.frame.DataFrame:
        return self.weight * X + self.bias

    """
        Function that computes the cost of our regression at a given iteration
    """
    def cost(self, 
                    X: pd.core.frame.DataFrame,
                    y: pd.core.frame.DataFrame
                ) -> float:
        N = len(X)
        total_error = 0.0
        for i in range(N):
            total_error += (y.iloc[i,0] - (self.weight * X.iloc[i,0] + self.bias)) ** 2
        return total_error / N

    """
        Gradient computation
    """
    def gradient(self, 
                    X: pd.core.frame.DataFrame,
                    y: pd.core.frame.DataFrame,
                    learning_rate: float
                ):
        weight_deriv = 0
        bias_deriv = 0
        N = len(X)
        weight_deriv = -2 * X * pd.DataFrame(y.values - (self.weight * X + self.bias).values, columns=X.columns)
        bias_deriv = -2 * pd.DataFrame(y.values - (self.weight * X + self.bias).values, columns=X.columns)

        weight_deriv = float(weight_deriv.sum().values[0])
        bias_deriv = float(bias_deriv.sum().values[0])
        
        self.weight -= (weight_deriv / N) * learning_rate
        self.bias -= (bias_deriv / N) * learning_rate

    """
        Train the model applying gradient at each iteration
    """
    def train(self,
                    X: pd.core.frame.DataFrame,
                    y: pd.core.frame.DataFrame,
                    learning_rate: float,
                    iterations: int
                ):
        for _ in range(iterations):
            self.iterations += 1
            self.gradient(X, y, learning_rate)
            cost = self.cost(X, y)
            self.cost_history.append(cost)

"""
    Class that creates a multiple linear regression model
"""
class myMultipleLinearRegression():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.weights = None
        self.cost_history = []
        self.iterations = 0

    """
        Sometimes the model can run on buffer overflow atfer a long time (after
        multiple iterations, to be more precise); normalizing the data is necessary
        to avoid this possible behavior.
    """
    def normalize(self,
                    X: pd.core.frame.DataFrame
                ) -> pd.core.frame.DataFrame:
        aux = X.values
        aux = aux.T
        for column in aux:
            fmean = np.mean(column)
            frange = np.amax(column) - np.amin(column)
            if frange != 0:
                column -= fmean
                column /= frange
        X = pd.DataFrame(aux.T, columns=X.columns).fillna(1)
        return X

    """
        Split dataset in 4 pieces: training features, testing features, training
        targets and testing targets
    """
    def _train_test_split(
                self,
                X: pd.core.frame.DataFrame,
                y: pd.core.frame.DataFrame,
                test_size: float
            ) -> Union[ 
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame
                    ]:
        N = len(X)
        N_train = int(N * (1 - test_size))
        X_train = X.iloc[:N_train]
        X_test = X.iloc[N_train:]
        y_train = y.iloc[:N_train]
        y_test = y.iloc[N_train:]
        return X_train, X_test, y_train, y_test
    
    """
        Fit data into the model
    """
    def fit(self,
                X: pd.core.frame.DataFrame, 
                y: pd.core.frame.DataFrame,
                split_tests: float = 0.2
            ) -> Union[ 
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame,
                        pd.core.frame.DataFrame
                    ]:
        N = len(X.columns)
        self.weights = np.zeros(N)
        return self._train_test_split(X, y, split_tests)

    """
        Prediction function
    """
    def prediction(self,
                    X: pd.core.frame.DataFrame
                ) -> np.ndarray:

        return np.dot(X, self.weights)

    """
        Function that computes the cost of our regression at a given iteration
    """
    def cost(self,
            X: pd.core.frame.DataFrame,
            y: pd.core.frame.DataFrame
            ) -> float:
        N = len(y)
        X_arr = X.values
        y_arr = y.values
        predictions = self.prediction(X_arr)
        err = (predictions - y_arr.T) ** 2
        return 1.0 / (2 * N) * err.sum(axis=1)

    """
        Gradient computation
    """
    def gradient(self, 
                    X: pd.core.frame.DataFrame,
                    y: pd.core.frame.DataFrame,
                    learning_rate: float
                ):
        N = len(X)
        X_arr = X.values
        y_arr = y.values
        predictions = self.prediction(X_arr)
        err = y_arr.T - predictions
        gradient = np.dot(-X_arr.T, err.T)
        gradient /= N
        gradient *= learning_rate
        gradient = gradient.reshape((-1,))
        self.weights -= gradient

    """
        Train the model applying gradient at each iteration
    """
    def train(self,
                    X: pd.core.frame.DataFrame,
                    y: pd.core.frame.DataFrame,
                    learning_rate: float,
                    iterations: int
                ):
        for _ in range(iterations):
            self.iterations += 1
            self.gradient(X, y, learning_rate)
            cost = self.cost(X, y)
            self.cost_history.append(cost)

# for development and debugging only; the "production" will be present in the notebook
def main():
    train_file = "../data/train.csv"
    train_df = pd.read_csv(train_file).set_index('id')
    X = train_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
    X["bias"] = np.ones(shape=(len(X), 1))

    y = train_df[['MedHouseVal']]
    lin_reg = myMultipleLinearRegression()
    X = lin_reg.normalize(X)
    X_train, X_test, y_train, y_test = lin_reg.fit(X, y)
    
    lin_reg.train(X_train, y_train, learning_rate=0.05, iterations=5000)
    
    y_pred_test = lin_reg.prediction(X_test)
    y_pred_train = lin_reg.prediction(X_train)

    y_test = y_test.values
    y_test = y_test.T[0]
    print("pred: ", y_pred_test)
    print("test: ", y_test)


    plt.plot(y_pred_test[0:20],'blue', label="Predict")
    plt.plot(y_test[0:20],'red', label="Actual Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
