import numpy as np

eps = np.finfo(float).eps


def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))


def loss(y_true, y_pred):
    return - np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))


def dloss_dw(y_true, y_pred, X):
    return - (X.T @ (y_true - y_pred)) / X.shape[0]


class MyLogisticRegression:

    def __init__(self):
        self._w = None

    def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
        n_samples, n_features = X.shape

        self.w = np.random.randn(n_features) * 0.001

        for e in range(n_epochs):
            p = sigmoid(X @ self.w.T)
            L = loss(Y, p)
            self.w = self.w - learning_rate * dloss_dw(Y, p, X)
        return

    def predict(self, X):
        return np.round(sigmoid(X @ self.w.T))
