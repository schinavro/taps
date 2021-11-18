import numpy as np

class Mean:
    """
    Zero mean function
    """
    def __init__(self, hyperparameters=0.):
        self.hyperparameters = hyperparameters

    def __call__(self, X):
        D, M = X.shape
        Vave = np.zeros(M) + self.hyperparameters
        Fave = np.zeros((D, M))
        return np.vstack([Vave, Fave]).flatten()

    def V(self, X):
        D, M = X.shape
        return np.zeros(M) + self.hyperparameters

    def dV(self, X):
        return 0.

    def H(self, X):
        return 0.

    def set_hyperparameters(self, hyperparameters=None, data=None):
        if hyperparameters is not None:
            self.hyperparameters = hyperparameters

    def get_hyperparameters(self):
        return self.hyperparameters

class Average(Mean):
    """
    Return Average mean
    """

    def set_hyperparameters(self, hyperparameters=None, data=None):
        self.hyperparameters = np.average(data['potential'])
