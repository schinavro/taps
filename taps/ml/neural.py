import numpy as np


class Mean:
    def __init__(self, type='average', data=None, **kwargs):
        self.type = type
        self._data = data
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, X, data=None, hess=True, type=None):
        """
        Gaussianprocess.org chap 2 pg 28, but yet return ave
        will retrun shape of X_s
        X : (D x M x P)
        return : (1 + M x D) x P
        """
        if type is None:
            type = self.type
        if type == 'zero':
            return 0.
        if data is None:
            data = self._data
        V = data['V']
        D, M = X.shape
        F = np.zeros((D, M))
        # F = np.zeros((D, M, P)) + np.average(data['F'], axis=2)[..., nax]
        if type == 'average':
            e = np.zeros(M) + np.average(V)
        elif type == 'min':
            e = np.zeros(M) + np.min(V)
        elif type == 'manual':
            e = np.zeros(M) + self.Em
        else:
            e = np.zeros(M) + np.max(V)
        if not hess:
            return e
        ef = np.vstack([e, F])
        # ef = np.vstack([e, F.reshape((D * M, P))])
        return ef.flatten()

    def dm(self, X, data=None, hess=False):
        if data is None:
            data = self._data
        return 0
