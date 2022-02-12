import numpy as np
from scipy.optimize import minimize
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import inv, cholesky
from numpy import log, sum, diagonal


class Regression:
    """
    Return a function that should be minimized
    log_likelihood with gradient data involves.
    """

    def __init__(self, kernel_regression_kwargs=None,
                 mean_regression_kwargs=None, optimized=False):
        # self.kernel_regression=None or KernelRegression()
        self.kernel_regression_kwargs = kernel_regression_kwargs or \
                  {"method": "BFGS"}
        # self.mean_regression=None or MeanRegression()
        self.mean_regression_kwargs = mean_regression_kwargs or {}

        self.optimized = optimized

    def __call__(self, *args, kernel=None, mean=None, **kwargs):
        if mean is not None:
            self.mean_regression(mean, kernel, *args, **kwargs)
        if kernel is not None:
            self.kernel_regression(kernel, mean, *args, **kwargs)

    def kernel_regression(self, kernel, mean, *args, data=None, **kwargs):
        """
        k : class Kernel; k(X, X) ((DxN + 1) x m) x ((DxN + 1) x n) array
        X : imgdb['X']; position of atoms, (D x N) x m dimension array
        Y : imgdb['Y']; energy and forces of atoms One dimensional array
            with length (m + m x (D x N)) Potential comes first.
        m : mean function
        M : a number of data
        """
        x0 = kernel.get_hyperparameters()
        likelihood = self.likelihood(kernel, mean, data)
        res = minimize(likelihood, x0=x0, **self.kernel_regression_kwargs)
        kernel.set_hyperparameters(res.x)

    def mean_regression(self, mean, kernel, *args, data=None, **kwargs):
        mean.set_hyperparameters(data=data)

    def likelihood(self, kernel=None, mean=None, data=None):
        k, m = kernel, mean
        X = data['kernel']['X']
        Y = data['kernel']['Y']
        Y_m = Y - m(X)

        def likelihood(hyperparameters):
            k.set_hyperparameters(hyperparameters)
            K = k(X, X, noise=True)
            detK = np.linalg.det(K)
            try:
                detK = diagonal(cholesky(K))
                log_detK = sum(log(detK))
            except LinAlgError:
                # Postive definite matrix
                detK = np.linalg.det(K)
                # print(detK)
                if detK <= 1e-5:
                    log_detK = -5
                else:
                    log_detK = log(detK)
            return log_detK + 0.5 * (Y_m.T @ (inv(K) @ Y_m))
        return likelihood

    def reg_kwargs(self, regression_method=None, hyperparameters=None,
                   hyperparameters_bounds=None):
        if regression_method is None:
            regression_method = self.regression_method
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        no_boundary = False
        if hyperparameters_bounds is None:
            hyperparameters_bounds = self.hyperparameters_bounds
            if hyperparameters_bounds == {}:
                no_boundary = True
        method = regression_method
        number_of_hyperparameters = len(self.kernel.key2idx)
        x0 = np.zeros(number_of_hyperparameters)
        bounds = np.zeros((number_of_hyperparameters, 2))
        for key, idx in self.kernel.key2idx.items():
            x0[idx] = hyperparameters[key]
            if not no_boundary:
                bounds[idx] = hyperparameters_bounds[key]
        if no_boundary:
            bounds = None
        return {'x0': x0, 'bounds': bounds, 'method': method}


class NonGradientRegression(Regression):
    def likelihood(hyperparameters_list):
        k.set_hyperparameters(hyperparameters_list)
        K = k(X, X, orig=True)
        try:
            detK = diagonal(cholesky(K))
            log_detK = sum(log(detK))
        except LinAlgError:
            # Postive definite matrix
            detK = np.linalg.det(K)
            if detK <= 1e-5:
                log_detK = -5
            else:
                log_detK = log(detK)
        return log_detK + 0.5 * (Y_m.T @ (inv(K) @ Y_m))


class PseudoGradientRegression(Regression):
    def calculate(self):
        data = self.get_data(paths)
        D, M, P = data['X'].shape
        _X = data['X']
        _V = data['V']
        _F = data['F']
        X = np.zeros((D, M, (D * M + 1) * P))
        Y = np.zeros((D * M + 1) * P)
        X[:, :, :P] = _X
        Y[:P] = _V
        for i in range(1, D * M):
            dX = np.zeros((D, M, 1))
            d, _m = i // M, i % M
            dX[d, _m] = dx
            X[:, :, P * i: P * (i + 1)] = _X + dX
            Y[P * i: P * (i + 1)] = _V + dx * _F[d, _m]
        Y_m = Y - m(X, data, hess=False)

    def pseudo_gradient_likelihood(hyperparameters_list):
        k.set_hyperparameters(hyperparameters_list)
        K = k(X, X, noise=True, orig=True)
        detK = diagonal(cholesky(K))
        return sum(log(detK)) + 0.5 * (Y_m.T @ (inv(K) @ Y_m))
