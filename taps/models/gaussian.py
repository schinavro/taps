import time
import numpy as np
from numpy import newaxis as nax

from numpy.linalg import inv, cholesky

from taps.models import Model
from taps.ml.kernels import Kernel
from taps.ml.means import Mean
from taps.ml.regressions.regressions import GaussianProcessRegressor


class Gaussian(Model):
    """
    Gaussian Potential Energy Surface model

    Using the given data, estimate the potential. Additionally, it can estimate
    the covariance

    Parameters
    ----------

    'real_model': Model class
        Actuall model that Gaussian PES supposed to be approximate
    'kernel': Kernel class
        kernel function for the Gaussian process
    'mean': Model class
        User define Mean function used in GP

    """

    implemented_properties = {'covariance', 'potential', 'gradients',
                              'hessian'}

    def __init__(self, real_model=None, kernel=None, mean=None,
                 regression=None, **kwargs):
        """
        data array
        """
        self.real_model = real_model or self
        self.kernel = kernel or Kernel()
        self.mean = mean or Mean()
        self.regression = regression or GaussianProcessRegressor()

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=None, **kwargs):
        """
        P : int | length of image to consider / or number of image to calculate
        D : int | Dimension
        N : int | Number of Atom in one image
        Xm :
           Data coords
        Xn :
           points want to calc
        Y :
        return : Dim x N x P - 2 array
        """
        data = paths.get_image_data(prj=self.prj)

        orig_shape = coords.shape[:-1]
        D, M, N = np.prod(coords.D), len(data['potential']), coords.N

        Xm = data['kernel']['X']
        Xn = coords.coords.reshape(D, N)
        Y = data['kernel']['Y']

        k, m = self.kernel, self.mean

        # Re calculate the hyperparameters if data has changed
        if data['data_ids'] != self._cache.get('data_ids'):
            self.regression(mean=m, kernel=k, data=data)
            self._cache['K_y_inv'] = inv(k(Xm, Xm, noise=True))
            self._cache['data_ids'] = data['data_ids'].copy()
        K_y_inv = self._cache['K_y_inv']

        if 'potential_and_gradient' in properties:
            N = len(Xn[..., :])
            K_s = k(Xm, Xn)  # (D+1)N x (D+1)M x P
            mu = m(Xn) + K_s.T @ K_y_inv @ (Y - m(Xn))
            E = mu[: N]
            F = -mu[N:].reshape(*orig_shape, N)
            self.results['potential_and_forces'] = E, F

        if 'potential' in properties:
            K_s = k(Xm, Xn, potential_only=True)  # (D+1)N x M
            potential = m.V(Xn) + K_s.T @ K_y_inv @ (Y - m(Xm))
            self.results['potential'] = potential
        if 'gradients' in properties:
            dK_s = k(Xm, Xn, gradient_only=True)  # (D+1)N x (D+1)M x P
            mu_f = m.dV(Xn) + dK_s.T @ K_y_inv @ (Y - m(Xm))
            self.results['gradients'] = mu_f.reshape(*orig_shape, N)
        if 'hessian' in properties:
            K_s = k(Xm, Xn, hessian_only=True)            # (D+1)N x DDM
            H = m.H(Xn) + K_s.T @ K_y_inv @ (Y - m(Xm))  # DDM
            self.results['hessian'] = H.reshape(D, D, N)

        if 'covariance' in properties:
            K = k(Xn, Xn, orig=True)
            K_s = k(Xm, Xn)
            K_s_T = k(Xn, Xm)
            self.results['covariance'] = K - (K_s_T @ K_y_inv @ K_s)[:N, :N]

    def get_covariance(self, paths, **kwargs):
        cov_coords = self.get_properties(paths, properties='covariance',
                                         **kwargs)
        _ = np.diag(cov_coords)
        cov_coords = _.copy()
        cov_coords[_ < 0] = 0
        return 1.96 * np.sqrt(cov_coords) / 2

    def get_hyperparameters(self, hyperparameters_list=None):
        return self.kernel.get_hyperparameters(hyperparameters_list)

    def get_state_info(self):
        info = []
        for k, v in self.kernel.hyperparameters.items():
            info.append(f"{k: <11}" + ": " + str(v))
        return '\n'.join(info)
