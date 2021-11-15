import time
import numpy as np
from numpy import newaxis as nax

from numpy.linalg import inv, cholesky

from taps.models import Model
from taps.ml.kernels import Kernel
from taps.ml.means import Mean
from taps.ml.regressions import Regression

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
    'mean': Mean class
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
        self.regression = regression or Regression()

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=None, **kwargs):
        """
        P : int | length of image to consider / or number of image to calculate
        D : int | Dimension
        N : int | Number of Atom in one image
        X :
        Y :
        return : Dim x N x P - 2 array
        """
        if properties == ['mass']:
            return
        shape_org = coords.shape

        k, m = self.kernel, self.mean
        X = coords.copy()
        data = paths.get_image_data(prj=self.prj)
        Xm = data['kernel']['X']
        Xn = coords.flat()
        Y = data['kernel']['Y']
        # If no data given
        if len(Y) == 0:
            return np.zeros(X.shape[-1])
        D, N = Xn.shape
        D, M = Xm.shape

        # Re calculate the hyperparameters if data has changed
        if not self.regression.optimized:
            self.regression(paths, coords, mean=m, kernel=k, data=data)
            self._cache['K_y_inv'] = inv(k(Xm, Xm, noise=True))
            self.regression.optimized = True
        K_y_inv = self._cache['K_y_inv']

        if 'potential_and_gradient' in properties:
            N = len(Xn[..., :])
            K_s = k(Xm, Xn)  # (D+1)N x (D+1)M x P
            mu = m(Xn) + K_s.T @ K_y_inv @ (Y - m(Xn))
            E = mu[: N]
            F = -mu[N:].reshape(D, N)
            self.results['potential_and_forces'] = E, F

        if 'potential' in properties:
            K_s = k(Xm, Xn, potential_only=True)  # (D+1)N x M
            potential = m.V(Xn) + K_s.T @ K_y_inv @ (Y - m(Xm))
            # Y = data['V']  # @@@
            # N = len(Y)  # @@@
            # K_s = K_s[:N] # @@@
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # return m(Xm, hess=False) + \
            #      K_s.T @ K_y_inv @ (Y - m(Xn, hess=False)) # @@@
            self.results['potential'] = potential
        if 'gradients' in properties:
            dK_s = k(Xm, Xn, gradient_only=True)  # (D+1)N x (D+1)M x P
            mu_f = m.dV(Xn) + dK_s.T @ K_y_inv @ (Y - m(Xm))
            # N = len(Y) # @@@
            # dK_s = dK_s[:N] # @@@
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # mu_f = m.dm(Xm) + dK_s.T @ K_y_inv @ (Y - m(Xn, hess=False))
            # return -mu_f.reshape(Xm.shape)
            # self.results['gradients'] = mu_f.reshape(Xn.shape)
            self.results['gradients'] = mu_f.reshape(shape_org)
        if 'hessian' in properties:
            # before = time.time()
            K_s = k(Xm, Xn, hessian_only=True)            # (D+1)N x DDM
            # after = time.time()
            # print('Hessian kernel construction', after - before, 's')
            # @@@@@@@@@@@ orig
            # before = time.time()
            H = m.H(Xn) + K_s.T @ K_y_inv @ (Y - m(Xm))  # DDM
            # after = time.time()
            # print('Hessian matrix multiplication ', after - before, 's')
            # @@@@@@@@@@@@ no forces
            # Y = data['V']
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # H = m.dm(Xm) + K_s.T @ K_y_inv @ (Y - m(Xn, hess=False))  # DDM
            #####
            # s = shape_org[:-1]
            D = np.prod(shape_org[:-1])
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
