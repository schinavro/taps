import numpy as np
from numpy.linalg import inv
from taps.models import Model
from taps.ml.kernels import Kernel
from taps.ml.means import Mean
from taps.projectors import Projector
from taps.coords import Cartesian


def serialize(database, prj, ids=None):
    """
    Database serialization function
    """
    entries = dict(coord='blob', potential='blob', gradients='blob')
    if ids is None:
        data_list = database.read_all(entries=entries)
    else:
        data_list = database.read(ids=ids, entries=entries)
    M = len(data_list)
    coords_shape = data_list[0]['coord'].shape
    gradients_shape = data_list[0]['gradients'].shape[:-1]
    coords = Cartesian(coords=np.zeros((*coords_shape, M)))
    potential = np.zeros(M)
    gradients = np.zeros((*gradients_shape, M))

    data = dict()
    for m in range(M):
        datum = data_list[m]
        coords.coords[..., m] = datum['coord']
        potential[m] = datum['potential']
        gradients[..., m] = datum['gradients'][..., 0]

    data['coords'] = coords
    data['potential'] = potential
    data['gradients'] = gradients

    data['kernel_coords'] = prj.x(coords)
    data['kernel_potential'] = potential
    data['kernel_gradients'] = prj.f(gradients, coords)[0]

    return data


def reshape_data(mean, data):
    m = mean
    dm = m.get_gradients(coords=data['kernel_coords']).flatten()
    potential = data['kernel_potential'] - m(data['coords'])
    gradients = data['kernel_gradients'].flatten() - dm

    X = data['kernel_coords'].coords
    Y_m = np.concatenate([potential, gradients])
    return X, Y_m


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
                 regression=None,  # Deprecated
                 **kwargs):
        """
        data array
        """
        self.real_model = real_model or self
        self.kernel = kernel or Kernel()
        self.mean = mean or Mean()

        super().__init__(**kwargs)

    def set_lambda(self, imgdb, ids=None, Θk=None, Θm=None):
        if Θk is not None:
            self.kernel.set_hyperparameters(Θk)
        if Θm is not None:
            self.mean.set_hyperparameters(Θm)

        data = serialize(imgdb, self.prj, ids=ids)
        Xm, Y_m = reshape_data(self.mean, data)
        self._cache['Xm'], self._cache['Y_m'] = Xm, Y_m
        k = self.kernel
        K_y_inv = inv(k(Xm, Xm, noise=True))
        self._cache['K_y_inv'] = K_y_inv
        self._cache['Λ'] = K_y_inv @ Y_m

    def calculate(self, coords, imgdb=None, properties=None, **kwargs):
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
        if imgdb is not None:
            self.set_lambda(imgdb)
        # data = paths.get_image_data(prj=self.prj)

        orig_shape = coords.shape[:-1]
        D, N = np.prod(coords.D), coords.N

        def m(x): return self.mean.get_potential(coords=x, **kwargs)
        def dm(x): return self.mean.get_gradients(coords=x, **kwargs).flatten()
        def ddm(x): return self.mean.get_hessian(coords=x, **kwargs).flatten()
        k = self.kernel

        Xn = coords.reshape(D, N)
        if self._cache.get('Λ') is not None:
            Xm, Λ = self._cache['Xm'], self._cache['Λ']
            if 'hessian' in properties:
                K_s, dK_s, ddK_s = k(Xm, Xn, hessian_only=True)  # (D+1)M x DDN
            elif 'gradients' in properties:
                K_s, dK_s = k(Xm, Xn, gradient_only=True)     # (D+1)M x DN
            else:
                K_s = k(Xm, Xn, potential_only=True)             # (D+1)M x N
        else:
            Λ = np.array([0.])
            K_s, dK_s, ddK_s = Λ, Λ, Λ

        if 'potential' in properties:
            potential = m(Xn) + K_s.T @ Λ                 # N
            self.results['potential'] = potential
        if 'gradients' in properties:
            mu_f = dm(Xn) + dK_s.T @ Λ                    # DN
            self.results['gradients'] = mu_f.reshape(*orig_shape, N)
        if 'hessian' in properties:
            # H = ddm(Xn) + ddK_s.T @ Λ                     # DDN
            # self.results['hessian'] = H.reshape(D, D, N)
            hessian = self.get_finite_hessian(coords=coords).T
            self.results['hessian'] = hessian
        if 'covariance' in properties:
            K_y_inv = self._cache['K_y_inv']
            K = k(Xn, Xn, orig=True)
            K_s = k(Xm, Xn)
            K_s_T = k(Xn, Xm)
            self.results['covariance'] = K - (K_s_T @ K_y_inv @ K_s)[:N, :N]
            # self.results['covariance'] = np.zeros((N, N))

    def get_covariance(self, sigma=3., **kwargs):
        cov_coords = self.get_properties(properties='covariance', **kwargs)
        _ = np.diag(cov_coords)
        cov_coords = _.copy()
        cov_coords[_ < 0] = 0
        return sigma * np.sqrt(cov_coords)

    def get_hyperparameters(self, hyperparameters_list=None):
        return self.kernel.get_hyperparameters(hyperparameters_list)

    def get_state_info(self):
        info = []
        for k, v in self.kernel.hyperparameters.items():
            info.append(f"{k: <11}" + ": " + str(v))
        return '\n'.join(info)


class Likelihood:
    """
    """
    def __init__(self, kernel=None, mean=None, database=None, kernel_prj=None,
                 ids=None):
        self.k, self.m = kernel, mean
        kernel_prj = kernel_prj or Projector()
        data = serialize(database, kernel_prj, ids=ids)
        self.X, self.Y_m = reshape_data(mean, data)
        self.Y_m_T = self.Y_m.T

    def __call__(self, hyperparameters):
        self.k.set_hyperparameters(hyperparameters)
        K = self.k(self.X, self.X, noise=True)
        detK = np.linalg.det(K)
        try:
            detK = np.diagonal(np.linalg.cholesky(K))
            log_detK = np.sum(np.log(detK))
        except np.linalg.LinAlgError:
            # Postive definite matrix
            detK = np.linalg.det(K)
            # print(detK)
            if detK <= 1e-5:
                log_detK = -5
            else:
                log_detK = np.log(detK)
        return log_detK + 0.5 * (self.Y_m_T @ (inv(K) @ self.Y_m))


class NonGradientLikelihood(Likelihood):
    def __call__(self, hyperparameters):
        self.k.set_hyperparameters(hyperparameters)
        K = self.k(self.X, self.X, noise=True)
        detK = np.linalg.det(K)
        try:
            detK = np.diagonal(np.linalg.cholesky(K))
            log_detK = np.sum(np.log(detK))
        except np.linalg.LinAlgError:
            # Postive definite matrix
            detK = np.linalg.det(K)
            # print(detK)
            if detK <= 1e-5:
                log_detK = -5
            else:
                log_detK = np.log(detK)
        return log_detK + 0.5 * (self.Y_m.T @ (inv(K) @ self.Y_m))
