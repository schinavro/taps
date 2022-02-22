import numpy as np
from numpy.linalg import inv
from taps.models import Model
from taps.ml.kernels import Kernel
from taps.ml.means import Mean
from taps.projectors import Projector
from taps.coords import Cartesian


def serialize(database, prj):
    """
    Database serialization function
    """
    entries = dict(coord='blob', potential='blob', gradients='blob')
    data_list = database.read_all(entries=entries)
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
    dm = m.get_gradients(coords=data['coords']).flatten()
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

    def set_lambda(self, imgdb, Θk=None, Θm=None):
        if Θk is not None:
            self.kernel.set_hyperparameters(Θk)
        if Θm is not None:
            self.mean.set_hyperparameters(Θm)

        self.data = serialize(imgdb, self.prj)
        self.Xm, self.Y_m = reshape_data(self.mean, self.data)
        k = self.kernel
        self.K_y_inv = inv(k(self.Xm, self.Xm, noise=True))
        self.Λ = self.K_y_inv @ self.Y_m

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

        Xm, Λ = self.Xm, self.Λ
        Xn = coords.reshape(D, N)
        # dY = data['kernel_gradients'].flatten()

        def m(x): return self.mean.get_potential(coords=x, **kwargs)
        def dm(x): return self.mean.get_gradients(coords=x, **kwargs).flatten()
        def ddm(x): return self.mean.get_hessian(coords=x, **kwargs).flatten()

        k = self.kernel
        if 'potential' in properties:
            K_s = k(Xm, Xn, potential_only=True)          # (D+1)M x N
            potential = m(Xn) + K_s.T @ Λ                 # N
            self.results['potential'] = potential
        if 'gradients' in properties:
            dK_s = k(Xm, Xn, gradient_only=True)          # (D+1)M x DN
            mu_f = dm(Xn) + dK_s.T @ Λ                    # DN
            self.results['gradients'] = mu_f.reshape(*orig_shape, N)
        if 'hessian' in properties:
            ddK_s = k(Xm, Xn, hessian_only=True)          # (D+1)M x DDN
            H = ddm(Xn) + ddK_s.T @ Λ                     # DDN
            self.results['hessian'] = H.reshape(D, D, N)

        if 'covariance' in properties:
            K_y_inv = self.K_y_inv
            K = k(Xn, Xn, orig=True)
            K_s = k(Xm, Xn)
            K_s_T = k(Xn, Xm)
            self.results['covariance'] = K - (K_s_T @ K_y_inv @ K_s)[:N, :N]

    def get_covariance(self, **kwargs):
        cov_coords = self.get_properties(properties='covariance', **kwargs)
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


class Likelihood:
    """
    """
    def __init__(self, kernel=None, mean=None, database=None, kernel_prj=None):
        self.k, self.m = kernel, mean
        kernel_prj = kernel_prj or Projector()
        data = serialize(database, kernel_prj)
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
