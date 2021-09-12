import time
import numpy as np
from numpy import newaxis as nax
from numpy import log, sum, diagonal
from numpy.linalg import inv, cholesky
from scipy.optimize import minimize

from taps.model.model import Model
from taps.utils.shortcut import dflt, isstr, isbool, isDct, asst
from taps.ml.kernels.kernel import Kernel
from taps.ml.neural import Mean


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
    model_parameters = {
        'kernel': {'default': "'Kernel'", 'assert': 'True',
                   'class': True, 'from': 'taps.ml.gaussian'},
        'kernel_type': {'default': "'Kernel'", 'assert': 'True'},
        'mean': {'default': "'Mean'", 'assert': 'True',
                 'class': True, 'from': 'taps.ml.gaussian'},
        'mean_type': {'default': "'average'", 'assert': isstr},
        'optimized': {'default': 'None', 'assert': isbool},
        'hyperparameters': {'default': 'dict()', 'assert': isDct},
        'hyperparameters_bounds': {'default': 'dict()', 'assert': isDct},
        'regression_method': {'default': '"L-BFGS-B"', 'assert': isstr},
        # 'likelihood_type': {dflt: '"pseudo_gradient_likelihood"', asst: isstr}
        # 'likelihood_type': {dflt: '"gradient_likelihood"', asst: isstr},
        'likelihood_type': {dflt: '"likelihood"', asst: isstr}
    }

    def __init__(self, real_model='Model', kernel='Kernel', mean='Mean',
                 mean_type=None, kernel_type=None,
                 optimized=None, hyperparameters=None,
                 hyperparameters_bounds=None, regression_method=None,
                 likelihood_type=None, **kwargs):
        """
        data array
        """
        # Silence!
        Kernel, Mean
        self.real_model = real_model
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)

        self.kernel = kernel
        self.kernel_type = kernel_type
        self.mean = mean
        self.mean_type = mean_type
        self.optimized = optimized
        self.hyperparameters = hyperparameters
        self.hyperparameters_bounds = hyperparameters_bounds
        self.regression_method = regression_method
        self.likelihood_type = likelihood_type
        self.data_ids = {}

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
        shape_org = coords.shape
        self.results = getattr(self, 'results', {})
        k, m = self.kernel, self.mean
        X = coords.copy()
        data = self.get_data(paths)
        Xm = self.flatten(paths, data['X'])
        Xn = self.flatten(paths, coords)
        Y = k.shape_data(data)    # N
        if len(Y) == 0:  # If no data given
            return np.zeros(X.shape[-1])

        # Xn = np.atleast_3d(Xn)
        # Xm = np.atleast_3d(Xm)
        D, N = Xn.shape
        D, M = Xm.shape
        #   len(self._cache['K_y_inv']) != (D + 1) * P:
        if not self.optimized or self._cache == {} or \
                self._cache.get('K_y_inv') is None:
            m.type = self.mean_type
            self.hyperparameters = self.regression(paths)
            k.hyperparameters.update(self.hyperparameters)
            # k.hyperparameters.update(self.regression(data))
            self._cache['K_y_inv'] = inv(k(Xm, Xm, noise=True))    # N x N x P
            m._data = data
            self.optimized = True

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
            potential = m(Xn, hess=False) + K_s.T @ K_y_inv @ (Y - m(Xm))
            # Y = data['V']  # @@@
            # N = len(Y)  # @@@
            # K_s = K_s[:N] # @@@
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # return m(Xm, hess=False) + \
            #      K_s.T @ K_y_inv @ (Y - m(Xn, hess=False)) # @@@
            self.results['potential'] = potential
        if 'gradients' in properties:
            dK_s = k(Xm, Xn, gradient_only=True)  # (D+1)N x (D+1)M x P
            mu_f = m.dm(Xn) + dK_s.T @ K_y_inv @ (Y - m(Xm))
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
            H = m.dm(Xn) + K_s.T @ K_y_inv @ (Y - m(Xm))  # DDM
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
        # sigma_f = self.model.hyperparameters.get('sigma_f', 1)
        return 1.96 * np.sqrt(cov_coords) / 2

    def regression(self, paths, likelihood_type=None, dx=1e-2,
                   **reg_kwargs):
        """
        k : class Kernel; k(X, X) ((DxN + 1) x m) x ((DxN + 1) x n) array
        X : imgdata['X']; position of atoms, (D x N) x m dimension array
        Y : imgdata['Y']; energy and forces of atoms One dimensional array
            with length (m + m x (D x N)) Potential comes first.
        m : mean function
        M : a number of data
        """
        M = getattr(self, 'data_ids', None)
        if M is None or len(M.get('image', [])) == 0:
            return self.hyperparameters
        k, m = self.kernel, self.mean
        m.type = self.mean_type
        likelihood_type = likelihood_type or self.likelihood_type
        if likelihood_type == 'pseudo_gradient_likelihood':
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
        else:
            hess = likelihood_type == 'gradient_likelihood'
            data = self.get_data(paths)
            X = self.flatten(paths, data['X'])
            Y = k.shape_data(data, hess=hess)
            Y_m = Y - m(X, data, hess=hess)
        reg_kwargs = self.reg_kwargs(**reg_kwargs)

        def log_likelihood(k, X, Y_m, likelihood_type):
            def likelihood(hyperparameters_list):
                k.set_hyperparameters(hyperparameters_list)
                K = k(X, X, orig=True)
                detK = diagonal(cholesky(K))
                return sum(log(detK)) + 0.5 * (Y_m.T @ (inv(K) @ Y_m))

            def gradient_likelihood(hyperparameters_list):
                k.set_hyperparameters(hyperparameters_list)
                K = k(X, X, noise=True)
                detK = np.linalg.det(K)
                if detK <= 1e-5:
                    log_detK = -5
                else:
                    log_detK = log(detK)
                print()
                return log_detK + 0.5 * (Y_m.T @ (inv(K) @ Y_m))

            def pseudo_gradient_likelihood(hyperparameters_list):
                k.set_hyperparameters(hyperparameters_list)
                K = k(X, X, noise=True, orig=True)
                detK = diagonal(cholesky(K))
                return sum(log(detK)) + 0.5 * (Y_m.T @ (inv(K) @ Y_m))

            return locals()[likelihood_type]

        res = minimize(log_likelihood(k, X, Y_m, likelihood_type), **reg_kwargs)
        self.optimized = True
        return self.kernel.get_hyperparameters(res.x)

    def get_hyperparameters(self, hyperparameters_list=None):
        return self.kernel.get_hyperparameters(hyperparameters_list)

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

    def get_data(self, paths, coords=None, data_ids=None,
                 keys=['coords', 'potential', 'gradients']):
        """
        data_ids : dictionary of lists
            {'image': [...], 'descriptor': [...]}
        for each key, return
         'coords' -> 'X'; D x M x P
         'potential'    -> 'V'; P
         'gradient'    -> 'F'; D x M x P
        return : dictionary contain X, V, F
            {'X' : np.array(D x M x P), 'V': np.array(P), }
        """
        data_ids = data_ids or getattr(self, 'data_ids', None)
        if data_ids is None or len(data_ids.get('image', [])) == 0:
            return None
        # shape = coords.shape[:-1]
        # shape = (3, 5)
        M = len(data_ids['image'])
        if self._cache.get('data_ids_image') is None:
            shape = self.prj.x(paths.coords(index=[0])).shape[:-1]
            self._cache['data_ids_image'] = []
            self._cache['data'] = {'X': np.zeros((*shape, 0), dtype=float),
                                   'V': np.zeros(0, dtype=float),
                                   'F': np.zeros((*shape, 0), dtype=float)}
        if self._cache.get('data_ids_image') == data_ids['image']:
            return self._cache['data']
        else:
            new_data_ids_image = []
            for id in data_ids['image']:
                if id not in self._cache['data_ids_image']:
                    new_data_ids_image.append(id)
        atomdata = paths.imgdata
        name2idx = atomdata.name2idx
        n2i = name2idx['image']
        new_data = atomdata.read({'image': new_data_ids_image})['image']
        M = len(new_data_ids_image)
        data = self._cache['data']
        if 'coords' in keys:
            coords_raw = []
            for i in range(M):
                coord_raw = new_data[i][n2i['coord']][..., nax]
                coords_raw.append(self.prj._x(coord_raw))
            if M != 0:
                new_coords = np.concatenate(coords_raw, axis=-1)
                data['X'] = np.concatenate([data['X'], new_coords], axis=-1)
        if 'potential' in keys:
            potential = []
            for i in range(M):
                potential.append(new_data[i][n2i['potential']])
            if M != 0:
                new_potential = np.concatenate(potential, axis=-1)
                data['V'] = np.concatenate([data['V'], new_potential], axis=-1)
        if 'gradients' in keys:
            gradients = []
            for i in range(M):
                coords_raw = new_data[i][n2i['coord']][..., nax]
                gradients_raw = new_data[i][n2i['gradients']]
                gradients_prj, _ = self.prj.f(gradients_raw, coords_raw)
                gradients.append(gradients_prj)
            if M != 0:
                new_gradients = np.concatenate(gradients, axis=-1)
                data['F'] = np.concatenate([data['F'], -new_gradients], axis=-1)
        self._cache['data_ids_image'].extend(new_data_ids_image)
        return data

    def flatten(self, paths, coords):
        """
        3 x A x N -> 3A x N
        D x N -> D x N
        """
        shape = coords.shape
        if len(shape) == 3:
            D, N = np.prod(shape[:-1]), shape[-1]
            return coords.reshape(D, N)
        return coords


class Atomic(Model):
    def calculate(self, paths, coords=None, properties=None, **kwargs):
        """
        Xm : M x A x Q
        """

        symbols = paths.symbols
        k = self.kernels
        m = self.means
        K_y_inv = []
        if self.optimized and self._cache.get('K_y_invs') is not None:
            for i, sym in enumerate(symbols):
                Xn, Y = self.read_data(paths, idx=i, symbol=sym)
                K_y_inv.append()
                K_s = None

        results = {}
        Xm = self.desc(paths, coords)
        for i, s in enumerate(symbols):
            if 'potential' in properties or 'potentials' in properties:
                K_s = k(Xn, Xm, potential_only=True)  # (D+1)N x M
                results['potentials'] = m(Xm, hess=False) + K_s.T @ K_y_inv @ (Y - m(Xn))
            if 'forces' in properties:
                K_s = k(Xn, Xm, potential_only=True)  # (D+1)N x M
                m(Xm, hess=False) + K_s.T @ K_y_inv @ (Y - m(Xn))
        if 'potential' in properties or 'potentials' in properties:
            results['potential'] = results['potentials'].sum(axis=2)
        self.results = results

    def read_data(self, paths, idx=None, symbol=None):
        # table = self.table_name or
        table = 'sbdesc'
        paths.imgdata.query(table=table, search_word='symbols=%s' % symbol)
        Xn = None
        Y = None
        return Xn, Y
