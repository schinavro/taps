import os
from collections import OrderedDict
import numpy as np
from numpy import identity as I
from numpy import newaxis as nax
from numpy import vstack, log, cos, sin, sum, diagonal, atleast_3d
from numpy.linalg import inv, cholesky, solve
from scipy.optimize import minimize
from taps.model import Model
from taps.data import PathsData
from taps.pathfinder import PathFinder
from taps.utils import dflt, isstr, isbool, isDct, asst, isLst


class Kernel:
    key2idx = {'sigma_f': 0, 'l^2': 1, 'sigma_n^e': 2, 'sigma_n^f': 3}
    hyperparameters = {'sigma_f': 1, 'l^2': 1, 'sigma_n^e': 0, 'sigma_n^f': 0}

    def __init__(self, **hyperparameters):
        pass
        # self.hyperparameters.update(hyperparameters)

    def __call__(self, Xn=None, Xm=None, orig=False, noise=False,
                 hyperparameters=None, gradient_only=False, hessian_only=False,
                 potential_only=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        http://home.zcu.cz/~jacobnzw/pdf/2016_mlsp_gradients.pdf
        978-1-5090-0746-2/16/$31.00 c 2016 IEE
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        ll = hyperparameters.get('l^2')
        sig_f = hyperparameters.get('sigma_f')
        if Xn is None:
            Xn = Xm.copy()
        Xm = atleast_3d(Xm)
        Xn = atleast_3d(Xn)
        N = Xn.shape[-1]
        M = Xm.shape[-1]
        D = np.prod(Xm.shape[:-1])

        X = Xn.reshape(D, N)
        Y = Xm.reshape(D, M)

        Xnm = X[:, :, nax] - Y[:, nax, :]            # D x N x M
        dists = (Xnm ** 2).sum(axis=0)               # N x M
        K = sig_f * np.exp(-.5 * dists / ll)         # N x M
        if orig:
            if noise:
                noise_f = hyperparameters.get('sigma_n^e', 0)
                return K + noise_f * I(N)    # N x M
            return K
        # Derivative coefficient D x N x M
        dc_gd = -Xnm / ll
        # DxNxM x NxM -> DxNxM -> DNxM
        Kgd = np.vstack(dc_gd * K[nax, ...])
        if potential_only:
            return vstack([K, Kgd])                  # (D+1)xN x M
        # DxNxM * 1xNxM -> NxDM
        Kdg = np.hstack(-dc_gd * K[nax, ...])
        # DxNxM -> NxDxM
        Xmn = np.swapaxes(Xnm, 0, 1)
        # DxNx1xM  * 1xNxDxM  -> D x N x D x M
        # dc_dd_glob = -Xnm[:, :, nax, :] * Xmn[nax, :, :, :] / ll / ll
        dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
        # ∂_mn exp(Xn - Xm)^2
        dc_dd_diag = I(D)[:, nax, :, nax] / ll
        # DxNxDxM - DxNxDxM
        Kdd = (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
        # DN x DM
        Kdd = Kdd.reshape(D * N, D * M)
        if gradient_only:
            # (D+1)N x DM
            return vstack([Kdg, Kdd])
        if hessian_only:
            # Delta _ dd
            dnm = np.arange(D)
            # DxNxM * DxNxM -> NxDxDxM
            dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
            # DxNxM -> NxDxM -> NxDDxM
            dc_hg_diag = np.zeros((N, D, D, M))
            # ∂_mm K(Xm,Xn) NxDxDxM
            dc_hg_diag[:, dnm, dnm, :] = -1 / ll
            # NxDxDxM + NxDxDxM -> NxDxDxM
            Khg = (dc_hg_glob + dc_hg_diag)
            # Bacground term: ∂_mmn K(Xm,Xn) DxNx1x1xM * 1xNxDxDxM -> DxNxDxDxM
            dc_hd_back = dc_gd[:, :, nax, nax, :] * Khg[nax, ...]
            # Diagonal term : DxNxDxDxM * 1xNxDxDxM -> DxNxDxDxM
            dc_hd_diag = np.zeros((D, N, D, D, M))
            # Global term :
            dc_hd_glob = np.zeros((D, N, D, D, M))
            dc_hd_glob[dnm, :, dnm, ...] += Xmn[nax, :, :, :] / ll / ll
            dc_hd_glob[dnm, :, :, dnm, ...] += Xmn[nax, :, :, :] / ll / ll
            # print(dc_hd_glob[0, 0, 0, :, 0])
            # print(dc_hd_glob.reshape(2, N, 2))
            Khd = (dc_hd_glob + dc_hd_diag) + dc_hd_back
            # NxDxDxM x Nx1x1xM -> NxDxDxM
            Khg *= K[:, nax, nax, :]
            # DxNxDxDxM * 1xNx1x1xM
            Khd *= K[nax, :, nax, nax, :]
            # NxDDxM -> N x DDM
            Khg = Khg.reshape(N, D * D * M)
            # DxNxDDxM -> DN x DxDxM
            Khd = Khd.reshape(D * N, D * D * M)
            # print(Khd.shape)
            return np.vstack([Khg, Khd])  # (D+1)N x DDM

        Kext = np.block([[K, Kdg],
                        [Kgd, Kdd]])  # (D+1)N x (D+1)M
        if noise:
            noise_f = hyperparameters.get('sigma_n^e', 0)
            noise_df = hyperparameters.get('sigma_n^f', 0)
            noise = np.array([noise_f] * N + [noise_df] * D * N)
            return Kext + noise * I((D + 1) * N)
        return Kext

    def set_hyperparameters(self, hyperparameters_list=None):
        for key, idx in self.key2idx.items():
            self.hyperparameters[key] = hyperparameters_list[idx]

    def get_hyperparameters(self, hyperparameters_list=None):
        hyperparameters = {}
        for key, idx in self.key2idx.items():
            hyperparameters[key] = hyperparameters_list[idx]
        return hyperparameters

        if hyperparameters_list is None:
            return self.hyperparameters

    def shape_data(self, data, hess=True):
        D, M, P = data['X'].shape
        if P == 0:
            return np.zeros(0)
        if not hess:
            return data['V']
        return np.vstack([data['V'], -data['F'].reshape(D * M, P)]).flatten()

    def __repr__(self):
        return self.__class__.__name__


class AtomicDistanceKernel(Kernel):
    def __call__(self, Xm, Xn=None, hyperparameters=None):
        """
        https://link.springer.com/chapter/10.1007/978-3-319-70087-8_55"""
        if Xm is not None:
            pass


class DescriptorKernel(Kernel):
    from taps.descriptor import SphericalHarmonicDescriptor
    """
    period : NxD array
    """
    hyperparameters = {'sigma_f': 1, 'l^2': 1, 'sigma_n^e': 1,
                       'sigma_n^f': 1}

    def __init__(self, weights=None, cutoff_radius=None, n_atom=None,
                 n_max=None, libname='libsbdesc.so',
                 libdir='/group/schinavro/libCalc/sbdesc/lib/'):
        desc_kwargs = {'weights': weights, 'cutoff_radius': cutoff_radius,
                       'n_atom': n_atom, 'n_max': n_max,
                       'libname': 'libsbdesc.so',
                       'libdir': '/group/schinavro/libCalc/sbdesc/lib/'}
        self.descriptor = self.SphericalHarmonicDescriptor()

        # self.hyperparameters.update(hyperparameters)

    def __call__(self, Xn=None, Xm=None, orig=False, noise=False,
                 hyperparameters=None, gradient_only=False, hessian_only=False,
                 potential_only=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        Xn : D x M x N array must be Scaled_coordinate
        X : D x N array, where DxN -> D; P -> N
            Left argument of the returned kernel k(X, Y)
        Xm : D x M x M array
        Y : D x M array
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        http://home.zcu.cz/~jacobnzw/pdf/2016_mlsp_gradients.pdf
        978-1-5090-0746-2/16/$31.00 c 2016 IEE
        https://arxiv.org/pdf/1703.04389.pdf
        Q : n_max * (n_max + 1) / 2
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if Xn is None:
            Xn = Xm.copy()
        Xm = atleast_3d(Xm).astype(float)  # 3 x A x M
        Xn = atleast_3d(Xn).astype(float)  # 3 x A x N
        Dm = self.descriptor(Xm)  # A x Q x M
        Dn = self.descriptor(Xn)  # A x Q x N
        N = Dn.shape[-1]
        M = Dm.shape[-1]
        A = Dn.shape[1]

        K = np.zeros((A, M, N))
        for i, X, Y in zip(np.arange(A), Dm, Dn):
            K[i] = np.einsum('ijk, ijl -> ikl', X, Y)
        if orig:
            if noise:
                noise_f = hyperparameters.get('sigma_n^e', 0)
                return K + noise_f * I(N)
            return K                               # N x M


class PeriodicKernel(Kernel):
    """
    period : NxD array
    """
    hyperparameters = {'sigma_f': 1, 'l^2': 1, 'sigma_n^e': 1,
                       'sigma_n^f': 1}

    def __init__(self, period=2 * np.pi, **hyperparameters):
        self.period = period
        # self.hyperparameters.update(hyperparameters)

    def scaled_coords(self, coords, D=None, M=None, P=None):
        """
        coords : D x M x P
        return : D x M x P
        """
        p = coords.copy()
        if D is None:
            D, M, P = coords.shape
        if P == 0:
            return np.zeros((D, M, P))
        Dcell = len(self.period)
        assert D == Dcell, 'Dim: cell %d != coords %d' % (Dcell, D)
        scaled_coords = solve(self.period.T, p.reshape(D * M, P))
        raise NotImplementedError()
        return scaled_coords.reshape(D, M, P)

    def __call__(self, Xn=None, Xm=None, orig=False, noise=False,
                 hyperparameters=None, gradient_only=False, hessian_only=False,
                 potential_only=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        Xn : D x M x N array must be Scaled_coordinate
        X : D x N array, where DxN -> D; P -> N
            Left argument of the returned kernel k(X, Y)
        Xm : D x M x M array
        Y : D x M array
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        http://home.zcu.cz/~jacobnzw/pdf/2016_mlsp_gradients.pdf
        978-1-5090-0746-2/16/$31.00 c 2016 IEE
        https://arxiv.org/pdf/1703.04389.pdf
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        ll = hyperparameters.get('l^2', 1)
        sig_f = hyperparameters.get('sigma_f', 0)
        if Xn is None:
            Xn = Xm.copy()
        Xm = atleast_3d(Xm).astype(float)
        Xn = atleast_3d(Xn).astype(float)
        # scaled_Xm = self.scaled_coords(Xm, *(Xm.shape))
        # scaled_Xn = self.scaled_coords(Xn, *(Xn.shape))
        N = Xn.shape[-1]
        M = Xm.shape[-1]
        D = np.prod(Xm.shape[:-1])

        # X = scaled_Xn.reshape(D, N)
        # Y = scaled_Xm.reshape(D, M)
        X = Xn.reshape(D, N)
        Y = Xm.reshape(D, M)

        pi = np.pi / self.period
        Xnm = pi * (X[:, :, nax] - Y[:, nax, :])   # D x N x M
        dists = (sin(Xnm) ** 2).sum(axis=0)        # N x M
        K = sig_f * np.exp(-dists / ll)            # N x M

        if orig:
            if noise:
                noise_f = hyperparameters.get('sigma_n^e', 0)
                return K + noise_f * I(N)
            return K                               # N x M
        # 2 x Cosnm * Sinnm   # D x N x M
        Sin2nm = sin(2 * Xnm)
        dc_gd = -pi * Sin2nm / ll
        # DxNxM x NxM -> DxNxM -> DNxM
        Kgd = np.vstack(dc_gd * K[nax, ...])
        if potential_only:
            return vstack([K, Kgd])                  # (D+1)xN x M
        # DxNxM x 1xNxM -> DxNxM -> NxDM
        Kdg = np.hstack(-dc_gd * K[nax, ...])
        # DxNxM -> NxDxM
        Sin2mn = np.swapaxes(Sin2nm, 0, 1)
        # # DxNx1xM  * 1xNxDxM  -> D x N x D x M
        pipi = pi * pi
        llll = ll * ll
        dc_dd_glob = -pipi * Sin2nm[:, :, nax] * Sin2mn[nax, ...] / llll
        # DxNx1xM x Dx1xDx1 -> DxNxDxM
        Cos2nm = cos(2 * Xnm)
        dc_dd_diag = 2 * pipi * Cos2nm[:, :, nax] * I(D)[:, nax, :, nax] / ll
        # DxNxDxM - DxNxDxM
        Kdd = dc_dd_glob + dc_dd_diag
        Kdd *= K[nax, :, nax, :]

        # DN x DM
        Kdd = Kdd.reshape(D * N, D * M)
        if gradient_only:
            return np.vstack([Kdg, Kdd])  # (D+1)N x DM
        if hessian_only:
            # https://arxiv.org/pdf/1704.00060.pdf
            dnm = np.arange(D)
            # DxNxM -> NxDxM
            Cos2mn = np.swapaxes(Cos2nm, 0, 1)
            # DxNxM * DxNxM -> NxDxDxM
            dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
            # DxNxM -> NxDxM -> NxDxM
            dc_hg_diag = np.zeros((N, D, D, M))
            # ∂_mm K(Xm,Xn) NxDxM <- NxDxM => NxDxDxM
            dc_hg_diag[:, dnm, dnm, :] = -2 * pipi * Cos2mn / ll
            # NxDxDxM + NxDxDxM -> NxDxDxM
            Khg = (dc_hg_glob + dc_hg_diag)
            # Bacground term: ∂_mmn K(Xm,Xn) DxNx1x1xM * 1xNxDxDxM -> DxNxDxDxM
            dc_hd_back = dc_gd[:, :, nax, nax, :] * Khg[nax, ...]
            # Diagonal term : DxNx1x1xM * 1xNxDxDxM -> DxNxDxDxM
            dc_hd_diag = np.zeros((D, N, D, D, M))
            pi3 = pi * pi * pi
            dc_hd_diag[dnm, :, dnm, dnm] += 4 * pi3 * Sin2nm / ll
            # Global term : DxNx1xM * 1xNxDxM -> DxNxDxM
            dc_hd_glob = np.zeros((D, N, D, D, M))
            Cos2_x_Sin2 = np.einsum('inm, jnm -> injm', Cos2nm, Sin2nm)
            dc_hd_glob[dnm, :, dnm, ...] += 2 * pi3 * Cos2_x_Sin2 / llll
            dc_hd_glob[dnm, :, :, dnm, ...] += 2 * pi3 * Cos2_x_Sin2 / llll
            Khd = dc_hd_glob + dc_hd_diag + dc_hd_back
            # NxDxDxM x Nx1x1xM -> NxDxDxM
            Khg *= K[:, nax, nax, :]
            # DxNxDxDxM * 1xNx1x1xM
            Khd *= K[nax, :, nax, nax, :]
            # NxDDxM -> N x DDM
            Khg = Khg.reshape(N, D * D * M)
            # DxNxDDxM -> DN x DxDxM
            Khd = Khd.reshape(D * N, D * D * M)
            # print(Khd.shape)
            # return Khg
            return np.vstack([Khg, Khd])  # (D+1)N x DDM

        Kext = np.block([[K, Kdg],
                        [Kgd, Kdd]])  # (D+1)N x (D+1)M
        if noise:
            noise_f = hyperparameters.get('sigma_n^e', 0)
            noise_df = hyperparameters.get('sigma_n^f', 0)
            noise = np.array([noise_f] * N + [noise_df] * D * N)
            return Kext + noise * I((D + 1) * N)
        return Kext


class Mean:
    def __init__(self, type='average', data=None):
        self.type = type
        self._data = data

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
        X = np.atleast_3d(X)
        V = data['V']
        D, M, P = X.shape
        F = np.zeros((D * M, P))
        # F = np.zeros((D, M, P)) + np.average(data['F'], axis=2)[..., nax]
        if type == 'average':
            e = np.zeros(P) + np.average(V)
        elif type == 'min':
            e = np.zeros(P) + np.min(V)
        else:
            e = np.zeros(P) + np.max(V)
        if not hess:
            return e
        ef = np.vstack([e, F])
        # ef = np.vstack([e, F.reshape((D * M, P))])
        return ef.flatten()

    def dm(self, X, data=None, hess=False):
        if data is None:
            data = self._data
        return 0


class Gaussian(Model):
    implemented_properties = {'covariance', 'potential', 'forces', 'hessian'}
    model_parameters = {
        'real_model': {'default': "'Model'", 'assert': 'True',
                       'class': True, 'from': 'taps.model'},
        'kernel': {'default': "'Kernel'", 'assert': 'True',
                   'class': True, 'from': 'taps.gaussian'},
        'mean': {'default': "'Mean'", 'assert': 'True',
                 'class': True, 'from': 'taps.gaussian'},
        'mean_type': {'default': "'average'", 'assert': isstr},
        'kernel_type': {'default': "'Total'", 'assert': isstr},
        'optimized': {'default': 'None', 'assert': isbool},
        'hyperparameters': {'default': 'dict()', 'assert': isDct},
        'hyperparameters_bounds': {'default': 'dict()', 'assert': isDct},
        'regression_method': {'default': '"L-BFGS-B"', 'assert': isstr},
        # 'likelihood_type': {dflt: '"pseudo_gradient_likelihood"', asst: isstr}
        'likelihood_type': {dflt: '"gradient_likelihood"', asst: isstr}
    }

    def __init__(self, real_model='Model', kernel='Kernel', mean='Mean',
                 mean_type=None, kernel_type=None,
                 optimized=None, hyperparameters=None,
                 hyperparameters_bounds=None, regression_method=None,
                 likelihood_type=None, **kwargs):
        """
        data array
        """
        self.real_model = real_model
        self.model_parameters.update(self.real_model.model_parameters)
        # super().implemented_properties.update(self.implemented_properties)
        # self.implemented_properties.update(super().implemented_properties)

        self.kernel = kernel
        self.kernel_type = kernel_type
        self.mean = mean
        self.mean_type = mean_type
        self.optimized = optimized
        self.hyperparameters = hyperparameters
        # self.kernel.hyperparameters.update(self.hyperparameters)
        # self.hyperparameters = self.kernel.hyperparameters
        self.hyperparameters_bounds = hyperparameters_bounds
        self.regression_method = regression_method
        self.likelihood_type = likelihood_type
        self._cache = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        elif key in ['real_model']:
            if type(value) == str:
                from_ = 'taps.model'
                module = __import__(from_, {}, None, [value])
                value = getattr(module, value)()
            super().__setattr__(key, value)
        elif key in self.real_model.model_parameters:
            default = self.model_parameters[key]['default']
            assertion = self.model_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super(Model, self.real_model).__setattr__(key, value)
            super().__setattr__(key, value)
        elif key in self.model_parameters:
            default = self.model_parameters[key]['default']
            assertion = self.model_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def calculate(self, paths, coords, properties=None, **kwargs):
        """
        P : int | length of image to consider / or number of image to calculate
        D : int | Dimension
        N : int | Number of Atom in one image
        X :
        Y :
        return : Dim x N x P - 2 array
        """
        self.results = getattr(self, 'results', {})
        k, m = self.kernel, self.mean
        X = coords.copy()
        data = self.get_data(paths)
        Xn = data['X']
        Xm = coords
        Y = k.shape_data(data)    # N
        if len(Y) == 0:  # If no data given
            return np.zeros(X.shape[-1])

        # Xn = np.atleast_3d(Xn)
        Xm = np.atleast_3d(Xm)

        D, M, P = Xm.shape
        #   len(self._cache['K_y_inv']) != (D + 1) * P:
        if not self.optimized or self._cache == {} or \
                self._cache.get('K_y_inv') is None:
            m.type = self.mean_type
            self.hyperparameters = self.regression(paths)
            k.hyperparameters.update(self.hyperparameters)
            # k.hyperparameters.update(self.regression(data))
            self._cache['K_y_inv'] = inv(k(Xn, Xn, noise=True))    # N x N x P
            m._data = data
            self.optimized = True

        K_y_inv = self._cache['K_y_inv']

        if 'potential' in properties:
            K_s = k(Xn, Xm, potential_only=True)  # (D+1)N x M
            potential = m(Xm, hess=False) + K_s.T @ K_y_inv @ (Y - m(Xn))
            # Y = data['V']  # @@@
            # N = len(Y)  # @@@
            # K_s = K_s[:N] # @@@
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # return m(Xm, hess=False) + \
            #      K_s.T @ K_y_inv @ (Y - m(Xn, hess=False)) # @@@
            self.results['potential'] = potential
        if 'forces' in properties:
            dK_s = k(Xn, Xm, gradient_only=True)  # (D+1)N x (D+1)M x P
            mu_f = m.dm(Xm) + dK_s.T @ K_y_inv @ (Y - m(Xn))
            # N = len(Y) # @@@
            # dK_s = dK_s[:N] # @@@
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # mu_f = m.dm(Xm) + dK_s.T @ K_y_inv @ (Y - m(Xn, hess=False))
            # return -mu_f.reshape(Xm.shape)
            self.results['forces'] = -mu_f.reshape(Xm.shape)
        if 'hessian' in properties:
            K_s = k(Xn, Xm, hessian_only=True)            # (D+1)N x DDM
            # @@@@@@@@@@@ orig
            H = m.dm(Xm) + K_s.T @ K_y_inv @ (Y - m(Xn))  # DDM
            # @@@@@@@@@@@@ no forces
            # Y = data['V']
            # K_y_inv = inv(k(Xn, Xn, orig=True))  # @@@
            # H = m.dm(Xm) + K_s.T @ K_y_inv @ (Y - m(Xn, hess=False))  # DDM
            #####
            self.results['hessian'] = H.reshape(D, M, D, M, P)
        if 'covariance' in properties:
            K = k(Xm, Xm, orig=True)
            K_s = k(Xn, Xm)
            K_s_T = k(Xm, Xn)
            self.results['covariance'] = K - (K_s_T @ K_y_inv @ K_s)[:P, :P]
        if 'potential_and_forces' in properties:
            P = len(Xm[..., :])
            K_s = k(Xm, Xn)  # (D+1)N x (D+1)M x P
            mu = m(Xm) + K_s.T @ K_y_inv @ (Y - m(Xn))
            E = mu[: P]
            F = -mu[P:].reshape(D, M, P)
            self.results['potential_and_forces'] = E, F

    def get_covariance(self, paths, **kwargs):
        return self.get_properties(paths, properties=['covariance'], **kwargs)

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
        if likelihood_type is None:
            likelihood_type = self.likelihood_type
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
            X = data['X']
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
                # @@@@@
                hyperparameters_list = np.concatenate([[1], hyperparameters_list])
                # @@@@@
                k.set_hyperparameters(hyperparameters_list)
                K = k(X, X, noise=True)
                detK = np.linalg.det(K)
                if detK <= 1e-5:
                    log_detK = -5
                else:
                    log_detK = log(detK)
                return log_detK + 0.5 * (Y_m.T @ (inv(K) @ Y_m))

            def pseudo_gradient_likelihood(hyperparameters_list):
                k.set_hyperparameters(hyperparameters_list)
                K = k(X, X, noise=True, orig=True)
                detK = diagonal(cholesky(K))
                return sum(log(detK)) + 0.5 * (Y_m.T @ (inv(K) @ Y_m))

            return locals()[likelihood_type]

        res = minimize(log_likelihood(k, X, Y_m, likelihood_type), **reg_kwargs)
        self.optimized = True
        # return self.kernel.get_hyperparameters(res.x)
        return self.kernel.get_hyperparameters(np.concatenate([[1], res.x]))

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
        # @@@@@
        x0 = x0[1:]
        bounds = bounds[1:]
        # @@@@@
        return {'x0': x0, 'bounds': bounds, 'method': method}

    def get_data(self, paths, data_ids=None,
                 keys=['coords', 'potential', 'gradient']):
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
        D, M = paths.DM
        if self._cache.get('data_ids_image') is None:
            self._cache['data_ids_image'] = []
            self._cache['data'] = {'X': np.zeros((D, M, 0), dtype=float),
                                   'V': np.zeros(0, dtype=float),
                                   'F': np.zeros((D, M, 0), dtype=float)}
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
        P = len(new_data_ids_image)
        data = self._cache['data']
        if 'coords' in keys:
            coords = np.zeros((D, M, P), dtype=float)
            for i in range(P):
                coords[..., i] = new_data[i][n2i['coord']]
            data['X'] = np.concatenate([data['X'], coords], axis=2)
        if 'potential' in keys:
            potential = np.zeros(P, dtype=float)
            for i in range(P):
                potential[i] = new_data[i][n2i['potential']]
            data['V'] = np.concatenate([data['V'], potential])
        if 'gradient' in keys:
            gradients = np.zeros((D, M, P), dtype=float)
            for i in range(P):
                gradients[..., i] = new_data[i][n2i['gradient']]
            data['F'] = np.concatenate([data['F'], -gradients], axis=2)
        self._cache['data_ids_image'].extend(new_data_ids_image)
        return data


class AtomicGaussian(Model):
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


class GaussianSearch(PathFinder):

    finder_parameters = {
        'real_finder': {'default': '"ADMD"', 'assert': 'True'},
        'log': {'default': 'None', 'assert': isstr},
        'phases': {'default': '["Maximum uncertainty", "Alternate energy"]',
                   'assert': isLst},
        'convergence_checker': {'default': 'None', 'assert': isLst},
        'phase': {'default': 'None', 'assert': 'True'},
        'gptol': {'default': '0.1', 'assert': 'True'},
        'maxtrial': {'default': '50', 'assert': 'True'},
        'cov_max_tol': {'default': '0.05', 'assert': 'True'},
        'E_max_tol': {'default': '0.05', 'assert': 'True'},
        'distance_tol': {'default': '0.05', 'assert': 'True'},
        'last_checker': {'default': "'Uncertain or Maximum energy'",
                         'assert': 'True'}
    }

    display_map_parameters = OrderedDict({})
    display_graph_parameters = OrderedDict({})
    display_graph_title_parameters = OrderedDict({
        '_cov_max': {
            'label': r'$\Sigma^{{(max)}}_{{95\%}}$', 'isLaTex': True,
            'under_the_condition':
                # "True",
                "{pf:s}.__dict__.get('_maximum_uncertainty_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        '_muErr': {
            'label': r'$\mu_{{err}}^{{(max)}}$', 'isLaTex': True,
            'under_the_condition':
                "{pf:s}.__dict__.get('_maximum_energy_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        '_mu_Et': {
            'label': r'$\mu^{{(max)}}-E_{{t}}$', 'isLaTex': True,
            'under_the_condition':
                "{pf:s}.__dict__.get('_target_energy_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        'deltaMu': {
            'label': r'$\left|\Delta\mu^{{(max)}}\right|$', 'isLaTex': True,
            'under_the_condition':
                "'maximum mu' == {pf:s}.Phase.lower() and "
                "len({pf:s}.__dict__.get('_Emaxlst', [])) > 1",
            'unit': "{p:s}.model.potential_unit",
            'value': "np.abs({pf:s}._Emaxlst[-1] - {pf:s}._Emaxlst[-2])",
            'kwargs': "{'fontsize': 13}"
        },
    })

    def __init__(self, real_finder=None, log=None, gptol=None, cov_max_tol=None,
                 E_max_tol=None, maxtrial=None, phase=0, phases=None,
                 last_checker=None, distance_tol=None,
                 _pbs_walltime="walltime=48:00:00", **kwargs):
        self.real_finder = real_finder
        self.finder_parameters.update(self.real_finder.finder_parameters)
        self.display_map_parameters.update(
            self.real_finder.display_map_parameters)
        self.display_graph_parameters.update(
            self.real_finder.display_graph_parameters)
        self.display_graph_title_parameters.update(
            self.real_finder.display_graph_title_parameters)
        self.log = log
        self.gptol = gptol
        self.cov_max_tol = cov_max_tol
        self.E_max_tol = E_max_tol
        self.distance_tol = distance_tol
        self.maxtrial = maxtrial
        self.phase = phase
        self.phases = phases
        self.last_checker = last_checker
        self._E = []
        self._cov = []
        self._pbs = False
        self._write_pbs = True
        self._write_pbs_only = False
        self._pbs_walltime = _pbs_walltime
        super().__init__(**kwargs)
        #for key, value in kwargs.items():
        #    setattr(self, key, value)

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        elif key in ['real_finder']:
            if value is None:
                value = eval(self.finder_parameters['real_finder']['default'])
            if isinstance(value, str):
                from_ = 'taps.pathfinder'
                module = __import__(from_, {}, None, [value])
                value = getattr(module, value)()
            super().__setattr__(key, value)
        elif key in self.real_finder.finder_parameters:
            default = self.finder_parameters[key]['default']
            assertion = self.finder_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super(PathFinder, self.real_finder).__setattr__(key, value)
            super().__setattr__(key, value)
        elif key in self.finder_parameters:
            default = self.finder_parameters[key]['default']
            assertion = self.finder_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in self.__dict__.get('real_finder', {}).finder_parameters.keys():
            return getattr(self.__dict__['real_finder'], key)
        elif key == 'convergence_checker':
            return self.__dict__['phases']
        else:
            super().__getattribute__(key)

    @property
    def Phase(self):
        if self.phase >= len(self.convergence_checker):
            return self.last_checker
        return self.convergence_checker[self.phase]

    @property
    def Phases(self):
        return self.convergence_checker

    def maximum_uncertainty(self, paths, gptol=None, iter=None):
        cov = 1.96 * np.sqrt(np.diag(paths.model.get_covariance(paths)))
        self._maximum_uncertainty_checked = True
        self._cov_max = cov.max() / paths.model.hyperparameters['sigma_f']
        return np.argmax(cov)

    def check_maximum_uncertainty_convergence(self, paths, idx=None, **kwargs):
        if not self._maximum_uncertainty_checked:
            paths.add_data(index=idx)
            return 0
        paths.add_data(index=idx)
        if self._cov_max < self.cov_max_tol:
            return 1
        return 0

    def maximum_energy(self, paths, gptol=None, iter=None):
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_maximum_energy_convergence(self, paths, idx=None, **kwargs):
        if not self._maximum_energy_checked:
            paths.add_data(index=idx)
            return 0
        data_ids = paths.add_data(index=idx)
        imgdata = paths.get_data(data_ids=data_ids)
        self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        if self._muErr < self.E_max_tol:
            return 1
        return 0

    def uncertain_or_maximum_energy(self, paths, iter=None, **kwargs):
        cov = paths.get_covariance(index=np.s_[:])
        self._maximum_uncertainty_checked = True
        self._cov_max = cov.max() / paths.model.hyperparameters['sigma_f']
        if self._cov_max > self.cov_max_tol:
            return np.argmax(cov)
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_uncertain_or_maximum_energy_convergence(self, paths, idx=None,
                                                      **kwargs):
        if self._cov_max > self.cov_max_tol:
            data_ids = paths.add_data(index=idx)
            return 0
        data_ids = paths.add_data(index=idx)
        imgdata = paths.get_data(data_ids=data_ids)
        self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        if self._muErr < self.E_max_tol:
            return 1
        return 0

    def manual_et(self, paths, **kwargs):
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_manual_et_convergence(self, paths, idx=None, **kwargs):
        return self.check_maximum_energy_convergence(paths, idx=idx, **kwargs)

    def auto_et(self, paths, **kwargs):
        self.Et = self.get_next_et(paths)
        self.Et_type = 'manual'
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_auto_et_convergence(self, paths, idx=None, **kwargs):
        V = paths.get_potential_energy(index=np.s_[1:-1])
        data_ids = paths.add_data(index=idx)
        imgdata = paths.get_data(data_ids=data_ids)
        if self._maximum_energy_checked:
            self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        if np.abs(V.max() - self.Et) < self.Et_opt_tol:
            return 1
        return 0

    def maximum_mu(self, paths, **kwargs):
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_maximum_mu_convergence(self, paths, idx=None, **kwargs):
        E = paths.get_potential_energy(index=np.s_[:])
        self._Emaxlst = self.__dict__.get('_Emaxlst', [])
        self._Emaxlst.append(np.max(E))
        if len(self._Emaxlst) < 3:
            paths.add_data(index=idx)
            return 0
        paths.add_data(index=idx)
        V0, V1 = self._Emaxlst[-2:]
        if np.abs(V0 - V1) < self.Et_opt_tol:
            return 1
        return 0

    def get_next_et(self, paths, **kwargs):
        V = paths.get_potential_energy(index=np.s_[1:-1])
        self._target_energy_checked = True
        self._mu_Et = V.max() - self.Et

        # self._Kinetic = self.__dict__.get('_Kinetic', 0.1)
        # Et too low
        # Et = (np.max(V) + self.Et - self.Et_opt_tol) / 2
        # Et = np.max(V)
        # Et = (np.max(V) + self.Et) / 2
        Et = (np.max(V) + self.Et - self.Et_opt_tol / 2) / 2
        return Et

    def alternate_energy(self, paths, gptol=None, iter=None):
        if iter % 2 == 0:
            return self.maximum_uncertainty(paths, gptol=gptol)
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_alternate_energy_convergence(self, paths, idx=None, **kwargs):
        if kwargs['iter'] % 2 == 0:
            return 0
        return self.check_maximum_energy_convergence(paths, idx=idx, **kwargs)

    def alternate_saddle(self, paths, iter=None):
        if iter % 2 == 0:
            return self.maximum_uncertainty(paths)
        E = paths.get_potential_energy(index=np.s_[:])
        X = paths.get_displacements(index=np.s_[:])
        dE = np.diff(E)
        dX = np.diif(X)
        ddEdX2 = np.diff(dE / dX) / dX[:-1]
        saddle = np.arange(1, paths.P - 1)[np.abs(ddEdX2) < self.gptol]
        if len(saddle) == 0:
            return np.argmax(E)
        return np.random.choice(saddle)

    def check_alternate_saddle_convergence(self, paths, idx=None, **kwargs):
        if kwargs['iter'] % 2 == 0:
            return False
        E = paths.get_potential_energy(index=np.s_[:])
        data_ids = paths.add_data(index=idx)
        imgdata = paths.get_data(data_ids=data_ids)
        if np.abs(E[idx] - imgdata['V'][-1]) < self.gptol:
            return 1
        return 0

    def acquisition(self, paths, phase=None, iter=0):
        if phase is None:
            phase = self.phase
        self._maximum_uncertainty_checked = False
        self._maximum_energy_checked = False
        self._target_energy_checked = False
        phase = self.Phase.lower().replace(" ", "_")
        return getattr(self, phase)(paths, iter=iter)

    def check_convergence(self, paths, phase=None, iter=None, logfile=None,
                          **kwargs):
        if phase is None:
            phase = self.phase
        imgdata = paths.get_data()
        if len(imgdata['V']) < 3:
            paths.add_data(index=paths.P // 3)
            logfile.write("Initial index : %d \n" % (paths.P // 3))
            return False

        idx = self.acquisition(paths, phase=phase, iter=iter)
        logfile.write("Add new idx  : %d \n" % idx)
        logfile.flush()

        phase_name = self.Phase.lower().replace(" ", "_")
        phase = 'check_' + phase_name + '_convergence'
        self.phase += getattr(self, phase)(paths, idx=idx, iter=iter,
                                           logfile=logfile, **kwargs)
        logfile.write("Energy added : %.4f\n" % imgdata['V'][-1])
        logfile.flush()
        if self.phase >= len(self.Phases):
            logfile.write("Last phase. Checking dS...")
            if paths.real_finder.isConverged(paths):
                cov_max = np.max(paths.get_covariance())
                logfile.write(" converged, checking Cov max %f" % cov_max)
                if cov_max < self.cov_max_tol:
                    logfile.write("..Converged!")
                    return True
                logfile.write(".. Too big\n")
                # dist = frechet_distance(self._prevcoords, paths)
                # logfile.write(" converged, checking displacement %f" % dist)
                # if dist < self.distance_tol:
                #     logfile.write('Distance %f < tol, Converged!' % dist)
                #     return True
                # self._prevcoords = paths.copy()
                return False
            logfile.write("NOT converged\n")
        # self._prevcoords = paths.copy()
        return False

    def I_prepared_my_paths_in_various_ways(self, paths, **kwargs):
        if 'auto et' == self.Phase.lower():
            pass
        # if True:
            # logfile = kwargs.get('logfile')
            # logfile.write('Prepare my paths in various ways \n')
            # paths.fluctuate(initialize=True)

    def optimize(self, paths, gptol=0.01, maxiter=50, **search_kwargs):
        label = getattr(self, 'label', None) or paths.label
        log = self.log or label + '.log'
        logdir = os.path.dirname(log)
        if logdir == '':
            logdir = '.'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        logfile = open(log, 'a+')
        dir = os.path.dirname(log)
        if dir == '':
            dir = '.'
        if not os.path.exists(dir):
            os.makedirs(dir)
        if os.path.exists(label + '_pathsdata.db'):
            logfile.write("\nReading %s \n" % (label + '_pathsdata.db'))
            pathsdata = PathsData(label + '_pathsdata.db')
            query = "rowid DESC LIMIT 1;"
            where = " ORDER BY "
            columns = ['rowid', 'paths']
            data = pathsdata.read(query=query, where=where, columns=columns)[0]
            paths = data['paths']
            iter_number = data['rowid'] + 1
        else:
            filename = label + '_initial'
            logfile.write("Writting %s \n" % (label + '_pathsdata.db'))
            self._save(paths, filename=filename)
            iter_number = 1

        logfile.flush()
        i = iter_number
        while not self.check_convergence(paths, iter=i, logfile=logfile):
            dat = [str(d) for d in paths.model.data_ids['image']]
            logfile.write("Iteration    : %d\n" % i)
            logfile.write("Phase        : %s\n" % self.Phase)
            logfile.write("Number of Dat: %d\n" % len(dat))
            logfile.write("ImgData idx  : %s\n" % ', '.join(dat))
            filename = label + '_{i:02d}'.format(i=i)
            self.I_prepared_my_paths_in_various_ways(paths, logfile=logfile)
            paths.search(real_finder=True, logfile=logfile, **search_kwargs)
            self.results.update(paths.real_finder.results)
            self._save(paths, filename=filename)
            i += 1
            if self.maxtrial < i:
                logfile.write("Max iteration, %d, reached! \n" % self.maxtrial)
                break

        logfile.close()
        return paths

    def _save(self, paths, filename=None):
        label = getattr(self, 'label', None) or paths.label
        paths.plot(filename=filename, savefig=True, gaussian=True)
        pathsdata = PathsData(label + '_pathsdata.db')
        data = [{'paths': paths}]
        pathsdata.write(data=data)
