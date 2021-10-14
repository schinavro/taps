import numpy as np
from numpy import identity as I
from numpy import newaxis as nax
from numpy import vstack, atleast_3d


class Kernel:
    """ Function like class that generate kernel matrix.

    Parameters
    ----------
    key2idx: dict
       linking between name of hyperparameters and index of that parameters in
       array that will used in optimization process
    hyperparameters: dict
       Dictionary contains hyperparameters.
       'sigma_f': 0, 'l^2': 1, 'sigma_n^e': 2, 'sigma_n^f': 3

    """
    key2idx = {'sigma_f': 0, 'l^2': 1, 'sigma_n^e': 2, 'sigma_n^f': 3}
    hyperparameters = {'sigma_f': 1, 'l^2': 1, 'sigma_n^e': 0, 'sigma_n^f': 0}

    def __call__(self, Xn=None, Xm=None, orig=False, noise=False,
                 hyperparameters=None, gradient_only=False, hessian_only=False,
                 potential_only=False):
        """ Isotropic squared exponential kernel

        Parameters
        ----------
        Xn : array size of DxN
           Suppose to be a matrix contains points to be estimated.
        Xm : array size of DxM
           Suppose to be a matrix contains data points have been calculated.
        orig: bool
           Whether return non extended kerenl or not
        noise: bool
           wether it will use noise parameter or not
        hyperparameters: dict
           use custom hyperparameters set if given
        gradient_only: bool
           If one want to calculate gradient.
           If it is true, it will return only gradient part of kernel matrix
        hessian_only: bool
           If it is true, it will return the hessian kernel matrix
        potential_only: bool
           If it is true, it will return only potential part of kernel matrix

        Returns
        -------
        orig:
             M x N array
        potential_only:
             DN x M array
        gradient_only:
             DN x DM array
        hessian_only:
             DN x DDM array
        """

        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        ll = hyperparameters.get('l^2')
        sig_f = hyperparameters.get('sigma_f')
        if Xn is None:
            Xn = Xm.copy()
        # Xm = atleast_3d(Xm)
        # Xn = atleast_3d(Xn)
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
            # dc_hg_diag[:, dnm, dnm, :] = -1 / ll[:, 0, 0]
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
        if hyperparameters_list is None:
            return self.hyperparameters
        hyperparameters = {}
        for key, idx in self.key2idx.items():
            hyperparameters[key] = hyperparameters_list[idx]
        return hyperparameters


    def shape_data(self, data, hess=True):
        shape = data['X'].shape
        D, M = np.prod(shape[:-1]), shape[-1]
        if M == 0:
            return np.zeros(0)
        if not hess:
            return data['V']
        return np.vstack([data['V'], -data['F'].reshape(D, M)]).flatten()

    def __repr__(self):
        return self.__class__.__name__


class AtomicDistanceKernel(Kernel):
    def __call__(self, Xm, Xn=None, hyperparameters=None):
        """
        https://link.springer.com/chapter/10.1007/978-3-319-70087-8_55"""
        if Xm is not None:
            pass


class DescriptorKernel(Kernel):
    from taps.ml.descriptor import SphericalHarmonicDescriptor
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
