import numpy as np
import numexpr as ne
from numpy import identity as II
from numpy import newaxis as nax
from numpy import vstack
from taps.coords import Coordinate


class SquaredExponential:
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
    hyperparameters = {'sigma_f': 1, 'l^2': 1, 'sigma_n^e': 0.1,
                       'sigma_n^f': 0.1}

    def __init__(self, hyperparameters=None, key2idx=None):
        self.hyperparameters = hyperparameters or self.hyperparameters
        self.key2idx = key2idx or self.key2idx

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

        if isinstance(Xm, Coordinate):
            Xm = Xm.coords
        if isinstance(Xn, Coordinate):
            Xn = Xn.coords

        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        ll = hyperparameters.get('l^2')
        sig_f = hyperparameters.get('sigma_f')
        if Xn is None:
            Xn = Xm.copy()
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
                return K + noise_f * II(N)    # N x M
            return K
        # Derivative coefficient D x N x M
        dc_gd = -Xnm / ll
        # DxNxM x NxM -> DxNxM -> DNxM
        Kgd = np.vstack(dc_gd * K[nax, ...])
        Kp = vstack([K, Kgd])
        if potential_only:
            return Kp                       # (D+1)xN x M
        # DxNxM * 1xNxM -> NxDM
        Kdg = np.hstack(-dc_gd * K[nax, ...])
        # DxNxM -> NxDxM
        Xmn = np.swapaxes(Xnm, 0, 1)
        # DxNx1xM  * 1xNxDxM  -> D x N x D x M
        # dc_dd_glob = -Xnm[:, :, nax, :] * Xmn[nax, :, :, :] / ll / ll
        dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
        # ∂_mn exp(Xn - Xm)^2
        dc_dd_diag = II(D)[:, nax, :, nax] / ll
        # DxNxDxM - DxNxDxM
        Kdd = (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
        # DN x DM
        Kdd = Kdd.reshape(D * N, D * M)
        dK = vstack([Kdg, Kdd])
        if gradient_only:
            # (D+1)N x DM
            return Kp, dK
        if hessian_only:
            # Delta _ dd
            dnm = np.arange(D)
            # DxNxM * DxNxM -> NxDxDxM
            # dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
            dc_hg_glob = np.empty((D, D, N, M))
            DC1 = -dc_gd[nax, :, :, :]  #
            DC2 = -dc_gd[:, nax, :, :]
            # 1xDxNxM * D'x1xNxM -> DxD'xNxM
            ne.evaluate("DC1 * DC2", out=dc_hg_glob)
            # DxD'xNxM -> D'xDxNxM -> NxDxD'xM
            dc_hg_glob = np.swapaxes(dc_hg_glob, 0, 1).swapaxes(0, 2)
            # DxD'xNxM ->  NxD'xDxM
            # dc_hg_glob = np.swapaxes(dc_hg_glob, 0, 2)

            # DxNxM -> NxDxM -> NxDDxM
            dc_hg_diag = np.zeros((N, D, D, M))
            # ∂_mm K(Xm,Xn) NxDxDxM
            # dc_hg_diag[:, dnm, dnm, :] = -1 / ll[:, 0, 0]
            dc_hg_diag[:, dnm, dnm, :] = -1 / ll
            # NxDxDxM + NxDxDxM -> NxDxDxM
            Khg = np.empty((N, D, D, M))
            ne.evaluate("dc_hg_glob + dc_hg_diag", out=Khg)
            # Bacground term: ∂_mmn K(Xm,Xn) DxNx1x1xM * 1xNxDxDxM -> DxNxDxDxM
            DC_GD = dc_gd[:, :, nax, nax, :]
            KHG1 = Khg[nax, ...]
            dc_hd_back = np.empty((D, N, D, D, M))
            ne.evaluate("DC_GD * KHG1", out=dc_hd_back)
            # dc_hd_back = dc_gd[:, :, nax, nax, :] * Khg[nax, ...]
            # Diagonal term : DxNxDxDxM * 1xNxDxDxM -> DxNxDxDxM
            dc_hd_diag = np.zeros((D, N, D, D, M))
            # Global term :
            dc_hd_glob = np.zeros((D, N, D, D, M))
            DCHDGLOB1 = dc_hd_glob[dnm, :, dnm, ...]
            DCHDGLOB2 = dc_hd_glob[dnm, :, :, dnm, ...]
            XMN = Xmn[nax, :, :, :]
            llll = ll*ll
            XMNLLLL = np.empty(XMN.shape)
            ne.evaluate("XMN / llll", out=XMNLLLL)
            ne.evaluate("DCHDGLOB1 + XMNLLLL",
                        out=dc_hd_glob[dnm, :, dnm, ...])
            ne.evaluate("DCHDGLOB2 + XMNLLLL",
                        out=dc_hd_glob[dnm, :, :, dnm, ...])
            #dc_hd_glob[dnm, :, dnm, ...] += Xmn[nax, :, :, :] / ll / ll
            #dc_hd_glob[dnm, :, :, dnm, ...] += Xmn[nax, :, :, :] / ll / ll
            # print(dc_hd_glob[0, 0, 0, :, 0])
            # print(dc_hd_glob.reshape(2, N, 2))
            Khd = np.empty((D, N, D, D, M))
            ne.evaluate("(dc_hd_glob + dc_hd_diag) + dc_hd_back", out=Khd)
            # NxDxDxM x Nx1x1xM -> NxDxDxM
            K1 = K[:, nax, nax, :]
            ne.evaluate("Khg * K1", out=Khg)
            #Khg *= K[:, nax, nax, :]
            # DxNxDxDxM * 1xNx1x1xM
            K2 = K[nax, :, nax, nax, :]
            ne.evaluate("Khd * K2", out=Khd)
            # Khd *= K[nax, :, nax, nax, :]
            # NxDDxM -> N x DDM
            Khg = Khg.reshape(N, D * D * M)
            # DxNxDDxM -> DN x DxDxM
            Khd = Khd.reshape(D * N, D * D * M)
            # print(Khd.shape)
            return Kp, dK, np.vstack([Khg, Khd])  # (D+1)N x DDM

        Kext = np.block([[K, Kdg],
                        [Kgd, Kdd]])  # (D+1)N x (D+1)M
        if noise:
            noise_f = hyperparameters.get('sigma_n^e', 0)
            noise_df = hyperparameters.get('sigma_n^f', 0)
            noise = np.array([noise_f] * N + [noise_df] * D * N)
            I1 = II((D + 1) * N)
            return ne.evaluate("Kext + noise * I1")
        return Kext

    def set_hyperparameters(self, hyperparameters=None):
        for key, idx in self.key2idx.items():
            self.hyperparameters[key] = hyperparameters[idx]

    def get_hyperparameters(self):
        hyperparameters = [None] * len(self.hyperparameters)
        for key, idx in self.key2idx.items():
            hyperparameters[idx] = self.hyperparameters[key]
        return hyperparameters

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        # [(k, v) for k,v in self.hyperparameters.items()]
        return ''
