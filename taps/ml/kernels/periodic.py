import numpy as np
from numpy.linalg import solve
from numpy import identity as II
from numpy import newaxis as nax
from numpy import vstack, cos, sin

from taps.ml.kernels import Kernel


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
        """
        Isotropic sine squared kernel

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
        ll = hyperparameters.get('l^2', 1)
        sig_f = hyperparameters.get('sigma_f', 0)
        if Xn is None:
            Xn = Xm.copy()
        # Xm = atleast_3d(Xm).astype(float)
        # Xn = atleast_3d(Xn).astype(float)
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
                return K + noise_f * II(N)
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
        dc_dd_diag = 2 * pipi * Cos2nm[:, :, nax] * II(D)[:, nax, :, nax] / ll
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
            return Kext + noise * II((D + 1) * N)
        return Kext
