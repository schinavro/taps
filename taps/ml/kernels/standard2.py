import numpy as np
from numpy import identity as II
from numpy import newaxis as nax
from numpy import vstack

from taps.ml.kernels import Kernel


class Standard(Kernel):
    """
    Standard (squared exponential kernel)
    """

    def __call__(self, Xm=None, Xn=None, hyperparameters=None, noise=False,
                 orig=False, use_potential_data_only=False,
                 potential_only=False, gradient_only=False, hessian_only=False,
                 ):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------

        Xm : array, shape (number of data, D)

        Xn : array, shape (points to estimate, D)

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        http://home.zcu.cz/~jacobnzw/pdf/2016_mlsp_gradients.pdf
        978-1-5090-0746-2/16/$31.00 c 2016 IEE
        """
        ll = hyperparameters.get('l^2')                     # D or 1
        sig_f = hyperparameters.get('sigma_f')              # 1
        noise_f = hyperparameters.get('sigma_n^e', 0)
        if Xn is None:
            Xn = Xm.copy()
        N = Xn.shape[-1]
        M = Xm.shape[-1]
        D = np.prod(Xm.shape[:-1])

        Xmn = Xm[:, nax] - Xn[nax]              # M x N x D
        dists = (Xmn ** 2 / ll).sum(axis=2)     # M x N
        K = sig_f * np.exp(-.5 * dists)         # M x N
        if orig:
            if noise:
                return K + noise_f * II(M)      # M x N
            return K
        # Derivative coefficient
        dc_gd = -Xmn / ll                       # M x N x D
        Kgd = (dc_gd * K[..., nax])             # MxNxD x MxN -> MxNxD
        Kgd = Kgd.reshape(M, N*D)               # MxNxD -> MxND
        if potential_only:
            return vstack([K, Kgd])             # MxN + MxND -> MxN(D+1)
        # DxMxN + 1xNxM -> MxDN
        Kdg = np.hstack(-dc_gd * K[nax, ...])
        # DxMx1xN  * 1xMxDxN  -> D x M x D x N
        # dc_dd_glob = -Xnm[:, :, nax, :] * Xmn[nax, :, :, :] / ll / ll
        dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
        # ∂_mn exp(Xn - Xm)^2
        dc_dd_diag = II(D)[:, nax, :, nax] / ll
        # DxNxDxM - DxNxDxM
        Kdd = (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
        # DN x DM
        Kdd = Kdd.reshape(D * N, D * M)
        if gradient_only:
            # (D+1)M x DN
            return vstack([Kdg, Kdd])
        if hessian_only:
            # DxNxM -> NxDxM
            Xmn = np.swapaxes(Xmn, 0, 1)
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
            return Kext + noise * II((D + 1) * N)
        return Kext
