from taps.pathfinder import PathFinder


class NEB(PathFinder):
    finder_parameters = {
        'k': {
            'default': '1',
            'assert': 'np.isscalar({name:s})'
        },
        'iter': {
            'default': '10',
            'assert': 'isinstance({name:s}, int)'
        },
        'dt': {
            'default': '0.1',
            'assert': '{name:s} > 0'
        },
        'l0': {
            'default': '0.',
            'assert': 'True'
        },
        'shaper': {
            'default': 'None',
            'assert': 'True'
        },
        'gam': {
            'default': '1',
            'assert': 'True'
        }
    }

    def __init__(self, k=None, dt=None, iter=None, l0=None, shaper=None,
                 gam=None):
        """
        k : spring const
        """
        super().finder_parameters.update(self.finder_parameters)
        self.finder_parameters.update(super().finder_parameters)
        self.k = k
        self.dt = dt
        self.iter = iter
        self.l0 = l0
        self.shaper = shaper
        self.gam = gam

    def F_NEB(self, r, F, k=None, l0=None):
        """
        https://en.wikipedia.org/wiki/Energy_minimization#Nudged_elastic_band
        https://theory.cm.utexas.edu/henkelman/pubs/jonsson98_385.pdf

        Return : D x M x P - 2
        """
        if k is None:
            k = self.k
        if l0 is None:
            l0 = self.l0

        r = np.concatenate([r[..., 0, nax], r, r[..., -1, nax]], axis=-1)
        # r1 = r[:, :, 2:]
        # r0 = r[:, :, 1:-1]
        # r_1 = r[:, :, :-2]
        # r = np.concatenate([r, r[..., -1, nax]], axis=-1)
        r1 = r[:, :, 2:]                                       # DxMxP
        r0 = r[:, :, 1:-1]
        r_1 = r[:, :, :-2]
        dr1 = r1 - r0                                          # DxMxP
        dr0 = r0 - r_1                          # DxMxP
        # ddr = dr1 - dr0

        ldr1l = norm(dr1, axis=0)               # MxP
        ldr0l = norm(dr0, axis=0)               # MxP
        # ldr1_1l = norm(r1 - r_1, axis=0)        # MxP
        # Zero divide handler
        ldr1l[ldr1l < 1e-8] = 1e-8
        ldr0l[ldr0l < 1e-8] = 1e-8
        # ldr1_1l[ldr1_1l < 1e-8] = 1e-8

        e_dr1 = dr1 / ldr1l                                     # DxMxP
        e_dr0 = dr0 / ldr0l                     # DxMxP
        # tau = (r1 - r_1) / ldr1_1l              # DxMxP
        tau = e_dr1 + e_dr0                                 # DxMxP
        tau = tau / norm(tau, axis=0)
        self._tau = tau

        dV = -F                                                 # DxMxP
        # f1 = -k * (ldr1l - l0) * e_dr1        # DxMxP
        # f0 = k * (ldr1l - l0) * e_dr1
        # fi = k * ((dr1 - dr0) * tau).sum(axis=0) * tau
        fi = k * (dr1 - dr0)
        # fi = f1 - f0                                            # DxMxP
        # fi = k * ((ldr1l - l0) * e_dr1 + (ldr0l - l0) * e_dr0)
        # D x M x P  -> D x M x P
        dV_parallel = np.sum(dV * tau, axis=0) * tau
        fi_parallel = np.sum(fi * tau, axis=0) * tau
        dV_perpendicular = dV - dV_parallel  # DxMxP
        fi_perpendicular = fi - fi_parallel  # DxMxP

        Fi = -dV_perpendicular + fi_parallel  # DxMxP

        # cos_theta = (e_dr1 * e_dr0).sum(axis=0)
        cos_theta = (e_dr1 * e_dr0).sum(axis=0)
        angle_constraint = (1 + np.cos(np.pi * cos_theta)) / 2
        angle_constraint[cos_theta < 0.] = 1
        # print(norm(angle_constraint * fi_perpendicular, axis=0))
        # print(angle_constraint)
        # print(norm(dV, axis=0) / norm(fi, axis=0))

        F_neb = Fi + angle_constraint * fi_perpendicular
        self._dV = dV
        self._fi = fi
        self._F_neb = F_neb
        # return Fi
        return F_neb[..., 1:-1]

    def stormer_verlet(self, paths):
        """
        set x1 = x0 + v0 * dt + a/2 * ddt
        for n = 1, 2, 3, ... iterate
            x[n+1] = 2 * x[n] - x[n-1] + a[n] * ddt
        """
        P, D = self.paths.N, self.paths.D
        mask = self.paths.mask
        masses = np.outer(self.paths.masses, np.ones(P - 2))
        dt = self.dt
        ddt = dt * dt

        x = self.paths.coords[:D, mask, 1:-1].copy()  # D x M x (P - 2)
        v = self.v         # D x M x (P - 2)

        x1 = (x + v * dt + self.F_NEB(x) / masses / 2 * ddt).copy()
        for i in range(self.iter):
            x2 = 2 * x1 - x + self.F_NEB(x1) * ddt
            x = x1.copy()
            x1 = x2.copy()
        self.v = v
        return x2.copy()

    def velocity_verlet(self, paths):
        P, D = self.paths.N, self.paths.D
        mask = self.paths.mask
        masses = np.outer(self.paths.masses, np.ones(P - 2))
        dt = self.dt
        ddt = dt * dt

        x = paths.coords[:D, mask, 1:-1].copy()  # D x M x (P - 2)
        a = self.F_NEB(x) / masses
        v = self.v
        x1 = x + v * dt + 0.5 * a * ddt
        for i in range(self.iter):
            a1 = self.F_NEB(x1) / masses
            v1 = v + 0.5 * (a + a1) * dt
            x = x1.copy()
            v = v1.copy()
            a = a1.copy()
            x1 = x + v * dt + 0.5 * a * ddt
        self.v = v
        return x1.copy()

    def relax(self, paths, dt=None, iter=None, k=None, l0=None):
        r = paths.coords.copy()
        v = np.zeros(r[..., 1:-1].shape)
        a0 = np.zeros(r[..., 1:-1].shape)
        if dt is None:
            dt = self.dt
        if iter is None:
            iter = self.iter

        for i in range(iter):
            paths.coords = r.copy()
            F = paths.get_forces(index=np.s_[:]).reshape(*self.shaper)
            a = self.F_NEB(r.copy(), F, k=k, l0=l0)
            v = v / 1.2 + (a + a0) / 2 * dt
            r[..., 1:-1] = r[..., 1:-1] + v * dt + 0.5 * a * dt * dt
            a0 = a.copy()

        return r[..., 1:-1]

    def optimize(self, paths=None, dt=None, method='velocity-verlet', k=None,
                 iter=None, l0=None, shaper=None, **kwargs):
        """
        Should return full paths
        """
        D, M, P = paths.DMP
        if D == 1:
            self.shaper = (1, 1, -1)
        elif M * D == 2:
            self.shaper = (2, 1, -1)
        elif paths.M * paths.D == 3:
            self.shaper = (3, 1, -1)
        else:
            self.shaper = paths.coords.shape
        r = paths.coords.reshape(*self.shaper)
        if l0 is None and self.l0 == 0.:
            l0 = norm(r[..., -1] - r[..., 0], axis=0).reshape(M, 1) / P  # M
            l0 == 0.
        elif l0 is None:
            l0 = self.l0
        if dt is None:
            dt = self.dt
        if k is None:
            k = self.k
        if iter is None:
            iter = self.iter
        if method is None:
            method = self.method
        md_method = {'stormer-verlet': self.stormer_verlet,
                     'velocity-verlet': self.velocity_verlet}
        paths.coords[..., 1:-1] = self.relax(paths, dt=dt, iter=iter, k=k, l0=l0)
        return paths.copy()
