import sys
import numpy as np
from numpy import newaxis as nax
from numpy.linalg import norm
from scipy.optimize import minimize, check_grad
from scipy.fftpack import dst, idst
from collections import OrderedDict
from ase.pathway.utils import isbool, isdct, isStr, isstr, isflt, issclr


class PathFinder:
    finder_parameters = {
        'results': {'default': 'None', 'assert': isdct},
        'relaxed': {'default': 'None', 'assert': isbool}
    }

    display_map_parameters = OrderedDict()
    display_graph_parameters = OrderedDict()
    display_graph_title_parameters = OrderedDict()
    results = dict({})
    relaxed = False

    def __init__(self):
        pass

    def __setattr__(self, key, value):
        if key in self.finder_parameters:
            default = self.finder_parameters[key]['default']
            assertion = self.finder_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        else:
            raise AttributeError('key `%s`not exist!' % key)

    def __getattr__(self, key):
        if key == 'real_finder':
            return self
        else:
            raise AttributeError('key `%s`not exist!' % key)

    def isConverged(self, *args, **kwargs):
        return True


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

        r = np.concatenate([r[..., 0, nax], r, r[..., -1, nax]], axis=2)
        # r1 = r[:, :, 2:]
        # r0 = r[:, :, 1:-1]
        # r_1 = r[:, :, :-2]
        # r = np.concatenate([r, r[..., -1, nax]], axis=2)
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
        P, D = self.paths.P, self.paths.D
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
        P, D = self.paths.P, self.paths.D
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

    def search(self, paths=None, dt=None, method='velocity-verlet', k=None,
               iter=None, l0=None, shaper=None, **kwargs):
        """
        Should return full paths
        """
        D, M, P = paths.DMP
        if M * D == 1:
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


class ADMD(PathFinder):
    finder_parameters = {
        'action_name': {'default': "'Onsager Machlup'", 'assert': 'True'},
        'muE': {'default': '0.', 'assert': 'np.isscalar({name:s})'},
        'muT': {'default': '0.', 'assert': 'np.isscalar({name:s})'},
        'muP': {'default': '0.', 'assert': 'np.isscalar({name:s})'},
        'muL': {'default': '0.', 'assert': 'np.isscalar({name:s})'},
        'tol': {'default': '0.1', 'assert': '{name:s} > 0'},
        'Et_opt_tol': {'default': '0.05', 'assert': isflt},
        'gam': {'default': '1', 'assert': '{name:s} > 0'},
        'method': {'default': "'BFGS'", 'assert': 'isinstance({name:s}, str)'},
        'maxiter': {'default': '500', 'assert': '{name:s} > 0'},
        'disp': {'default': 'False', 'assert': 'isinstance({name:s}, bool)'},
        'eps': {'default': '1e-4', 'assert': '{name:s} > 0'},
        'T': {'default': '300', 'assert': '{name:s} > 0.'},
        'sin_search': {'default': 'True', 'assert': isbool},
        'use_grad': {'default': 'True', 'assert': isbool},
        'Et_type': {'default': 'None', 'assert': isstr},
        'Et': {'default': 'None', 'assert': issclr},
        'dEt': {'default': 'None', 'assert': issclr},
        'dH': {'default': 'None', 'assert': issclr},
        'res': {'default': 'None', 'assert': 'True'}
    }
    display_map_parameters = OrderedDict({
        'results': {
            'label': r'$S_{OM}$', 'force_LaTex': True,
            'value': "{pf:s}.results.get('Onsager Machlup', 0.)",
            'significant_digit': 3
        },
        'Et': {
            'under_the_condition':
                '{pf:s}.real_finder.__dict__.get("{key:s}") is not None',
            'label': r'$E_{t}$',
            'value': "{pf:s}.real_finder.Et",
            'unit': "{p:s}.model.potential_unit"
        },
    })
    display_graph_parameters = OrderedDict({
        'Et': {
            'isLabeld': True, 'plot': 'plot', 'lighten_color_amount': 0.6,
            'under_the_condition': "True",
            'args': "(np.linspace(*{x:s}[[0, -1]], 11), [{pf:s}.{key:s}] * 11)",
            'kwargs': "{{'label': r'$E_{{t}}$', 'marker': 'D'"
                      ", 'linestyle': '--', 'markersize': 3}}"},
                      # (0, (0, 5, 1, 5))
    })
    display_graph_title_parameters = OrderedDict({
        'Et': {
            'label': r'$E_{t}$', 'isLaTex': True,
            'under_the_condition': '{pf:s}.real_finder.__dict__.get("{key:s}")'
                                   ' is not None',
            'value': "{pf:s}.Et",
            'unit': "{p:s}.model.potential_unit",
            'kwargs': "{'fontsize': 13}"
        }
    })

    Et_type = "manual"
    Et = 0.
    dEt = 0.
    dH = 0.

    def __init__(self, action_name=None, muE=None, muT=None, muP=None, muL=None,
                 T=None, tol=None, method=None, eps=None, disp=None,
                 gam=None, sin_search=None, use_grad=None,
                 maxiter=None, Et_opt_tol=None, **kwargs):
        super().finder_parameters.update(self.finder_parameters)
        self.finder_parameters.update(super().finder_parameters)
        self.action_name = action_name
        self.muE = muE
        self.muT = muT
        self.muP = muP
        self.muL = muL
        self.tol = tol
        self.eps = eps
        self.disp = disp
        self.gam = gam
        self.sin_search = sin_search
        self.use_grad = use_grad
        self.method = method
        self.maxiter = maxiter
        self.Et_opt_tol = Et_opt_tol

        for key, value in kwargs.items():
            setattr(self, key, value)

    def action(self, paths, *args, action_name=None, Et=None, muE=None,
               sin_search=None):
        action_name = action_name or self.action_name
        if type(action_name) == str:
            action_name = [action_name]
        if Et is None:
            Et = self.get_target_energy(paths)
        if muE is None:
            muE = self.muE
        if sin_search is None:
            sin_search = self.sin_search
        muT, muP, muL = self.muT, self.muP, self.muL
        D, M, Pk, P = paths.D, paths.M, paths.Pk, paths.P
        dt = paths.prj.dt
        gam = self.gam

        def sin_handler(S):
            def action(rcoords):
                paths.rcoords = rcoords.reshape((D, M, Pk))
                return S()
            return action

        def cartesian_handler(S):
            def action(rcoords):
                paths.coords[..., 1:-1] = rcoords.reshape((D, M, P - 2))
                return S()
            return action

        if sin_search:
            handler = sin_handler
        else:
            handler = cartesian_handler

        @handler
        def classic():
            K = paths.get_kinetic_energy(index=np.s_[:])      # P
            V = paths.get_potential_energy(index=np.s_[:])    # P
            L = K - V                                         # P
            S = L.sum(axis=0) * dt                            # 0
            return S

        @handler
        def onsager_machlup():
            """
            2qj 2qj21 2qj11 2D2
            """
            # Vi, Vf = paths.get_potential_energy(index=[0, -1])
            F = paths.get_forces(index=np.s_[:])
            dV = -np.concatenate([F, F[..., -1, np.newaxis]], axis=2)
            v = paths.get_velocity(np.s_[:])
            m = paths.get_effective_mass(np.s_[:])
            _gam = gam * m
            ldVl2 = (dV * dV).sum(axis=0)
            # DxMx(P-1) -> P
            Som = (_gam * (v * v)
                   + (ldVl2[..., 1:] + ldVl2[..., :-1]) / 2 / _gam
                   - v * (dV[..., 1:] - dV[..., :-1])).sum() * dt / 4
            # Som += (Vf - Vi) / 2
            # Som = Som.sum()
            # Som *= 0.25 * dt
            return Som

        @handler
        def energy_conservation():
            H = paths.get_total_energy(index=np.s_[:])
            return muE * ((H - Et) ** 2).sum()

        @handler
        def hamiltonian():
            return paths.get_total_energy(index=np.s_[:]).sum()

        local = locals()
        S = [local[act.replace(' ', '_').lower()] for act in action_name]
        return lambda x: sum([f(x) for f in S])

    def grad_action(self, paths, *args, action_name=None, Et=None, muE=None,
                    sin_search=None):
        if not self.use_grad:
            return None
        action_name = action_name or self.action_name
        if type(action_name) == str:
            action_name = [action_name]
        if Et is None:
            Et = self.get_target_energy(paths)
        if muE is None:
            muE = self.muE
        if sin_search is None:
            sin_search = self.sin_search

        D, M, Pk, P = paths.D, paths.M, paths.Pk, paths.P
        muT, muP, muL = self.muT, self.muP, self.muL
        dt = paths.prj.dt
        # energy_conservation_grad = two_points_e_grad
        gam = self.gam

        def sin_handler(dS):
            def grad_action(rcoords):
                paths.rcoords = rcoords.reshape((D, M, Pk))
                return dst(dS(), type=1, norm='ortho')[..., :Pk].flatten()
            return grad_action

        def cartesian_handler(dS):
            def grad_action(rcoords):
                paths.coords[..., 1:-1] = rcoords.reshape((D, M, P - 2))
                return dS().flatten()
            return grad_action

        if sin_search:
            handler = sin_handler
        else:
            handler = cartesian_handler

        @handler
        def classic():
            Fp = paths.get_kinetic_energy_gradient(np.s_[:])  # D x M x P
            F = paths.get_forces(np.s_[:])                    # D x M x P
            dS = (Fp + F) * dt   # grad action
            # action - reaction
            dS[..., 1] -= dS[..., 0]
            dS[..., -2] -= dS[..., -1]
            return dS[..., 1:-1]

        @handler
        def onsager_machlup():
            F = paths.get_forces(index=np.s_[:]).reshape((D * M, P))
            dV = -np.hstack([F[:, 0, np.newaxis], F, F[:, -1, np.newaxis]])
            # dV = -np.hstack([np.zeros((D * M, 1)), F, np.zeros((D * M, 1))])
            H = paths.get_hessian(index=np.s_[:]).reshape((D * M, D * M, P))
            a = paths.get_acceleration(index=np.s_[:]).reshape((D * M, P))
            m = paths.get_effective_mass(index=np.s_[:]).reshape((M, -1))
            _gam = gam * m

            notation = 'ij...,i...->j...'
            dS = (0.5 * _gam * a * dt) \
                - 0.25 * (2 * dV[..., 1:-1] - dV[..., 2:] - dV[..., :-2]) \
                + np.einsum(notation, H,
                            0.5 * dt * dV[..., 1:-1] / _gam
                            - 0.25 * dt * dt * a)
            # action - reaction
            dS[..., 1] -= dS[..., 0]
            dS[..., -2] -= dS[..., -1]
            return dS[..., 1:-1]

        @handler
        def energy_conservation():
            F = paths.get_forces(index=np.s_[:])
            p = paths.get_momentum(index=np.s_[:])                 # D x M x P
            K = paths.get_kinetic_energy(index=np.s_[:])           # P
            V = paths.get_potential_energy(index=np.s_[:])         # P
            H = K + V                                        # P
            # D x M x (P - 1)
            dS = ((H[:-2] - Et) * p[..., :-2] / dt
                  - ((H[1:-1] - Et) * (p[..., 1:-1] / dt + F[..., 1:-1])))
            # action - reaction
            dS[..., 0] -= -(H[0] - Et) * (p[..., 0] / dt + F[..., 0])
            return 2 * muE * dS

        @handler
        def hamiltonian():
            Fp = paths.get_kinetic_energy_gradient(index=np.s_[1:-1])
            F = paths.get_forces(index=np.s_[1:-1])
            return Fp - F

        local = locals()
        if type(action_name) == str:
            action_name = [action_name]
        dS = [local[act.replace(' ', '_').lower()] for act in action_name]
        return lambda x: np.array([df(x) for df in dS]).sum(axis=0)

    def search(self, paths=None, logfile=None, Et=None, Et_type=None,
               action_name=None, **args):
        if logfile is not None:
            stdout = sys.stdout
            if isinstance(logfile, str):
                close_log = True
                logfile = open(logfile, 'a')
                sys.stdout = logfile
            else:
                close_log = False
                sys.stdout = logfile
        action_name = action_name or self.action_name
        if type(action_name) == str:
            action_name = [action_name]

        print('Action name  : ', ' + '.join(action_name))
        Et = self.get_target_energy(paths, Et=Et, Et_type=Et_type)
        action_name_lower = [action.lower() for action in action_name]
        if 'energy conservation' in action_name_lower:
            print('Target energy: ', Et)
            print('Target type  : ', self.Et_type)
            print('muE          : ', self.muE)
        if 'onsager machlup' in action_name_lower:
            print('gamma        : ', self.gam)

        act = self.action(paths, Et=Et)
        jac = self.grad_action(paths, Et=Et)
        search_kwargs = self.get_search_kwargs(**args)
        print('            Iter   nfev   njev        S   dS_max')
        while True:
            # print(res.message)
            if self.sin_search:
                x0 = paths.rcoords.flatten()
            else:
                x0 = paths.coords.flatten()
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            print('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 4:
                break

        # Gradient - Error handling by directly use action
        while self.results['jac_max'] > self.tol:
            if self.sin_search:
                x0 = paths.rcoords.flatten()
            else:
                x0 = paths.coords.flatten()
            print("Gradient Error above tolerence! Emergency mode; Run without gradient")
            res = minimize(act, x0, jac=None, **search_kwargs)
            self.set_results(res)
            print('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 3:
                break

        om = 'Onsager Machlup'
        self.results[om] = self.action(paths, action_name=om,
                                       muE=0.)(paths.rcoords)
        if logfile is not None:
            sys.stdout = stdout
            if close_log:
                logfile.close()
        if self.sin_search:
            paths.rcoords = res.x.reshape((paths.D, paths.M, paths.Pk))
            return paths.copy()
        else:
            p = res.x.reshape((paths.D, paths.M, paths.P - 2))
            paths.coords[..., 1:-1] = p
            return paths.copy()

    def set_results(self, res, mode=None):
        if mode is None:
            if self.results.get('nfev') is not None:
                mode = 'a'
            else:
                mode = 'w'
        self.results['fun'] = res.fun
        self.results['jac_max'] = res.jac.max()
        self.results['message'] = res.message
        self.results['success'] = res.success
        self.results['status'] = res.status
        msg_lower = res.message.lower()
        if 'desired' in msg_lower:
            self.results['msg'] = 'Warning'
        elif 'maximum' in msg_lower:
            self.results['msg'] = 'Max out'
        elif 'successfully' in msg_lower:
            self.results['msg'] = 'Success'
        if mode == 'w':
            self.results['nfev'] = res.nfev
            self.results['nit'] = res.nit
            self.results['njev'] = res.njev
        elif mode == 'a':
            self.results['nfev'] += res.nfev
            self.results['nit'] += res.nit
            self.results['njev'] += res.njev

    def get_target_energy(self, paths, Et_type=None, Et=None):
        if Et_type is None:
            Et_type = self.Et_type
        if Et is None:
            Et = self.Et
        # E = paths.atomsdata['V'].copy()
        # E = paths.get_potential_energy(index=np.s_[:])
        # E = paths.plotter.__dict__.get('_ave_map')
        # if E is None:
        # coords = paths.plotter.get_meshgrid(grid_type='paths')
        # E = paths.model.get_potential_energy(paths, coords=coords)
        if Et_type == 'manual':
            pass
        elif Et_type == 'vi':
            Et = paths.get_potential_energy(index=[0])
        elif Et_type == 'vf':
            Et = paths.get_potential_energy(index=[-1])
        elif Et_type == 'min(vi, vf)':
            E = paths.get_potential_energy(index=[0, -1])
            Et = np.min(E)
        elif Et_type == 'max(vi, vf)':
            E = paths.get_potential_energy(index=[0, -1])
            Et = np.max(E)
        elif Et_type == 'average':
            coords = paths.plotter.get_meshgrid(grid_type='coords')
            E = paths.model.get_potential_energy(paths, coords=coords)
            Et = np.average(E)
        elif Et_type == 'adjust':
            V = paths.get_potential_energy()
            T = paths.get_kinetic_energy_energy()
            H = V + T
            H_max = H.max()
            H_min = H.min()
            if H_max - H_min < 0.1:
                Et = H_min / 2
            elif H_max - H_min > 0.5:
                Et = V.max()
        elif Et_type == 'min':
            coords = paths.plotter.get_meshgrid(grid_type='coords')
            E = paths.model.get_potential_energy(paths, coords=coords)
            Et = np.min(E)
        elif Et_type == 'max':
            coords = paths.plotter.get_meshgrid(grid_type='coords')
            E = paths.model.get_potential_energy(paths, coords=coords)
            Et = np.max(E)
        elif Et_type in ['var', 'convex', 'drag']:
            dt = paths.prj.dt
            E = paths.get_potential_energy(index=np.s_[:])
            dE = np.diff(E)
            critical_points = np.abs(dE) < 1.e-2 * dt
            ddE = np.diff(dE)
            convex = critical_points[:-1] & (ddE > 1e-3 * dt * dt)
            if np.any(convex):
                maxE = np.max(E)
                minE = np.min(E)
                cenE = 0.5 * (maxE + minE)
                convE = np.min(E[:-2][convex])
                if convE > cenE:
                    self.Et_type = 'convex'
                    return convE
                else:
                    self.Et_type = 'drag'
                    return minE + (maxE - minE) * 0.8
            self.Et_type = 'var'
            Et = np.max(E)
        else:
            raise NotImplementedError('No %s Et type' % Et_type)
        self.Et = Et
        return Et

    def get_search_kwargs(self, **args):
        method = args.get('method')
        tol = args.get('tol')
        disp = args.get('disp')
        maxiter = args.get('maxiter')
        eps = args.get('eps')
        if method is None:
            method = self.method
        if tol is None:
            tol = self.tol
        if disp is None:
            disp = self.disp
        if maxiter is None:
            maxiter = self.maxiter
        if eps is None:
            eps = self.eps

        search_kwargs = {'method': method, 'tol': tol,
                         'options': {'disp': disp,
                                     'maxiter': maxiter,
                                     'eps': eps}}
        # search_kwargs.update(args)
        return search_kwargs

    def check_grad(self, paths=None, **kwargs):
        if len(paths.get_data()['V']) < 2:
            print('Need to set up data first')
            return
        if self.sin_search:
            x0 = paths.rcoords.flatten()
        else:
            x0 = paths.coords.flatten()
        act = self.action(paths)
        jac = self.grad_action(paths)
        print(check_grad(act, jac, x0, **kwargs))
        if self.sin_search:
            paths.rcoords = x0.reshape((paths.D, paths.M, paths.Pk))
        else:
            paths.coords = x0.reshape((paths.D, paths.M, paths.P))

    def isConverged(self, paths):
        # jac = self.grad_action(paths)
        # grad = jac(paths.rcoords.flatten())
        # if np.abs(np.max(grad) / np.sqrt(2 * paths.P)) < self.tol:
        #     return True
        # return False
        if self.results.get('jac_max') is None:
            return False
        return self.results.get('jac_max') < self.tol


def two_points_e_grad(p, H, F, Et, muE, dt):
    #return 2 * muE * ((H[:-1] - Et) * m * v[..., :-1] / dt + (
    #                  -(H[1:] - Et) * (m * v[..., 1:] / dt + F)))
    g = np.zeros(F.shape)
    # g[..., 0] = -((H[0] - Et) * (p[..., 0] / dt + F[..., 0]))
    # g[..., 0] = 0
    g[..., 1:-1] = ((H[:-2] - Et) * p[..., :-2] / dt -
                    ((H[1:-1] - Et) * (p[..., 1:-1] / dt + F[..., 1:-1])))
    return 2 * muE * g


def two_points_p_grad(p, H, F, Et, muE, dt):
    return None


def two_points_t_grad(p, H, F, Et, muE, dt):
    return None


def two_points_l_grad(p, H, F, Et, muE, dt):
    return None


def three_points_e_grad(p, H, F, Et, muE, dt):
    return 2 * muE * (-(H[1:-1] - Et) * F +
                      p[..., :-2] * (H[:-2] - Et) / (2 * dt) -
                      p[..., 2:] * (H[2:] - Et) / (2 * dt))
