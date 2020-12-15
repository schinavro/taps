from taps.pathfinder import PathFinder
from taps.utils.shortcut import isbool, isstr, isflt, issclr
from collections import OrderedDict
import numpy as np
import sys
from scipy.fftpack import dst
from scipy.optimize import minimize, check_grad


class DAO(PathFinder):
    """
    Direct action optimizer (DAO)
    """
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
        'prj_search': {'default': 'True', 'assert': isbool},
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
                 gam=None, sin_search=None, prj_search=None, use_grad=None,
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
        self.prj_search = prj_search
        self.use_grad = use_grad
        self.method = method
        self.maxiter = maxiter
        self.Et_opt_tol = Et_opt_tol

        super().__init__(**kwargs)

    def action(self, paths, *args, action_name=None, Et=None, muE=None,
               sin_search=None, prj_search=None):
        action_name = action_name or self.action_name
        sin_search = sin_search or self.sin_search
        prj_search = prj_search or self.prj_search

        if type(action_name) == str:
            action_name = [action_name]
        if Et is None:
            Et = self.get_target_energy(paths)
        if muE is None:
            muE = self.muE
        muT, muP, muL = self.muT, self.muP, self.muL
        D, Nk, N = paths.D, paths.Nk, paths.N
        shape = self.prj.x(paths.coords)
        dt = paths.coords.epoch / N
        gam = self.gam

        def prj_handler(S):
            def action(coords):
                paths.coords = self.prj.x_inv(coords.reshape(shape))
                return S()
            return action

        def sin_handler(S):
            def action(rcoords):
                paths.coords.rcoords = rcoords.reshape((D, Nk))
                return S()
            return action

        def cartesian_handler(S):
            def action(coords):
                paths.coords[..., 1:-1] = coords.reshape((D, N - 2))
                return S()
            return action

        if sin_search:
            handler = sin_handler
        elif prj_search:
            handler = prj_handler
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
            F = -paths.get_gradients(index=np.s_[:])
            dV = -np.concatenate([F, F[..., -1, np.newaxis]], axis=-1)
            v = paths.get_velocity(index=np.s_[:])
            m = paths.get_effective_mass(index=np.s_[:])
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
                    sin_search=None, prj_search=None):
        if not self.use_grad:
            return None
        action_name = action_name or self.action_name
        if type(action_name) == str:
            action_name = [action_name]
        if Et is None:
            Et = self.get_target_energy(paths)
        if muE is None:
            muE = self.muE
        sin_search = sin_search or self.sin_search
        prj_search = prj_search or self.prj_search

        D, Nk, N = paths.D, paths.Nk, paths.N
        muT, muP, muL = self.muT, self.muP, self.muL
        dt = paths.coords.epoch / N
        shape = self.prj.x(paths.coords)
        # energy_conservation_grad = two_points_e_grad
        gam = self.gam

        def prj_handler(dS):
            def grad_action(coords):
                paths.coords = self.prj.x_inv(coords.reshape(shape))
                ds = self.prj.f(dS(), coords)
                return ds
            return grad_action

        def sin_handler(dS):
            def grad_action(rcoords):
                # self.prj.x_inv(rcoords)
                paths.coords.rcoords = rcoords.reshape((D, Nk))
                return dst(dS(), type=1, norm='ortho')[..., :Nk].flatten()
            return grad_action

        def cartesian_handler(dS):
            def grad_action(rcoords):
                paths.coords[..., 1:-1] = rcoords.reshape((D, N - 2))
                return dS().flatten()
            return grad_action

        if sin_search:
            handler = sin_handler
        elif prj_search:
            handler = prj_handler
        else:
            handler = cartesian_handler

        @handler
        def classic():
            Fp = paths.get_kinetic_energy_gradient(index=np.s_[:])  # D x P
            F = -paths.get_gradients(index=np.s_[:])                # D x P
            dS = (Fp + F) * dt   # grad action
            # action - reaction
            dS[..., 1] -= dS[..., 0]
            dS[..., -2] -= dS[..., -1]
            return dS[..., 1:-1]

        @handler
        def onsager_machlup():
            F = -paths.get_gradients(index=np.s_[:]).reshape((D, N))
            dV = -np.hstack([F[:, 0, np.newaxis], F, F[:, -1, np.newaxis]])
            # dV = -np.hstack([np.zeros((D * M, 1)), F, np.zeros((D * M, 1))])
            H = paths.get_hessian(index=np.s_[:]).reshape((D, D, N))
            a = paths.get_acceleration(index=np.s_[:]).reshape((D, N))
            m = paths.get_effective_mass(index=np.s_[:])
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
            F = -paths.get_gradients(index=np.s_[:])
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
            F = -paths.get_gradients(index=np.s_[1:-1])
            return Fp - F

        local = locals()
        if type(action_name) == str:
            action_name = [action_name]
        dS = [local[act.replace(' ', '_').lower()] for act in action_name]
        return lambda x: np.array([df(x) for df in dS]).sum(axis=0)

    def optimize(self, paths=None, logfile=None, Et=None, Et_type=None,
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
                x0 = paths.coords.rcoords.flatten()
            elif self.prj_search:
                x0 = self.prj.x(paths.coords)
            else:
                x0 = paths.coords[..., 1:-1].flatten()
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            print('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 4:
                break

        # Gradient - Error handling by directly use action
        # cart = False
        while self.results['jac_max'] > self.tol:
            # if self.results['msg'] == 'Warning':
            #     self.sin_search = False
            #     jac = jac
            #     cart = True
            # elif cart:
            #     self.sin_search = True
            #     jac = None
            if self.sin_search:
                x0 = paths.coords.rcoords.flatten()
            elif self.prj_search:
                x0 = self.prj.x(paths.coords)
            else:
                x0 = paths.coords[..., 1:-1].flatten()
            print("jac_max > tol(%.2f); Run without gradient" % self.tol)
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            print('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 3:
                break

        om = 'Onsager Machlup'
        self.results[om] = self.action(paths, action_name=om,
                                       muE=0.)(paths.coords.rcoords)
        if logfile is not None:
            sys.stdout = stdout
            if close_log:
                logfile.close()
        if self.sin_search:
            paths.coords.rcoords = res.x.reshape((paths.D, paths.Nk))
            return paths.copy()
        elif self.prj_search:
            paths.coords = self.prj.x_inv(res.x)
        else:
            p = res.x.reshape((paths.D, paths.N - 2))
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
        # E = paths.imgdata['V'].copy()
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
        elif Et_type in ['min(vi, vf)', 'min']:
            E = paths.get_potential_energy(index=[0, -1])
            Et = np.min(E)
        elif Et_type in ['max', 'max(vi, vf)']:
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
            x0 = paths.coords.rcoords.flatten()
        else:
            x0 = paths.coords.flatten()
        act = self.action(paths)
        jac = self.grad_action(paths)
        print(check_grad(act, jac, x0, **kwargs))
        if self.sin_search:
            paths.coords.rcoords = x0.reshape((paths.D, paths.M, paths.Nk))
        else:
            paths.coords = x0.reshape((paths.D, paths.M, paths.N))

    def isConverged(self, paths):
        # jac = self.grad_action(paths)
        # grad = jac(paths.coords.rcoords.flatten())
        # if np.abs(np.max(grad) / np.sqrt(2 * paths.N)) < self.tol:
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
