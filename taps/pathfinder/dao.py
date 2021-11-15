from taps.pathfinder import PathFinder
from taps.utils.shortcut import isbool, isstr, isflt, issclr
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize, check_grad


class Classic:
    required_static_property_S = ['potential']
    required_static_property_dS = ['gradients']
    required_kinetic_property_S = ['kinetic_energy']
    required_kinetic_property_dS = ['kinetic_energy_gradient']

    def __init__(self, dt=None):
        self.dt = dt

    def S(self, kinetic_energy=None, potential=None, **kwargs):
        # K = paths.get_kinetic_energies(index=np.s_[:])
        # V = paths.get_potential_energy(index=np.s_[:])
        L = kinetic_energy - potential  # K-V
        S = L.sum(axis=0) * self.dt
        return S

    def dS(self, kinetic_energy_graidient=None, gradients=None, **kwargs):
        # Fp = paths.get_kinetic_energy_gradients(index=np.s_[:])
        # F = -paths.get_gradients(index=np.s_[:])
        dS = (kinetic_energy_graidient - gradients) * self.dt   # Fp + F
        # action - reaction
        # dS[..., 1] -= dS[..., 0]
        # dS[..., -2] -= dS[..., -1]
        return dS[..., 1:-1]


class EnergyRestraint:
    def __init__(self, muE=None, Et=None):
        self.muE = muE
        self.Et = None

    def S(self, kinetic_energy=None, potential=None):
        # H = paths.get_total_energy(index=np.s_[:])
        H = kinetic_energy + potential
        return muE * ((H - Et) ** 2).sum()

    def dS(self, potential=None, gradients=None, momentum=None,
           kinetic_energy=None):
        F = -paths.get_gradients(index=np.s_[:])
        p = paths.get_momentums(index=np.s_[:])                 # D x M x N
        K = paths.get_kinetic_energies(index=np.s_[:])           # P
        V = paths.get_potential_energy(index=np.s_[:])         # P
        H = K + V                                        # P
        # D x M x (P - 1)
        dS = ((H[:-2] - Et) * p[..., :-2] / dt
              - ((H[1:-1] - Et) * (p[..., 1:-1] / dt + F[..., 1:-1])))
        # action - reaction
        dS[..., 0] -= -(H[0] - Et) * (p[..., 0] / dt + F[..., 0])
        return 2 * muE * dS


class Hamiltonian:
    required_static_property_S = ['gradients']
    required_static_property_dS = ['gradients', 'hessian']
    required_kinetic_property_S = ['velocity', 'mass']
    required_kinetic_property_dS = ['acceleration', 'mass']

    def S(self, potential=None, kinetic_energy=None):
        return paths.get_total_energy(index=np.s_[:]).sum()

    def dS(self, gradients=None, kinetic_energy_gradient=None):
        Fp = paths.get_kinetic_energy_gradients(index=np.s_[1:-1])
        F = -paths.get_gradients(index=np.s_[1:-1])
        return Fp - F


class OnsagerMachlup:
    required_static_property_S = ['gradients']
    required_static_property_dS = ['gradients', 'hessian']
    required_kinetic_property_S = ['velocity', 'mass']
    required_kinetic_property_dS = ['acceleration', 'mass']
    def __init__(self, gam=None):
        self.gam = gam

    def S(self, F=None, velocity=None, mass=None):
        """
        2qj 2qj21 2qj11 2D2
        """
        # Vi, Vf = paths.get_potential_energy(index=[0, -1])
        # F = -paths.get_gradients(index=np.s_[:]).reshape((D, N))
        dV = -np.concatenate([F, F[..., -1, np.newaxis]], axis=-1)
        # v = paths.get_velocities(index=np.s_[:]).reshape((D, N))
        # m = paths.get_effective_mass(index=np.s_[:])
        _gam = gam * m
        # ldVl2 = (dV * dV).sum(axis=0)
        ldVl2 = dV * dV
        # DxMx(P-1) -> P
        Som = (_gam * (v * v)
               + (ldVl2[..., 1:] + ldVl2[..., :-1]) / 2 / _gam
               - v * (dV[..., 1:] - dV[..., :-1])).sum() * dt / 4
        # Som += (Vf - Vi) / 2
        # Som = Som.sum()
        # Som *= 0.25 * dt
        return Som

    def dS(self, gradients=None, hessian=None, acceleration=None, mass=None):
        # F = -paths.get_gradients(index=np.s_[:]).reshape((D, N))
        dV = -np.hstack([F[:, 0, np.newaxis], F, F[:, -1, np.newaxis]])
        # dV = -np.hstack([np.zeros((D * M, 1)), F, np.zeros((D * M, 1))])
        # H = paths.get_hessian(index=np.s_[:]).reshape((D, D, N))
        # a = paths.get_accelerations(index=np.s_[:]).reshape((D, N))
        # m = paths.get_effective_mass(index=np.s_[:])
        _gam = gam * m

        notation = 'ij...,i...->j...'
        dS = (0.5 * _gam * a * dt) \
            - 0.25 * (2 * dV[..., 1:-1] - dV[..., 2:] - dV[..., :-2]) \
            + np.einsum(notation, H,
                        0.5 * dt * dV[..., 1:-1] / _gam
                        - 0.25 * dt * dt * a)
        # action - reaction
        # dS[..., 1] -= dS[..., 0]
        # dS[..., -2] -= dS[..., -1]
        return dS[..., 1:-1]


class DAO(PathFinder):
    """
    Direct action optimizer (DAO)

    Parameters
    ----------

    use_grad: bool , default True
        Whether to use graidnet form of the action.

    search_kwargs: Dict,
        Optimization keyword for scipy.minimize

    action_kwargs: Dict of dict,
        {'Onsager Machlup': {'gamma': 2},
        'Total Energy restraint': ... }
    """
    
    def __init__(self, action_kwargs=None, search_kwargs=None, use_grad=None,
                 logfile=None, **kwargs):
        self.action_kwargs = action_kwargs
        self.search_kwargs = search_kwargs
        self.use_grad = use_grad

        super().__init__(**kwargs)

    def action(self, paths, action_kwargs=None, prj=None):
        """
        return a function calculating specified action
        """
        action_kwargs = action_kwargs or self.action_kwargs
        prj = prj or self.prj
        rcoords = prj.x(paths.coords).similar()
        dt = paths.coords.dt

        S = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs:
            module = __import__('.actions', {}, None, [name])
            act = getattr(module, name)(paths, **kwargs)
            S.append(act)
            for prop in act.required_static_property_S:
                static_properties.add(prop)
            for prop in act.required_kinetic_property_S:
                kinetic_properties.add(prop)
        static_properties = list(static_properties)
        kinetic_properties = list(kinetic_properties)

        def calculator(_rcoords):
            rcoords.coords = _rcoords
            paths.coords = prj.x_inv(rcoords)

            results = paths.get_properties(properties=static_properties)
            results.update(paths.get_kinetics(properties=kinetic_properties))

            res = 0.
            for s in S:
                res += s.S(**results)
            return res

        return calculator

    def grad_action(self, paths, *args, prj_search=None, action_kwargs=None):
        if not self.use_grad:
            return None

        # Prepare step
        action_kwargs = action_kwargs or self.action_kwargs
        prj = prj or self.prj
        rcoords = prj.x(paths.coords).similar()
        dt = paths.coords.dt

        dS = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs:
            module = __import__('.actions', {}, None, [name])
            act = getattr(module, name)(paths, **kwargs)
            dS.append(act)
            for prop in act.required_static_property_dS:
                static_properties.add(prop)
            for prop in act.required_kinetic_property_dS:
                kinetic_properties.add(prop)
        static_properties = list(static_properties)
        kinetic_properties = list(kinetic_properties)

        # Build Calculator
        def calculator(_rcoords):
            rcoords.coords = _rcoords
            paths.coords = prj.x_inv(rcoords)

            results = paths.get_properties(properties=static_properties)
            results.update(paths.get_kinetics(properties=kinetic_properties))

            res = []
            for ds in dS:
                res.append(ds.dS(**results))
            return np.array(res).sum(axis=0)

    def optimize(self, paths=None, logfile=None, action_kwargs=None,
                 search_kwargs=None, **args):
        action_kwargs = action_kwargs or self.action_kwargs
        search_kwargs = search_kwargs or self.search_kwargs

        close_log = False
        if logfile is None:
            printt = lambda line: print(line)
        elif isinstance(logfile, str):
            close_log = True
            logfile = open(logfile, 'a')
            printt = logfile.write
        elif logfile.__class__.__name__ == "TextIOWrapper":
            printt = lambda line: logfile.write(line + '\n')
        action_name = action_name or self.action_name
        if type(action_name) == str:
            action_name = [action_name]
        printt('Action name  : ' + ' + '.join(action_name))
        Et = self.get_target_energy(paths, Et=Et, Et_type=Et_type)
        action_name_lower = [action.lower() for action in action_name]
        if 'energy conservation' in action_name_lower:
            printt('Target energy: ' + str(Et))
            printt('Target type  : ' + str(self.Et_type))
            printt('muE          : ' + str(self.muE))
        if 'onsager machlup' in action_name_lower:
            printt('gamma        : ' + str(self.gam))

        act = self.action(paths, Et=Et)
        jac = self.grad_action(paths, Et=Et)
        search_kwargs = self.get_search_kwargs(search_kwargs=search_kwargs,
                                               **args)
        printt('            Iter   nfev   njev        S   dS_max')
        while True:
            # print(res.message)
            if self.prj_search:
                x0 = self.prj.x(paths.coords(index=np.s_[1:-1])).flatten()
            else:
                x0 = paths.coords[..., 1:-1].flatten()
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            printt('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 4:
                break
        printt(paths.model.get_state_info())

        if self.prj_search:
            x0 = self.prj.x(paths.coords(index=np.s_[1:-1]))
            origshape = x0.shape
            x0 = x0.flatten()
        else:
            x0 = paths.coords[..., 1:-1].flatten()
        om = 'Onsager Machlup'
        self.results[om] = self.action(paths, action_name=om,
                                       muE=0.)(x0)
        printt('OM      : %.2f' % self.results[om])
        if close_log:
            logfile.close()

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
        msg_lower = str(res.message).lower()
        if 'desired' in msg_lower:
            self.results['msg'] = 'Warning'
        elif 'maximum' in msg_lower:
            self.results['msg'] = 'Max out'
        elif 'successfully' in msg_lower:
            self.results['msg'] = 'Success'
        else:
            if len(msg_lower) > 7:
                self.results['msg'] = msg_lower[:7]
            else:
                self.results['msg'] = msg_lower + ' ' * (7-len(msg_lower))
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

    def get_search_kwargs(self, search_kwargs=None, **args):
        if search_kwargs is not None:
            return search_kwargs
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

        search_kwargs = {
            'method': method, 'tol': tol,
            'options': {'disp': disp, 'maxiter': maxiter,
                        'eps': eps}}
        # search_kwargs.update(args)
        return search_kwargs

    def check_grad(self, paths=None, search_kwargs=None, **kwargs):
        search_kwargs = search_kwargs or self.search_kwargs
        if self.prj_search:
            x0 = self.prj._x(paths.coords.coords[..., 1:-1])
        else:
            x0 = paths.coords.coords[..., 1:-1]
            shape = x0.shape
        act = self.action(paths)
        jac = self.grad_action(paths)
        print(check_grad(act, jac, x0.flatten(), **search_kwargs))
        if self.prj_search:
            paths.coords.coords[..., 1:-1] = self.prj._x_inv(x0)
        else:
            paths.coords.coords[..., 1:-1] = x0.reshape(org_shape)

    def isConverged(self, paths):
        if self.results.get('jac_max') is None:
            return False
        # return self.results.get('jac_max') < self.tol
        err = self.results.get('jac_max') / np.abs(self.results.get('fun', 1.))
        return err < self.tol


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
