from taps.pathfinder import PathFinder
from taps.utils.shortcut import isbool, isstr, isflt, issclr
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize, check_grad


class Classic:
    required_static_property_S = ['potential']
    required_static_property_dS = ['gradients']
    required_kinetic_property_S = ['kinetic_energies']
    required_kinetic_property_dS = ['kinetic_energies_gradient']

    def __init__(self, dt=None):
        self.dt = dt

    def S(self, kinetic_energies=None, potential=None, **kwargs):
        # K = paths.get_kinetic_energies(index=np.s_[:])
        # V = paths.get_potential_energy(index=np.s_[:])
        L = kinetic_energies - potential  # K-V
        S = L.sum(axis=0) * self.dt
        return S

    def dS(self, kinetic_energies_graidient=None, gradients=None, **kwargs):
        # Fp = paths.get_kinetic_energies_gradients(index=np.s_[:])
        # F = -paths.get_gradients(index=np.s_[:])
        dS = (kinetic_energies_graidient - gradients) * self.dt   # Fp + F
        # action - reaction
        # dS[..., 1] -= dS[..., 0]
        # dS[..., -2] -= dS[..., -1]
        return dS[..., 1:-1]


class EnergyRestraint:
    required_static_property_S = ['potential']
    required_static_property_dS = ['potential', 'gradients']
    required_kinetic_property_S = ['kinetic_energies', 'masses']
    required_kinetic_property_dS = ['momentums', 'kinetic_energies']

    def __init__(self, muE=None, Et=None, dt=None, D=None, N=None, shape=None,
                 **kwargs):
        self.muE = muE
        self.Et = Et
        self.dt = dt
        self.D = D
        self.N = N
        self.shape=shape

    def S(self, kinetic_energies=None, potential=None, **kwargs):
        # H = paths.get_total_energy(index=np.s_[:])
        H = kinetic_energies + potential
        return self.muE * ((H - self.Et) ** 2).sum()

    def dS(self, potential=None, gradients=None, momentums=None,
           kinetic_energies=None, **kwargs):
        D, N = self.D, self.N
        # F = -paths.get_gradients(index=np.s_[:])
        F = -gradients.reshape((D, N))
        # p = paths.get_momentums(index=np.s_[:])
        p = momentums.reshape((D, N))

        # K = paths.get_kinetic_energies(index=np.s_[:])
        K = kinetic_energies

        # V = paths.get_potential_energy(index=np.s_[:])
        V = potential

        H = K + V
        dS = np.zeros((self.D, self.N))
        # D x (N - 2)
        dS[..., 1:-1] = ((H[:-2] - self.Et) * p[..., :-2] / self.dt
              - ((H[1:-1] - self.Et) * (
              p[..., 1:-1] / self.dt + F[..., 1:-1])))
        # action - reaction
        # dS[..., 0] -= -(H[0] - self.Et) * (p[..., 0] / self.dt + F[..., 0])
        dS[..., [0, -1]] = 0.
        dS = 2 * self.muE * dS
        return dS.reshape(self.shape)


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
            T = paths.get_kinetic_energies_energy()
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


class Hamiltonian:
    required_static_property_S = ['gradients']
    required_static_property_dS = ['gradients', 'hessian']
    required_kinetic_property_S = ['velocities', 'masses']
    required_kinetic_property_dS = ['accelerations', 'masses']

    def S(self, potential=None, kinetic_energies=None, **kwargs):
        return paths.get_total_energy(index=np.s_[:]).sum()

    def dS(self, gradients=None, kinetic_energies_gradient=None):
        Fp = paths.get_kinetic_energies_gradients(index=np.s_[1:-1])
        F = -paths.get_gradients(index=np.s_[1:-1])

        dS[..., [0, -1]] = 0.
        return Fp - F


class OnsagerMachlup:
    required_static_property_S = ['gradients']
    required_static_property_dS = ['gradients', 'hessian']
    required_kinetic_property_S = ['velocities', 'masses']
    required_kinetic_property_dS = ['accelerations', 'masses']

    def __init__(self, gam=None, dt=None, D=None, N=None, shape=None,
                 **kwargs):
        self.gam=gam
        self.dt=dt
        self.D=D
        self.N=N
        self.shape=shape

    def S(self, potential=None, gradients=None, velocities=None, masses=None,
          **kwargs):
        """
        2qj 2qj21 2qj11 2D2
        """
        D, N = self.D, self.N
        # Vi, Vf = paths.get_potential_energy(index=[0, -1])
        # F = -paths.get_gradients(index=np.s_[:]).reshape((D, N))
        F = -gradients.reshape((D, N))
        dV = -np.concatenate([F, F[..., -1, np.newaxis]], axis=-1)
        # v = paths.get_velocities(index=np.s_[:]).reshape((D, N))
        v = velocities.reshape((D, N))
        # m = paths.get_effective_mass(index=np.s_[:])
        m = masses
        if isinstance(m, np.ndarray):
            m = m.reshape(D, 1)
        _gam = self.gam * m
        # ldVl2 = (dV * dV).sum(axis=0)
        ldVl2 = dV * dV
        # DxMx(P-1) -> P
        Som = (_gam * (v * v)
               + (ldVl2[..., 1:] + ldVl2[..., :-1]) / 2 / _gam
               - v * (dV[..., 1:] - dV[..., :-1])).sum() * self.dt / 4
        # Som += (Vf - Vi) / 2
        # Som = Som.sum()
        # Som *= 0.25 * dt
        return Som

    def dS(self, gradients=None, hessian=None, accelerations=None,
           masses=None, **kwargs):
        D, N = self.D, self.N
        # F = -paths.get_gradients(index=np.s_[:]).reshape((D, N))
        F = -gradients.reshape(D, N)
        dV = -np.hstack([F[:, 0, np.newaxis], F, F[:, -1, np.newaxis]])
        # H = paths.get_hessian(index=np.s_[:]).reshape((D, D, N))
        H = hessian.reshape(D, D, N)
        # a = paths.get_accelerations(index=np.s_[:]).reshape((D, N))
        a = accelerations.reshape(D, N)
        # m = paths.get_effective_mass(index=np.s_[:])
        m = masses
        if isinstance(m, np.ndarray):
            m = m.reshape(D, 1)
        _gam = self.gam * m

        notation = 'ij...,i...->j...'
        dS = (0.5 * _gam * a * self.dt) \
            - 0.25 * (2 * dV[..., 1:-1] - dV[..., 2:] - dV[..., :-2]) \
            + np.einsum(notation, H,
                        0.5 * self.dt * dV[..., 1:-1] / _gam
                        - 0.25 * self.dt * self.dt * a)
        # action - reaction
        # dS[..., 1] -= dS[..., 0]
        # dS[..., -2] -= dS[..., -1]
        dS[..., [0, -1]] = 0.
        return dS.reshape(self.shape)


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

    def __init__(self, action_kwargs=None, search_kwargs=None, use_grad=True,
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
        pcoords = prj.x(paths.coords)

        D = pcoords.D
        parameters = {
            'dt': paths.coords.dt,
            'N': paths.coords.N,
            'D': paths.coords.D
        }

        S = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.dao', {}, None, [class_name])
            act = getattr(module, class_name)(**parameters, **kwargs)
            S.append(act)
            for prop in act.required_static_property_S:
                static_properties.add(prop)
            for prop in act.required_kinetic_property_S:
                kinetic_properties.add(prop)
        static_properties = list(static_properties)
        kinetic_properties = list(kinetic_properties)

        def calculator(_pcoords):
            pcoords.coords = _pcoords.reshape(D, -1)
            paths.coords = prj.x_inv(pcoords)

            results = paths.get_properties(properties=static_properties,
                                           return_dict=True)
            results.update(paths.get_kinetics(properties=kinetic_properties,
                                              return_dict=True))

            res = 0.
            for s in S:
                res += s.S(**results)
            return res

        return calculator

    def grad_action(self, paths, action_kwargs=None, prj=None):
        if not self.use_grad:
            return None

        # Prepare step
        action_kwargs = action_kwargs or self.action_kwargs
        prj = prj or self.prj
        pcoords = prj.x(paths.coords)

        D = pcoords.D
        parameters = {
            'dt': paths.coords.dt,
            'N': paths.coords.N,
            'D': paths.coords.D,
            'shape': paths.coords.shape
        }

        dS = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.dao', {}, None, [class_name])
            act = getattr(module, class_name)(**parameters, **kwargs)
            dS.append(act)
            for prop in act.required_static_property_dS:
                static_properties.add(prop)
            for prop in act.required_kinetic_property_dS:
                kinetic_properties.add(prop)
        static_properties = list(static_properties)
        kinetic_properties = list(kinetic_properties)

        # Build Calculator
        def calculator(_pcoords):
            pcoords.coords = _pcoords.reshape((D, -1))
            paths.coords = prj.x_inv(pcoords)

            results = paths.get_properties(properties=static_properties,
                                           return_dict=True)
            results.update(paths.get_kinetics(properties=kinetic_properties,
                                           return_dict=True))

            res = 0.
            for ds in dS:
                res += ds.dS(**results)
            return prj.f(res, paths.coords)[0].flatten()
        return calculator

    def optimize(self, paths=None, logfile=None, action_kwargs=None,
                 search_kwargs=None, **args):
        action_kwargs = action_kwargs or self.action_kwargs
        search_kwargs = search_kwargs or self.search_kwargs

        close_log = False
        if logfile is None:
            def printt(*line, end='\n'):
                print(*line, end=end)
        elif isinstance(logfile, str):
            logdir = os.path.dirname(logfile)
            if logdir == '':
                logdir = '.'
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            logfile = open(logfile, 'a')
            def printt(*line, end='\n'):
                lines = ' '.join([str(l) for l in line]) + end
                logfile.write(lines)
                logfile.flush()
            close_log = True
        elif logfile.__class__.__name__ == "TextIOWrapper":
            def printt(*line, end='\n'):
                lines = ' '.join([str(l) for l in line]) + end
                logfile.write(lines)
                logfile.flush()
        else:
            printt = logfile
        action_name = list(action_kwargs.keys())
        printt('=================================')
        printt('      DAO Parameters')
        printt('=================================')
        for name, parameters in action_kwargs.items():
            printt('{0:<}'.format(name))

            for k, v in parameters.items():
                printt('  {0:<10} : {1:<5}'.format(k, v))
            printt(" ")

        act = self.action(paths, action_kwargs=action_kwargs)
        jac = self.grad_action(paths, action_kwargs=action_kwargs)

        printt('            Iter   nfev   njev        S   dS_max')
        while True:
            pcoords = self.prj.x(paths.coords)
            x0 = pcoords.coords.flatten()
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            printt('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                  '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 4:
                break
        printt(paths.model.get_state_info())

        pcoords = self.prj.x(paths.coords)
        origshape = pcoords.shape

        printt('=================================')
        printt('            Results')
        printt('=================================')
        total_S = 0.
        for action_name, S_kwargs in action_kwargs.items():
            S = self.action(paths, action_kwargs={action_name: S_kwargs})(
                            pcoords.coords.flatten())
            printt(" {0:<27} : {1:<5}".format(action_name, S))
            self.results[action_name] = S
            total_S += S

        printt(' {0:<27} : {1:<5}'.format("Total S", total_S))
        self.results["Total S"] = total_S

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
        elif 'convergence' in msg_lower:
            self.results['msg'] = 'Converge'
        elif 'abnormal_termination_in_lnsrch' in msg_lower:
            self.results['msg'] = 'LNSrch_high'

        else:
            printt(res.message)
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


    def check_grad(self, paths=None, search_kwargs=None, **kwargs):
        search_kwargs = search_kwargs or self.search_kwargs
        pcoords = self.prj.x(paths.coords)
        shape = pcoords.shape.shape
        act = self.action(paths)
        jac = self.grad_action(paths)
        print(check_grad(act, jac, pcoords.coords.flatten(), **search_kwargs))
        paths.coords = self.prj.x_inv(pcoords)

    def isConverged(self, paths):
        if self.results.get('jac_max') is None:
            return False
        # return self.results.get('jac_max') < self.tol
        err = self.results.get('jac_max') / np.abs(self.results.get('fun', 1.))
        return err < self.tol
