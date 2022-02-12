import os
from taps.pathfinder import PathFinder
import numpy as np
from scipy.optimize import minimize, check_grad

from .dao import DAO


class EnergyRestraint:
    required_static_property_S = ['potential']
    required_static_property_dS = ['potential', 'gradients']
    required_kinetic_property_S = ['kinetic_energies', 'masses']
    required_kinetic_property_dS = ['momentums', 'kinetic_energies', 'epoch']

    def __init__(self, muE=None, Et=None, D=None, N=None, shape=None,
                 **kwargs):
        self.muE = muE
        self.Et = Et
        self.D = D
        self.N = N
        self.shape = shape

    def S(self, kinetic_energies=None, potential=None, **kwargs):
        # H = paths.get_total_energy(index=np.s_[:])
        H = kinetic_energies + potential
        return self.muE * ((H - self.Et) ** 2).sum()

    def dS(self, potential=None, gradients=None, momentums=None,
           kinetic_energies=None, epoch=None, **kwargs):
        D, N, dt = self.D, self.N, epoch / self.N
        # F = -paths.get_gradients(index=np.s_[:])
        F = -gradients.reshape((D, N))
        # p = paths.get_momentums(index=np.s_[:])
        p = momentums.reshape((D, N))

        # K = paths.get_kinetic_energies(index=np.s_[:])
        K = kinetic_energies

        # V = paths.get_potential_energy(index=np.s_[:])
        V = potential

        H = K + V
        dS = np.zeros((D, N))
        # D x (N - 2)
        dS[..., 1:-1] = ((H[:-2] - self.Et) * p[..., :-2] / dt
                         - ((H[1:-1] - self.Et) * (
                            p[..., 1:-1] / dt + F[..., 1:-1])))
        # action - reaction
        # dS[..., 0] -= -(H[0] - self.Et) * (p[..., 0] / self.dt + F[..., 0])
        dS[..., [0, -1]] = 0.
        dS = 2 * self.muE * dS

        dH = -2 * kinetic_energies / dt / N
        dSdt = 2 * self.muE * ((H - self.Et) * dH).sum()
        return dS.reshape(self.shape), dSdt


class OnsagerMachlup:
    required_static_property_S = ['gradients']
    required_static_property_dS = ['gradients', 'hessian']
    required_kinetic_property_S = ['velocities', 'masses', 'epoch']
    required_kinetic_property_dS = ['velocities', 'accelerations', 'masses', 'epoch']

    def __init__(self, gam=None, D=None, N=None, shape=None,
                 **kwargs):
        self.gam = gam
        self.D = D
        self.N = N
        self.shape = shape

    def S(self, epoch=None, potential=None, gradients=None, velocities=None,
          masses=None, **kwargs):
        """
        2qj 2qj21 2qj11 2D2
        """
        D, N, dt = self.D, self.N, epoch / self.N

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
               - v * (dV[..., 1:] - dV[..., :-1])).sum() * dt / 4
        # Som += (Vf - Vi) / 2
        # Som = Som.sum()
        # Som *= 0.25 * dt
        return Som

    def dS(self, epoch=None, velocities=None, gradients=None, hessian=None,
           accelerations=None, masses=None, **kwargs):
        D, N, dt = self.D, self.N, epoch / self.N
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
        dS = (0.5 * _gam * a * dt) \
            - 0.25 * (2 * dV[..., 1:-1] - dV[..., 2:] - dV[..., :-2]) \
            + np.einsum(notation, H,
                        0.5 * dt * dV[..., 1:-1] / _gam
                        - 0.25 * dt * dt * a)
        # action - reaction
        # dS[..., 1] -= dS[..., 0]
        # dS[..., -2] -= dS[..., -1]
        dS[..., [0, -1]] = 0.

        dV = -np.concatenate([F, F[..., -1, np.newaxis]], axis=-1)
        ldVl2 = dV * dV
        v = velocities.reshape((D, N))
        dSdt = (-_gam * (v * v)
                + (ldVl2[..., 1:] + ldVl2[..., :-1]) / 2 / _gam
                ).sum() / 4 / N
        return dS.reshape(self.shape), dSdt


class TransitionTimePenalty:
    required_static_property_S = []
    required_static_property_dS = []
    required_kinetic_property_S = ['epoch']
    required_kinetic_property_dS = ['epoch']

    def __init__(self, muT=None, Tt=None, D=None, N=None, shape=None, **kwargs):
        self.muT = muT
        self.Tt = Tt
        self.D = D
        self.N = N
        self.shape = shape

    def S(self, epoch=None, **kwargs):
        """
        2qj 2qj21 2qj11 2D2
        """
        return self.muT * (epoch - self.Tt) ** 2


    def dS(self, epoch=None, **kwargs):

        return np.zeros((self.D, self.N)), 2 * self.muT * (epoch - self.Tt)


class TAO(DAO):
    """ Time and Action Optimizer (TAO)

    """
    def action(self, paths, coords=None, action_kwargs=None, prj=None):
        """
        return a function calculating specified action
        """
        action_kwargs = action_kwargs or self.action_kwargs
        prj = prj or self.prj
        coords = coords or paths.coords
        pcoords = prj.x(coords)

        pshap = pcoords.shap
        parameters = {
            'dt': coords.dt,
            'N': coords.N,
            'D': coords.D
        }

        S = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.tao', {}, None, [class_name])
            act = getattr(module, class_name)(**parameters, **kwargs)
            S.append(act)
            for prop in act.required_static_property_S:
                static_properties.add(prop)
            for prop in act.required_kinetic_property_S:
                kinetic_properties.add(prop)
        static_properties = list(static_properties)
        kinetic_properties = list(kinetic_properties)

        def calculator(_pcoords):
            _pcoords, epoch = _pcoords[:-1], _pcoords[-1]
            pcoords.coords = _pcoords.reshape(*pshap, -1)
            paths.coords = prj.x_inv(pcoords)
            paths.coords.epoch = epoch

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
        # if True:
        if not self.use_grad:
            return None

        # Prepare step
        action_kwargs = action_kwargs or self.action_kwargs
        prj = prj or self.prj
        pcoords = prj.x(paths.coords)

        pshap = pcoords.shap
        parameters = {
            'N': paths.coords.N,
            'D': paths.coords.D,
            'shape': paths.coords.shape
        }

        dS = []
        static_properties = set()
        kinetic_properties = set()
        for name, kwargs in action_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.tao', {}, None, [class_name])
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
            _pcoords, epoch = _pcoords[:-1], _pcoords[-1]
            pcoords.coords = _pcoords.reshape((*pshap, -1))
            paths.coords = prj.x_inv(pcoords)
            paths.coords.epoch = epoch

            results = paths.get_properties(properties=static_properties,
                                           return_dict=True)
            results.update(paths.get_kinetics(properties=kinetic_properties,
                                              return_dict=True))

            resx = 0.
            rest = 0.
            for ds in dS:
                dSdx, dSdt = ds.dS(**results)
                resx += dSdx
                rest += dSdt
            return np.concatenate([prj.f(resx, paths.coords)[0].flatten(), [rest]])
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
        printt('      TAO Parameters')
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
            x0 = np.concatenate([pcoords.coords.flatten(), [pcoords.epoch]])
            res = minimize(act, x0, jac=jac, **search_kwargs)
            self.set_results(res)
            printt('{msg} : {nit:6d} {nfev:6d} {njev:6d} '
                   '{fun:8.4f} {jac_max:8.4f}'.format(**self.results))
            if res.nit < 4:
                break
            if self.results['msg'] == 'Max out':
                break
        printt(paths.model.get_state_info())

        pcoords = self.prj.x(paths.coords)

        printt('=================================')
        printt('            Results')
        printt('=================================')
        total_S = 0.
        for action_name, S_kwargs in action_kwargs.items():
            x0 = np.concatenate([pcoords.coords.flatten(), [pcoords.epoch]])
            S = self.action(paths, action_kwargs={action_name: S_kwargs})(x0)
            printt(" {0:<27} : {1:<5}".format(action_name, S))
            self.results[action_name] = S
            total_S += S

        printt(' {0:<27} : {1:<5}'.format("Total S", total_S))
        self.results["Total S"] = total_S

        if close_log:
            logfile.close()
