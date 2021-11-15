import copy
import numpy as np
# from abc import ABC
from taps.utils.arraywrapper import arraylike

@arraylike
class Coordinate:
    """ Discretized Coordinates
    ttt : Total transition time

    """

    def __init__(self, coords=None, epoch=3, unit='ang/fs'):
        coords = np.asarray(coords, dtype=float)
        self.coords = coords
        self.epoch = epoch # Total transition time
        self.unit = unit

    def __call__(self, index=np.s_[:], coords=None):
        if coords is not None:
            kwargs = self.__dict__.copy()
            del kwargs['coords']
            return self.__class__(coords=coords, **kwargs)
        if index.__class__.__name__ == 'slice' and index == np.s_[:]:
            return self
        kwargs = self.__dict__.copy()
        del kwargs['coords']
        idx = np.arange(self.N)[index].reshape(-1)
        coords = self.coords[..., idx]
        return self.__class__(coords=coords, **kwargs)

    def similar(self):
        dct = dict([(k, v) for k, v in self.__dict__.items() if k != 'coords'])
        return self.__class__(**dct)

    def copy(self):
        """ Return deep copy of itself"""
        return copy.deepcopy(self)

    def flat(self):
        """ Return flat version of paths"""
        N = self.N
        return self.coords.reshape((-1, N))

    def get_kinetics(self, paths, properties=['kinetic_energies'], **kwargs):
        """
        Dumb way of calculate.. but why not.
        """
        if type(properties) == str:
            properties = [properties]

        # Make a list of requirments for minimal calulation
        irreplaceable = set()
        for prop in properties:
            if prop in ['masses', 'momentums', 'kinetic_energies',
                        'kinetic_energy_gradients']:
                irreplaceable.add('masses')
            if prop in ['displacements']:
                irreplaceable.add('displacements')
            if prop in ['velocities', 'distances', 'speeds', 'momentums',
                        'kinetic_energies']:
                irreplaceable.add('velocities')
            if prop in ['accelerations', 'kinetic_energy_gradients']:
                irreplaceable.add('accelerations')

        # Calculate
        parsed_properties = list(irreplaceable)
        parsed_results = {}
        for prop in parsed_properties:
            parsed_results[prop] = getattr(self, prop)(paths, **kwargs)

        # Name convention
        m, d, v, a = 'masses', 'displacements', 'velocities', 'accelerations'
        # Assemble
        results = {}
        for prop in properties:
            if prop in [m, d, v, a]:
                results[prop] = parsed_results[prop]
            elif prop in ['distances', 'speeds']:
                if results.get(prop) is not None:
                    continue
                lvl = np.linalg.norm(parsed_results[v], axis=0)
                if 'speeds' in properties:
                    results['speeds'] = lvl
                if 'distances' in properties:
                    results['distances'] = np.add.accumulate(lvl)*dt
            elif prop in ['momentums']:
                results[prop] = parsed_results[m] * parsed_results[v]
            elif prop in ['kinetic_energies']:
                vv = parsed_results[v] * parsed_results[v]
                results[prop] = 0.5 * parsed_results[m] * vv
            elif prop in ['kinetic_energy_gradients']:
                results[prop] = m * parsed_results[a]

        if len(properties) == 1:
            return results[properties[0]]
        return results

    def get_masses(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='masses', **kwargs)

    def get_distances(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='distances', **kwargs)

    def get_speeds(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='speeds', **kwargs)

    def get_displacements(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='displacements', **kwargs)

    def get_velocities(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='velocities', **kwargs)

    def get_accelerations(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='accelerations', **kwargs)

    def get_momentums(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='momentums', **kwargs)

    def get_kinetic_energies(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='kinetic_energy', **kwargs)

    def get_kinetic_energy_gradients(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='kinetic_energy_gradients',
                                 **kwargs)

    @property
    def N(self):
        """ Number of steps; coords.shape[-1]"""
        return self.coords.shape[-1]

    @property
    def D(self):
        """ Total dimension of coords. """
        shape = self.coords.shape
        if len(shape) == 3:
            return shape[0] * shape[1]
        else:
            return shape[0]

    @property
    def A(self):
        """ Number of individual atoms or components """
        shape = self.coords.shape
        if len(shape) == 3:
            return shape[1]
        else:
            return 1

    @property
    def dt(self):
        return self.ttt / self.N

    def __array__(self, dtype=float):
        return self.coords

    def __bool__(self):
        return bool(self.any())  # need to convert from np.bool_

    def __repr__(self):
        return self.__class__.__name__ + '{}'.format(self.coords.shape)
