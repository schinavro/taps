import copy
import numpy as np
from taps.utils.arraywrapper import arraylike

#@arraylike
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

    @property
    def shape(self):
        return self.coords.shape

    def reshape(self, *shape):
        self.coords.reshape(*shape)
        return self

    def similar(self, coords=None):
        dct = dict([(k, v) for k, v in self.__dict__.items() if k != 'coords'])
        return self.__class__(coords=coords, **dct)

    def tobytes(self):
        return self.coords.tobytes()

    def copy(self):
        """ Return deep copy of itself"""
        return copy.deepcopy(self)

    def flat(self):
        """ Return flat version of paths"""
        N = self.N
        self.coords.reshape((-1, N))
        return self

    def flatten(self):
        self.coords = self.coords.flatten()
        return self

    def set_coordinates(self, coords, index=None):
        if index is not None:
            self.coords[..., index] = coords
        else:
            self.coords = coords

    def simple_coords(self):
        """Simple line connecting between init and fin"""
        coords = np.zeros(self.coords.shape)
        init = self.coords[..., [0]]
        fin = self.coords[..., [-1]]
        dist = fin - init            # Dx1 or 3xAx1
        simple_line = np.linspace(0, 1, self.N) * dist
        coords = (simple_line + init)  # N x A x 3 -> 3 x A x N
        return coords


    def fluctuate(self, initialize=False, cutoff_f=10, fluctuation=0.03,
                  fourier={'type': 1}):
        """Give random fluctuation"""
        from scipy.fftpack import idst
        rand = np.random.rand
        NN = np.sqrt(2 * (cutoff_f + 1))
        if initialize:
            self.coords = self.simple_coords()
        size = self.coords[..., 1:-1].shape
        fluc = np.zeros(size)
        fluc[..., :cutoff_f] = fluctuation * (0.5 - rand(*size[:-1], cutoff_f))
        self.coords[..., 1:-1] += idst(fluc, **fourier) / NN

    def get_kinetics(self, paths, properties=['kinetic_energies'],
                     return_dict=False, **kwargs):
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
        dt = self.dt
        # Assemble
        results = {}
        for prop in properties:
            if prop in [m, d, v, a]:
                results[prop] = parsed_results[prop]
            elif prop in ['distances', 'speeds', 'kinetic_energies']:
                if results.get(prop) is not None:
                    continue
                vv = parsed_results[v] * parsed_results[v]
                N = vv.shape[-1]
                if 'kinetic_energies' in properties:
                    mvv = parsed_results[m] * vv
                    results[prop] = 0.5 * mvv.reshape(-1, N).sum(axis=0)
                else:
                    lvl = np.sqrt(vv.reshape(-1, N).sum(axis=0))
                if 'speeds' in properties:
                    results['speeds'] = lvl
                if 'distances' in properties:
                    results['distances'] = np.add.accumulate(lvl) * dt
            elif prop in ['momentums']:
                results[prop] = parsed_results[m] * parsed_results[v]
            elif prop in ['kinetic_energy_gradients']:
                results[prop] = m * parsed_results[a]

        if len(properties) == 1 and not return_dict:
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
        return self.get_kinetics(paths, properties='kinetic_energies', **kwargs)

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
        return self.epoch / self.N

    def __array__(self, dtype=float):
        return self.coords

    def __bool__(self):
        return bool(self.any())  # need to convert from np.bool_

    def __repr__(self):
        return self.__class__.__name__ + '{}'.format(self.coords.shape)
