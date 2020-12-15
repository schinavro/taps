import copy
import pickle
import numpy as np
from taps.coords import Coords

# Parameters that will be saved
paths_parameters = {
    'Pk': {'default': '{N:d}', 'assert': '{name:s} > 0'},
    'prefix': {'default': "'paths'", 'assert': 'len({name:s}) > 0'},
    'directory': {'default': "'.'", 'assert': 'len({name:s}) > 0'},
    'tag': {'default': "dict()", 'assert': 'True'}
}

class_objects = {
    'model': {'from': 'taps.model.model'},
    'finder': {'from': 'taps.pathfinder'},
    'imgdata': {'from': 'taps.db.data'}
}


class Paths:
    """
    coords : coordinate representation of atomic configuration
    epoch : total time spent from initial state to final state
    images : Atoms containing full cartesian coordinate
    prj : coordinate projector, function maps between reduced coordinate and
          full cartesian coordinate
    pk : fourier represenation of coords
    label : name

    """

    def __init__(self, coords=None, label=None, model='Model',
                 finder='PathFinder', imgdata='ImageData',
                 plotter='Plotter', tag=None, **kwargs):

        self.coords = coords
        self.label = label

        self.model = model
        self.finder = finder
        self.imgdata = imgdata
        self.tag = tag

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key in ['coords']:
            if value is None:
                value = np.zeros((0, 0))
            if 'Coords' in value.__class__.__name__:
                super().__setattr__(key, value)
            elif 'SineBasis' in value.__class__.__name__:
                super().__setattr__(key, value)
            else:
                crd = self.__dict__.get('coords')
                if crd is None:
                    value = Coords(coords=np.asarray(value))
                else:
                    value = crd.__class__(coords=np.asarray(value))
                super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        elif isinstance(getattr(type(self), key, None), property):
            super().__setattr__(key, value)

        elif key in paths_parameters.keys():
            default = paths_parameters[key]['default']
            assertion = paths_parameters[key]['assert']
            if value is None:
                value = eval(default.format(N=self.N))
            assert eval(assertion.format(name='value')), (key, assertion)
            super().__setattr__(key, value)
        elif key in class_objects:
            if type(value) == str:
                from_ = class_objects[key]['from']
                if 'Gaussian' in value:
                    from_ = 'taps.gaussian'
                module = __import__(from_, {}, None, [value])
                value = getattr(module, value)()
            super().__setattr__('_' + key, value)
        elif key in self._model.model_parameters:
            setattr(self._model, key, value)
        elif key in self._finder.finder_parameters:
            setattr(self._finder, key, value)
        elif key in self._imgdata.imgdata_parameters:
            setattr(self._imgdata, key, value)
        elif 'model_' in key and key[6:] in self._model.model_parameters:
            setattr(self._model, key[6:], value)
        elif 'finder_' in key and key[7:] in self._finder.finder_parameters:
            setattr(self._finder, key[7:], value)
        elif 'imgdata_' in key and \
             key[10:] in self._imgdata.imgdata_parameters:
            setattr(self._imgdata, key[10:], value)
        else:
            raise AttributeError('Can not set key `%s`' % key)

    def __getattr__(self, key):
        if key[0] == '_':
            return super().__getattribute__(key)
        elif key in class_objects:
            return self.__dict__['_' + key]
        elif key in self.__dict__:
            return self.__dict__[key]
        elif 'model_' in key and key[6:] in self._model.model_parameters:
            return getattr(self.__dict__['_model'], key[6:])
        elif 'finder_' in key and key[7:] in self._finder.finder_parameters:
            return getattr(self.__dict__['_finder'], key[7:])
        elif 'imgdata_' in key and \
             key[10:] in self._imgdata.imgdata_parameters:
            return self.__dict__['_imgdata'].__dict__[key[10:]]
        # Short hand notation
        elif key in self.__dict__['_finder'].finder_parameters:
            return getattr(self.__dict__['_finder'], key)
        elif key in self.__dict__['_model'].model_parameters:
            return getattr(self.__dict__['_model'], key)
        elif key in self.__dict__['_imgdata'].imgdata_parameters:
            return self.__dict__['_imgdata'].__dict__[key]
        else:
            raise AttributeError("Key called `%s` not exist" % key)

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        if self.prefix is None:
            return self.directory + '/'

        return '{}/{}'.format(self.directory, self.prefix)

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix

    def get_displacements(self, **kwargs):
        return self.model.get_displacements(self, **kwargs)

    def get_momentum(self, **kwargs):
        return self.model.get_momentum(self, **kwargs)

    def get_kinetic_energy(self, **kwargs):
        return self.model.get_kinetic_energy(self, **kwargs)

    def get_kinetic_energy_gradient(self, **kwargs):
        return self.model.get_kinetic_energy_gradient(self, **kwargs)

    def get_velocity(self, **kwargs):
        return self.model.get_velocity(self, **kwargs)

    def get_acceleration(self, **kwargs):
        return self.model.get_acceleration(self, **kwargs)

    def get_accelerations(self, **kwargs):
        return self.model.get_acceleration(self, **kwargs)

    def get_effective_mass(self, **kwargs):
        return self.model.get_effective_mass(self, **kwargs)

    def get_properties(self, **kwargs):
        return self.model.get_properties(self, **kwargs)

    def get_potential_energy(self, **kwargs):
        return self.model.get_potential_energy(self, **kwargs)

    def get_potential(self, **kwargs):
        return self.model.get_potential(self, **kwargs)

    def get_potential_energies(self, **kwargs):
        return self.model.get_potential_energies(self, **kwargs)

    def get_potentials(self, **kwargs):
        return self.model.get_potentials(self, **kwargs)

    def get_forces(self, **kwargs):
        return self.model.get_forces(self, **kwargs)

    def get_gradient(self, **kwargs):
        return self.model.get_gradient(self, **kwargs)

    def get_gradients(self, **kwargs):
        return self.model.get_gradients(self, **kwargs)

    def get_hessian(self, **kwargs):
        return self.model.get_hessian(self, **kwargs)

    def get_total_energy(self, **kwargs):
        V = self.model.get_potential_energy(self, **kwargs)
        T = self.model.get_kinetic_energy(self, **kwargs)
        return V + T

    def get_covariance(self, index=np.s_[1:-1]):
        cov_coords = self.model.get_covariance(self, index=np.s_[1:-1])
        _ = np.diag(cov_coords)
        cov_coords = _.copy()
        cov_coords[_ < 0] = 0
        sigma_f = self.model.hyperparameters.get('sigma_f', 1)
        return 1.96 * np.sqrt(cov_coords) / 2 / sigma_f

    def get_higest_energy_idx(self):
        E = self.get_potential_energy()
        return np.argmax(E)

    def get_lowest_confident_idx(self):
        cov = self.model.get_covariance(self)
        return np.argmax(np.diag(cov))

    def get_data(self, **kwargs):
        return self.model.get_data(self, **kwargs)

    def add_data(self, index=None, coords=None, cache_model=True):
        """
        if index given -> create coords -> add_data
        if coords given -> add_data
        coords : atomic configuration
        datum : atomic configuration with energy and forces
           shape,  A number of tuples
             [('H', desc, displacement, potential, forces, directory),
              ('C', desc, ...),
              ('N', desc, ...), ...]
        found : list of Boolean or None, search results
        id : list of int or None, where it is
        """
        if coords is None:
            # Create positional descriptor
            coords = self.coords(index=index)
        ids = self.imgdata.add_data(self, coords, search_similar_image=True)
        if cache_model:
            self.model.add_data_ids(ids)
        return ids

    def simple_coords(self, index=np.s_[:]):
        coords = np.zeros(self.DMP)
        endpoint = self.coords[..., [0, -1]]
        for d in range(self.D):
            for m in range(self.M):
                coords[d, m] = np.linspace(*endpoint[d, m], self.N)
        return coords[..., index]

    def fluctuate(self, initialize=True, cutoff_f=70, temperature=0.03):
        if initialize:
            self.coords = self.simple_coords()
        fluc = np.zeros(self.Pk)
        fluc[:cutoff_f] = temperature * np.linspace(2 * self.N, 0, cutoff_f)
        self.rcoords += fluc * (np.random.rand(*self.rcoords.shape) - 0.5)

    def search(self, **kwargs):
        self.finder.search(self, **kwargs)

    @property
    def N(self):
        return self.coords.N

    @property
    def D(self):
        return self.coords.D

    @property
    def Nk(self):
        return self.coords.Nk

    @property
    def A(self):
        return self.coords.A

    @property
    def masses(self):
        return self.model.get_mass()

    def plot(self, filename=None, savefig=False, gaussian=False, **kwargs):
        self.plotter.plot(self, filename=filename, savefig=savefig,
                          gaussian=gaussian, **kwargs)

    def reset_cache(self):
        self.model._cache = {}
        self.real_model._cache = {}
        self.imgdata._cache = {}
        self.prj._cache = {}

    def reset_results(self):
        self.model.results = {}
        self.real_model.results = {}
        self.finder.results = {}
        self.real_finder.results = {}

    def to_pickle(self, filename=None, simple=True):
        if simple:
            filename = filename or self.label + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(f, self)

    def copy(self):
        """Return a copy."""
        return copy.deepcopy(self)
