import copy
import pickle
import numpy as np
from taps.coords import Cartesian

# Parameters that will be saved
paths_parameters = {
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
    """ Paths class infterfacing other modules

    Centeral class for pathway calculation.
    Object contains coordinates class and potential class and database

    Parameters
    ----------

    coords  : Cartesian class
        Coordinate representation that connect initial state to final state.
        Calculation involving intermediate states, such as, kinetic energy
        or momentum, can be found in the Cartesian class.
    label   : string
        Set of character that distinguish from other Paths classes. This can be
        use in other module as default filename.
    model   : Model class
        Model that calculates potential energy of intermediate coordinates.
    finder  : PathFinder class
        Class that optimize the coordinates suitable for the finder class.
    imgdata : Database class
        Class made to easily accessible to the calculated database.
    tag     : dict
        Auxilary parameters for external use.

    Example
    -------

    >>> import numpy as np
    >>> from taps.paths import Paths
    >>> x = np.linspace(-0.55822365, 0.6234994, 300)
    >>> y = np.linspace(1.44172582, 0.02803776, 300)
    >>> paths = Paths(coords = np.array([x, y]))

    >>> from taps.model.mullerbrown import MullerBrown
    >>> paths.model = MullerBrown()

    >>> from taps.visualize import view
    >>> view(paths, calculate_map=True, viewer='MullerBrown')
    [Map should be shown]
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
            if 'Cartesian' in value.__class__.__name__:
                super().__setattr__(key, value)
            elif 'SineBasis' in value.__class__.__name__:
                super().__setattr__(key, value)
            else:
                crd = self.__dict__.get('coords')
                if crd is None:
                    value = Cartesian(coords=np.asarray(value))
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
        """ string used for default name of calculation """
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
        """ Calculate the distances of the pathway (accumulative)

        It usually used for plotting, i.e. potential energy / distance
        Calls the :meth:`get_displacements` at the ``paths.model``.
        Model calls the :meth:`get_displacements` in the ``paths.coords``.
        """
        return self.model.get_displacements(self, **kwargs)

    def get_momentum(self, **kwargs):
        """ Calculate momentum of the pathway"""
        return self.model.get_momentum(self, **kwargs)

    def get_kinetic_energy(self, **kwargs):
        """ Calculate kinetic energy of the pathway"""
        return self.model.get_kinetic_energy(self, **kwargs)

    def get_kinetic_energy_gradient(self, **kwargs):
        """ Calculate kinetic energy gradient.
        differentiate Kinetic energy w.r.t. each point, That is we calculate
        :math:`\partial_{\mathbf{x}}E_{\mathrm{kin}}`
        """
        return self.model.get_kinetic_energy_gradient(self, **kwargs)

    def get_velocity(self, **kwargs):
        """ Calculate velocity
        If :class:`Cartesian` is cartesian, velocity is calculated via
        :math:`\mathbf{x}_{i+1} - \mathbf{x}_{i}`"""
        return self.model.get_velocity(self, **kwargs)

    def get_acceleration(self, **kwargs):
        """ Calculate acceleration
        If :class:`Cartesian` is cartesian, velocity is calculated via
        """
        return self.model.get_acceleration(self, **kwargs)

    def get_accelerations(self, **kwargs):
        """ Calculate acceleration(s)
        If :class:`Cartesian` is cartesian, velocity is calculated via
        """
        return self.model.get_acceleration(self, **kwargs)

    def get_mass(self, **kwargs):
        """ Calculate mass """
        return self.model.get_mass(self, **kwargs)

    def get_effective_mass(self, **kwargs):
        """ Calculate effective mass"""
        return self.model.get_effective_mass(self, **kwargs)

    def get_properties(self, **kwargs):
        """ Directly calls the :meth:`get_properties` in ``paths.model``"""
        return self.model.get_properties(self, **kwargs)

    def get_potential_energy(self, **kwargs):
        """ Calculate potential( energy) """
        return self.model.get_potential_energy(self, **kwargs)

    def get_potential(self, **kwargs):
        """ Calculate potential """
        return self.model.get_potential(self, **kwargs)

    def get_potential_energies(self, **kwargs):
        """ Equivalanet to Calculate potentials"""
        return self.model.get_potential_energies(self, **kwargs)

    def get_potentials(self, **kwargs):
        """ Calculate potentials, individual energy of each atoms"""
        return self.model.get_potentials(self, **kwargs)

    def get_forces(self, **kwargs):
        """ Calculate - potential gradient"""
        return self.model.get_forces(self, **kwargs)

    def get_gradient(self, **kwargs):
        """ Calculate potential gradient"""
        return self.model.get_gradient(self, **kwargs)

    def get_gradients(self, **kwargs):
        """ Calculate potential gradient(s)"""
        return self.model.get_gradients(self, **kwargs)

    def get_hessian(self, **kwargs):
        """ Calculate Hessian of a potential"""
        return self.model.get_hessian(self, **kwargs)

    def get_total_energy(self, **kwargs):
        """ Calculate kinetic + potential energy"""
        V = self.model.get_potential_energy(self, **kwargs)
        T = self.model.get_kinetic_energy(self, **kwargs)
        return V + T

    def get_covariance(self, **kwargs):
        """ Calculate covariance. It only applies when potential is guassian"""
        return self.model.get_covariance(self, **kwargs)

    def get_higest_energy_idx(self):
        """ Get index of highest potential energy simplified"""
        E = self.get_potential_energy()
        return np.argmax(E)

    def get_lowest_confident_idx(self):
        """ Get index of lowest of covariance simplified"""
        cov = self.model.get_covariance(self)
        return np.argmax(np.diag(cov))

    def get_data(self, **kwargs):
        """" list of int; Get rowid of data"""
        return self.model.get_data(self, **kwargs)

    def add_data(self, index=None, coords=None, cache_model=True,
                 regression=True, **kwargs):
        """ Adding a calculation data to image database

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
        if regression:
            self.model.regression(self)
        #### Need to change
        self.model.optimized = False
        #### Need to remove
        return ids

    def simple_coords(self):
        """Simple line connecting between init and fin"""
        coords = np.zeros(self.coords.shape)
        init = self.coords[..., [0]]
        fin = self.coords[..., [-1]]
        dist = fin - init            # Dx1 or 3xAx1
        simple_line = np.linspace(0, 1, self.coords.N) * dist
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

    def search(self, **kwargs):
        """ Calculate optimized pathway"""
        self.finder.search(self, **kwargs)

    @property
    def N(self):
        """ Return number of steps. Usually, coords.shape[-1]"""
        return self.coords.N

    @property
    def D(self):
        """ Return dimension of the system. Usually, coords.shape[0]"""
        return self.coords.D

    @property
    def Nk(self):
        return self.coords.Nk

    @property
    def A(self):
        """ Return a number of atoms. It only applies coords is rank 3"""
        return self.coords.A

    def reset_cache(self):
        """ Delete current cache on all modules"""
        self.model._cache = {}
        self.real_model._cache = {}
        self.imgdata._cache = {}
        self.prj._cache = {}

    def reset_results(self):
        """ Delete current results on all modules"""
        self.model.results = {}
        self.real_model.results = {}
        self.finder.results = {}
        self.real_finder.results = {}

    def to_pickle(self, filename=None, simple=True):
        """ Saving pickle simplified"""
        if simple:
            filename = filename or self.label + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(f, self)

    def copy(self):
        """Return a copy."""
        return copy.deepcopy(self)

    def save(self, *args, **kwargs):
        """
        Different method of writing is required for different models.
        """
        self.model.save(self, *args, **kwargs)
