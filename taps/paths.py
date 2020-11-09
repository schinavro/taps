import os
import copy
import pickle
import numpy as np

from ase.data import atomic_masses
from scipy.fftpack import dst, idst

from ase.atoms import Atoms

from ase.calculators.calculator import get_calculator_class
from ase.symbols import Symbols
from taps.utils import Images, ImageIndexing, paths2dct, dct2pd_dct


allowed_properties = {
    'cell': {
        'call': '{atoms:s}.cell',
        'isvariable': "isinstance(value, list) and len(value) > 3",
        'isNone': 'np.all({atoms:s}.cell == [0, 0, 0])'
    },
    'momenta': {
        'call': "{atoms:s}.arrays['momenta']",
        'isvariable': "len(value.shape) > 2",
        'isNone': "'momenta' not in {atoms:s}.arrays"
    },
    'charges': {
        'call': "{atoms:s}.arrays['initial_charges']",
        'isvariable': "len(value.shape) > 1",
        'isNone': "'initial_charges' not in {atoms:s}.arrays"
    },
    'magmoms': {
        'call': "{atoms:s}.arrays['initial_magmoms']",
        'isvariable': "value.shape[-1] == len(self.paths[0, 0, :])",
        'isNone': "'initial_magmoms' not in {atoms:s}.arrays"
    },
    'calc': {
        'call': '{atoms:s}.calc',
        'isvariable': "type(value) == list",
        'isNone': '{atoms:s}.calc is None'
    },
    'pbc': {
        'call': '{atoms:s}.pbc',
        'isvariable': "False",
        'isNone': "np.all({atoms:s}.pbc == [False] * 3)"
    },
    'constraints': {
        'call': '{atoms:s}._constraints',
        'isvariable': "False",
        'isNone': '{atoms:s}._constraints == []'
    },
    'info': {
        'call': '{atoms:s}.info',
        'isvariable': "False",
        'isNone': "{atoms:s}.info == {{}}"
    }
}

necessary_parameters = {
    'Pk': {'default': '{P:d}', 'assert': '{name:s} > 0'},
    'prefix': {'default': "'paths'", 'assert': 'len({name:s}) > 0'},
    'directory': {'default': "'.'", 'assert': 'len({name:s}) > 0'},
    'id': {
        'default': "None",
        'assert': 'isinstance({name:s}, (int, np.int64))'
    },
    'candidates': {
        'default': "None",
        'assert': 'isinstance({name:s}, bool)'
    },
    'parents': {
        'default': "None",
        'assert': 'isinstance({name:s}, (tuple, str))'
    },
    'tag': {
        'default': "dict()",
        'assert': 'True'
    }
}

external_parameters = {
    'id': {
        'default': "None",
        'assert': 'isinstance({name:s}, (int, np.int64))'
    },
    'candidates': {
        'default': "None",
        'assert': 'isinstance({name:s}, bool)'
    },
    'parents': {
        'default': "None",
        'assert': 'isinstance({name:s}, (tuple, str))'
    },

}

class_objects = {
    'model': {'from': 'taps.model'},
    'finder': {'from': 'taps.pathfinder'},
    'prj': {'from': 'taps.projector'},
    'imgdata': {'from': 'taps.data'},
    'plotter': {'from': 'taps.plotter'}
}


def register_external_parameters(**parameters):
    for key in parameters.keys():
        assert key not in external_parameters
    external_parameters.update(parameters)


class Paths:
    """
    symbols : str or Atoms symbols object
    coords : coordinate representation of atomic configuration
    images : Atoms containing full cartesian coordinate
    prj : coordinate projector, function maps between reduced coordinate and
          full cartesian coordinate
    pk : fourier represenation of coords
    label : name

    """
    variables = {}
    invariants = {}

    def __init__(self, symbols, coords, label=None, Pk=None, model='Model',
                 finder='PathFinder', prj='Projector', imgdata='ImageData',
                 plotter='Plotter', tag=None, **kwargs):

        self.symbols = symbols
        self.coords = coords
        self.Pk = Pk
        self.label = label

        self._images = [Atoms(symbols) for i in range(self.P)]

        self.model = model
        self.finder = finder
        self.prj = prj
        self.imgdata = imgdata
        self.plotter = plotter
        self.tag = tag

        if 'calc' in kwargs:
            setattr(self, 'calc', kwargs['calc'])
            del kwargs['calc']

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key in ['symbols']:
            super().__setattr__(key, value)
        elif key in ['coords']:
            super().__setattr__(key, np.array(value).astype(float))
        elif key == 'Pk':
            if value is None:
                super().__setattr__(key, self.P - 2)
            else:
                super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        elif isinstance(getattr(type(self), key, None), (Images, property)):
            super().__setattr__(key, value)
        elif key in allowed_properties.keys():
            if eval(allowed_properties[key]['isvariable']):
                if key == 'calc' and isinstance(value[0], str):
                    calc = []
                    for val in value:
                        calc.append(get_calculator_class(value.lower())())
                    value = calc
                self.variables.update({key: value})
                super().__setattr__(key, value)
            else:
                if key == 'calc' and isinstance(value, str):
                    value = get_calculator_class(value.lower())()
                self.invariants[key] = value
                call = allowed_properties[key]['call']
                for i in range(self.P):
                    target = call.format(atoms='self._images[%d]' % i)
                    exec(target + ' = ' + 'copy.deepcopy(value)')
        elif key in necessary_parameters.keys():
            default = necessary_parameters[key]['default']
            assertion = necessary_parameters[key]['assert']
            if value is None:
                value = eval(default.format(P=self.P, N=self.N))
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
        elif key in self._prj.prj_parameters:
            setattr(self._prj, key, value)
        elif key in self._imgdata.imgdata_parameters:
            setattr(self._imgdata, key, value)
        elif key in self._plotter.plotter_parameters:
            setattr(self._plotter, key, value)
        elif 'model_' in key and key[6:] in self._model.model_parameters:
            setattr(self._model, key[6:], value)
        elif 'finder_' in key and key[7:] in self._finder.finder_parameters:
            setattr(self._finder, key[7:], value)
        elif 'prj_' in key and key[4:] in self._prj.prj_parameters:
            setattr(self._prj, key[4:], value)
        elif 'imgdata_' in key and \
             key[10:] in self._imgdata.imgdata_parameters:
            setattr(self._imgdata, key[10:], value)
        elif 'plotter_' in key and key[8:] in self._plotter.plotter_parameters:
            setattr(self._plotter, key[8:], value)
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
        elif 'prj_' in key and key[4:] in self._prj.prj_parameters:
            return self.__dict__['_prj'].__dict__[key[4:]]
        elif 'imgdata_' in key and \
             key[10:] in self._imgdata.imgdata_parameters:
            return self.__dict__['_imgdata'].__dict__[key[10:]]
        elif 'plotter_' in key and key[8:] in self._plotter.plotter_parameters:
            return self.__dict__['_plotter'].__dict__[key[8:]]
        elif key in self.__dict__['_plotter'].plotter_parameters:
            return self.__dict__['_plotter'].__dict__[key]
        # Short hand notation
        elif key in self.__dict__['_finder'].finder_parameters:
            return getattr(self.__dict__['_finder'], key)
        elif key in self.__dict__['_model'].model_parameters:
            # return getattr(self.__dict__['_model'].__dict__[key], key)
            return getattr(self.__dict__['_model'], key)
        elif key in self.__dict__['_imgdata'].imgdata_parameters:
            return self.__dict__['_imgdata'].__dict__[key]
        elif key in self.__dict__['_prj'].prj_parameters:
            return self.__dict__['_prj'].__dict__[key]
        else:
            raise AttributeError("Key called `%s` not exist" % key)

    @ImageIndexing
    def images(self, index):
        """
        try to use only for the potential calculation.
        """
        idx = np.arange(self.P)[index]
        full_coords = self.prj.inv(self.coords[:, :, idx])
        # coords inserting
        for i, ii in enumerate(idx):
            self._images[ii].positions = full_coords[:, :, i].T

        # Other variable inserting
        for key in self.variables:
            call = allowed_properties[key]['call']
            for i in idx:
                target = call.format(atoms='self._images[%d]' % i)
                exec(target + ' = ' + 'getattr(self, key)[%d]' % i)
        if len(idx) == 1:
            return self._images[idx[0]]
        return [self._images[i] for i in idx]

    @images.setter
    def images(self, index, images):
        idx = np.arange(self.P)[index].reshape(-1)
        full_coords = np.zeros((3, self.N, self.P))

        if isinstance(images, Atoms):
            images = [images]

        # coords setting
        for i, ii in enumerate(idx):
            self._images[ii].positions = images[i].positions
            full_coords[:, :, i] = images[i].positions.T

        self.coords[:, :, idx] = self.prj(full_coords)

        # other properties setting
        for key in self.variables:
            call = allowed_properties[key]['call']
            for i, ii in enumerate(idx):
                target = 'self.' + key + '[%d]' % ii
                value = call.format(atoms='images[%d]' % i)
                exec(target + ' = ' + value)

    @property
    def calc_parameters(self):
        if 'calc' in self.variables:
            return [calc.parameters for calc in self.calc]
        elif 'calc' in self.invariants:
            return self.invariants['calc'].parameters
        else:
            raise AttributeError('`calc` is not defined yet')

    @calc_parameters.setter
    def calc_parameters(self, value):
        if 'calc' in self.variables:
            for calc, parameters in zip(self.calc, value):
                calc.parameters = parameters
                calc.__init__(**parameters)
        elif 'calc' in self.invariants:
            self.invariants['calc'].parameters = value
            # Because Vasp need initialize the parameters we force to do it
            self.invariants['calc'].__init__(**value)
            for image in self._images:
                image.calc.parameters = value
                image.calc.__init__(**value)
        else:
            raise AttributeError('`calc` is not defined yet')

    @property
    def calc_results(self):
        if 'calc' in self.variables or self.invaraints:
            return [image.calc.results for image in self._images]
        else:
            raise AttributeError('`calc` is not defined yet')

    @calc_results.setter
    def calc_results(self, value):
        if 'calc' in self.variables:
            for calc, results in zip(self.calc, value):
                calc.results = results
        elif 'calc' in self.invariants:
            for image in self._images:
                image.calc.results = value
        else:
            raise AttributeError('`calc` is not defined yet')

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

    def get_displacements(self, index=np.s_[1:-1]):
        return self.prj.get_displacements(self, index=index)

    def get_momentum(self, index=np.s_[1:-1]):
        return self.prj.get_momentum(self, index=index)

    def get_kinetic_energy(self, index=np.s_[1:-1]):
        return self.prj.get_kinetic_energy(self, index=index)

    def get_kinetic_energy_gradient(self, index=np.s_[1:-1], reduced=None):
        if reduced is None:
            reduced = self.isreduced
        return self.prj.get_kinetic_energy_gradient(self, index=index,
                                                    reduced=reduced)

    def get_velocity(self, index=np.s_[1:]):
        return self.prj.get_velocity(self, index=index)

    def get_acceleration(self, index=np.s_[1:-1]):
        return self.prj.get_acceleration(self, index=index)

    def get_effective_mass(self, index=np.s_[1:-1]):
        return self.prj.get_effective_mass(self, index=index)

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

    def get_total_energy(self, index=np.s_[1:-1]):
        V = self.model.get_potential_energy(self, index=index)
        T = self.prj.get_kinetic_energy(self, index=index)
        return V + T

    def get_covariance(self, index=np.s_[1:-1]):
        cov_coords = self.model.get_covariance(self, index=np.s_[1:-1])
        cov_coords = np.diag(cov_coords).copy()
        cov_coords[cov_coords < 0] = 0
        sigma_f = self.model.hyperparameters.get('sigma_f', 1)
        return 1.96 * np.sqrt(cov_coords) / 2 / sigma_f

    def get_atoms_sample(self, index=None, coords=None):
        if index is None:
            index = [0]
        idx = np.arange(self.P)[index].reshape(-1)
        # Initialize
        atoms_sample = [Atoms(self.symbols) for i in range(len(idx))]

        # coords inserting
        if coords is None:
            coords = self.prj.inv(self.coords[:, :, idx])
        for i in range(len(idx)):
            atoms_sample[i].positions = coords[:, :, i].T

        # Other variable inserting
        for key in self.variables:
            call = allowed_properties[key]['call']
            for i in idx:
                target = call.format(atoms='atoms_sample[%d]' % i)
                exec(target + ' = ' + 'getattr(self, key)[%d]' % i)

        if len(idx) == 1:
            return atoms_sample[0]
        return atoms_sample

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
            coords = self.coords[..., index]
        ids = self.imgdata.add_data(self, coords, search_similar_image=True)
        if cache_model:
            self.model.add_data_ids(ids)
        return ids

    def simple_coords(self, index=np.s_[:]):
        coords = np.zeros(self.DMP)
        endpoint = self.coords[..., [0, -1]]
        for d in range(self.D):
            for m in range(self.M):
                coords[d, m] = np.linspace(*endpoint[d, m], self.P)
        return coords[..., index]

    def fluctuate(self, initialize=True, cutoff_f=70, temperature=0.03):
        if initialize:
            self.coords = self.simple_coords()
        fluc = np.zeros(self.Pk)
        fluc[:cutoff_f] = temperature * np.linspace(2 * self.P, 0, cutoff_f)
        self.rcoords += fluc * (np.random.rand(*self.rcoords.shape) - 0.5)

    def search(self, **kwargs):
        self.finder.search(self, **kwargs)

    @property
    def init_image(self):
        return self.coords[..., 0, np.newaxis]

    @property
    def rcoords(self):
        """ Recieprocal representation of a pathway
            Returns rcoords array
            D x M x (P - 2) array
        """
        # return dst(self.coords - self.init_image, type=4)[..., :self.Pk]
        ## return dst(self.coords[..., 1:-1], type=4)[..., :self.Pk]
        return dst(self.coords[..., 1:-1], type=1, norm='ortho')

    @rcoords.setter
    def rcoords(self, rcoords):
        """
            For numerical purpose, set rpath.
            D x M x (P - 2)  array
        """
        # P = self.P
        # coords = idst(rcoords, type=4, n=P) / (2 * P) + self.init_image
        # self.coords[..., 1:-1] = coords[..., 1:-1]
        ## P = self.P
        ## coords = idst(rcoords, type=4, n=P - 2) / (2 * (P - 2))
        ## self.coords[..., 1:-1] = coords
        self.coords[..., 1:-1] = idst(rcoords, type=1, norm='ortho')

    @property
    def symbols(self):
        """Get chemical symbols as a :class:`ase.symbols.Symbols` object.

        The object works like ``atoms.numbers`` except its values
        are strings.  It supports in-place editing."""
        return Symbols(self._numbers)

    @symbols.setter
    def symbols(self, value):
        self._numbers = Symbols.fromsymbols(value).numbers

    @property
    def P(self):
        return len(self.coords[0, 0, :])

    @property
    def NN(self):
        return len(self.coords[0, 0, :])

    @property
    def M(self):
        return len(self.coords[0, :])

    @property
    def D(self):
        # Deprecated
        return len(self.coords)

    @property
    def dim(self):
        return len(self.coords)

    @property
    def DD(self):
        return np.prod(self.coords.shape[:2])

    @property
    def N(self):
        return len(self._numbers)

    @property
    def A(self):
        # Number of Atoms
        return len(self._numbers)

    @property
    def DM(self):
        return self.coords.shape[:2]

    @property
    def DMP(self):
        return self.coords.shape

    @property
    def isreduced(self):
        D, M = self.DM
        return 3 * self.N != D * M

    @property
    def masses(self):
        return atomic_masses[self._numbers]

    def plot(self, filename=None, savefig=False, gaussian=False, **kwargs):
        self.plotter.plot(self, filename=filename, savefig=savefig,
                          gaussian=gaussian, **kwargs)

    def coords2images(self, coords):
        image = self._images[0].copy()
        calc = self._images[1].calc
        full_coords = self.prj.inv(coords=coords)
        images = []
        for positions in full_coords.T:
            image.positions = positions
            _image = image.copy()
            _image.calc = copy.deepcopy(calc)
            images.append(_image)
        if len(full_coords.T) == 1:
            return images[0]
        return images

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

    def to_traj(self, format='.traj'):
        from ase.io import Trajectory
        tr = Trajectory(self.label + '.traj', mode='w')
        for atoms in self.images[:]:
            tr.write(atoms)

    def to_csv(self, filename=None, save=True, save_calc=False,
               save_model=False, save_finder=False, save_prj=False,
               save_imgdata=False, save_plotter=False, format='csv',
               mode='w', header=True, index=None):
        """
        Every instance not starts with '_' will be saved.
           For example, value `self.symbols` will be saved but value
           `self._numbers` will not be saved. while saving, it will append the
           name so that it can tell where the parameter belong.
           If parameter is class object, only the name of it will be saved.
        """
        import pandas as pd
        if filename is None:
            filename = self.label
        if save == 'all':
            save_calc = True
            save_model = True
            save_finder = True
            save_prj = True
            save_imgdata = True
            save_plotter = True
        _ = paths2dct(self, necessary_parameters, allowed_properties,
                      class_objects, save_calc=save_calc, save_model=save_model,
                      save_finder=save_finder, save_prj=save_prj,
                      save_imgdata=save_imgdata, save_plotter=save_plotter)

        symbols, coords, dct = _
        pd_dct = dct2pd_dct(dct)

        df = pd.DataFrame({'symbols': str(symbols),
                           'coords': [coords.tolist()], **pd_dct}, index=index)
        old_label = filename
        self.label = filename
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.label = old_label
        if save and format == 'csv':
            if mode == 'w':
                df.to_csv(filename + '.' + format, mode=mode, header=header)
            elif mode == 'a':
                df.to_csv(filename + '.' + format, mode=mode, header=header,
                          index=index != [0])

            # elif mode == 'a':
            #     _df = pd.read_csv(filename + '.' + format)
            #     _df.append(df)
            #     _df.to_csv(filename + '.' + format, header=header)

        elif save and format == 'pkl':
            df.to_pickle(filename + '.' + format)
        else:
            NotImplementedError('Format %s not support' % format)
        return df

    def to_pickle(self, filename=None, simple=True):
        if simple:
            filename = filename or self.label + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(f, self)

    def copy(self, save_calc=True, save_model=True, save_finder=True,
             save_prj=True, save_imgdata=True, save_plotter=True,
             return_dict=False):
        """Return a copy."""
        return copy.deepcopy(self)
        _ = paths2dct(self, necessary_parameters, allowed_properties,
                      class_objects, save_calc=save_calc, save_model=save_model,
                      save_finder=save_finder, save_prj=save_prj,
                      save_imgdata=save_imgdata, save_plotter=save_plotter)
        if return_dict:
            return _
        symbols, coords, dct = _
        # return self.__class__(symbols, paths, **dct)
        return Paths(symbols, coords, **dct)
