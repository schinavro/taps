from taps.model.model import Model
from taps.utils.utils import ImageIndexing

from ase.atoms import Atoms
from ase.data import atomic_masses
from ase.calculators.calculator import get_calculator_class
from ase.symbols import Symbols

class AtomicModel(Model):

    variables = {}
    invariants = {}

    atomic_properties = {
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

    implemented_properties = {'label', 'positions', 'gradients', 'potential',
                              'forces'}
    model_parameters = {
        'projector': {'default': 'None', 'assert': 'True'},
        'enable_label': {'default': 'None', 'assert': 'True'},
        'calc': {'default': 'None', 'assert': 'True'}
    }

    def __init__(self, symbols=None, projector=None, enable_label=True,
                 **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)
        self.symbols = symbols
        self.projector = projector
        self.enable_label = enable_label

        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        if key in self.atomic_properties.keys():
            if eval(self.atomic_properties[key]['isvariable']):
                if key == 'calc' and isinstance(value[0], str):
                    calc = []
                    for val in value:
                        calc.append(self.get_calculator_class(value.lower())())
                    value = calc
                self.variables.update({key: value})
                super().__setattr__(key, value)
            else:
                if key == 'calc' and isinstance(value, str):
                    value = self.get_calculator_class(value.lower())()
                self.invariants[key] = value
                call = self.atomic_properties[key]['call']
                for i in range(self.P):
                    target = call.format(atoms='self._images[%d]' % i)
                    exec(target + ' = ' + 'copy.deepcopy(value)')
        else:
            super().__setattr__(key, value)

    @property
    def symbols(self):
        """Get chemical symbols as a :class:`ase.symbols.Symbols` object.

        The object works like ``atoms.numbers`` except its values
        are strings.  It supports in-place editing."""
        return self.Symbols(self._numbers)

    @symbols.setter
    def symbols(self, value):
        self._numbers = self.Symbols.fromsymbols(value).numbers

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

    def to_traj(self, format='.traj'):
        from ase.io import Trajectory
        tr = Trajectory(self.label + '.traj', mode='w')
        for atoms in self.images[:]:
            tr.write(atoms)


    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        meta = ['label', 'positions', 'gradients']
        images = paths.coords2images(coords)
        labels = self.get_labels(coords)
        properties = list(properties)
        calculation_properties = [v for v in properties if v not in meta]
        meta_properties = [v for v in properties if v in meta]
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        try:
            # Stress -> potentials -> forces -> potential
            i = properties.index('potentials')
            properties[0], properties[i] = properties[i], properties[0]
        except ValueError:
            pass
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
        results = {}
        if type(images) == Atoms:
            images = [images]
        for image, label in zip(images, labels):
            if self.enable_label:
                image.calc.label = label
            for property in calculation_properties:
                results[property] = results.get(property, [])
                if property == 'potentials':
                    results[property].append(image.get_potential_energies())
                elif property == 'potential':
                    results[property].append(image.get_potential_energy())
                else:
                    results[property].append(
                        image.calc.get_property(property, atoms=image))
            for property in meta_properties:
                results[property] = results.get(property, [])
                if property == 'label':
                    results['label'].append(label)
                elif property == 'gradients':
                    results['forces'] = results.get('forces', [])
                    results['forces'].append(image.get_forces())
                elif property == 'positions':
                    results[property].append(image.positions)
            if 'positions' not in calculation_properties and \
                    'positions' not in meta_properties:
                results['positions'] = results.get('positions', [])
                results['positions'].append(image.positions)
        self.results = self.concatenate_arr(results)
        if 'gradients' in meta_properties:
            positions = self.results['positions']
            forces = self.results['forces']
            self.results['gradients'] = paths.prj(positions.T, forces)

    def concatenate_arr(self, results_list):
        results = {}
        for key, value in results_list.items():
            if key == 'potentials':
                results[key] = np.array(value).T
            elif key == 'potential':
                results[key] = np.array(value)
            elif key == 'forces':
                results[key] = np.array(value).T
            elif key == 'positions':
                results[key] = np.array(value)
        return results

    def update_implemented_properties(self, paths):
        image = paths.coords2images(paths.coords[..., 0])
        calculatable_properties = image.calc.implemented_properties
        self.implemented_properties.update(set(calculatable_properties))

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


class AlanineDipeptide(AtomicModel):

    def get_labels2(self, paths, idx=None, coords=None):
        if coords is None:
            coords = paths.coords.copy()
        if self.label is None:
            self.label = paths.label
        directory = self.directory
        prefix = self.prefix
        conformation = paths.plotter.conformation
        translation = paths.plotter.translation
        labels = []
        for i in idx:
            ang = (coords[0, :, i] % (2 * np.pi) + translation) * conformation
            ang[ang > 180.] -= 360.
            _form = 5 + (ang < 0)
            meta = '{0:0{2}.1f}_{1:0{3}.1f}'.format(*ang, *_form)
            name = prefix + meta
            labels.append(directory + '/' + name + '/' + name)
        return labels


class AlanineDipeptide3D(AtomicModel):

    def get_labels2(self, paths, idx=None, coords=None):
        if coords is None:
            coords = paths.coords.copy()
        if self.label is None:
            self.label = paths.label
        directory = self.directory
        prefix = self.prefix
        conformation = paths.plotter.conformation
        translation = paths.plotter.translation
        labels = []
        for i in idx:
            # ang = (coords[0, :, i] % (2 * np.pi) + translation) * conformation
            ang = idx
            ang[ang > 180.] -= 360.
            _form = 5 + (ang < 0)
            meta = '{0:0{2}.1f}_{1:0{3}.1f}'.format(*ang, *_form)
            name = prefix + meta
            labels.append(directory + '/' + name + '/' + name)
        return labels
