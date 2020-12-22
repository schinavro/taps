
import copy
import numpy as np
from collections import OrderedDict

from taps.model.model import Model

from ase.atoms import Atoms
from ase.data import atomic_masses
# from ase.calculators.calculator import get_calculator_class
# from ase.symbols import Symbols


class AtomicModel(Model):
    implemented_properties = ['stresses', 'potential',
                              'gradients', 'positions', 'forces']

    calculation_properties = OrderedDict(
        stresses='{image}.get_stress()',
        potentials='{image}.get_potential_energies()',
        potential='{image}.get_potential_energy()',
        gradients='self.prj.f_inv(-{image}.get_forces().T[..., np.newaxis],'
                  ' {image}.positions.T[..., np.newaxis])[0][..., 0].T',
        positions='{image}.positions',
        forces='{image}.get_forces()'
    )
    model_parameters = {
        'image': {'default': 'None', 'assert': 'True'},
        'set_label': {'default': 'False', 'assert': 'True'}
    }

    def __init__(self, image=None, set_label=None, **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)

        self.image = image
        self.set_label = set_label

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        results_raw = {}
        if len(coords.shape) == 2:
            coords = coords[..., np.newaxis]
        for positions in coords.T:
            image = self.coord2image(positions)
            if False:
                image
            for property in self.calculation_properties.keys():
                if property not in properties:
                    continue
                if results_raw.get(property) is None:
                    results_raw[property] = []
                call = self.calculation_properties[property]
                results_raw[property].append(eval(call.format(image='image')))
        self.results = self.packaging(results_raw)

    def packaging(self, raw):
        results = {}
        for key, value in raw.items():
            if key == 'potentials':
                results[key] = np.array(value).T
            elif key == 'potential':
                results[key] = np.array(value)
            elif key == 'forces':
                results[key] = np.array(value).T
            elif key == 'positions':
                results[key] = np.array(value).T
            else:
                results[key] = np.array(value).T
        return results

    def get_mass(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        return atomic_masses[image.symbols.numbers, np.newaxis]

    def get_effective_mass(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        m = atomic_masses[image.symbols.numbers]
        return np.repeat(m, 3)[..., np.newaxis]

    def coord2image(self, coord=None, image=None, set_label=None):
        image = image or self.image
        set_label = set_label or self.set_label
        image.positions = coord
        if set_label:
            label = self.get_label(coord)
            image.calc.label = label
        return copy.deepcopy(image)

    @classmethod
    def Coord2image(cls, coord, label, **kwargs):
        image = Atoms(positions=coord, **kwargs)
        image.calc.label = label
        return copy.deepcopy(image)


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
