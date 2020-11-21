import copy
import hashlib
import binascii
import numpy as np
from collections import OrderedDict
from numpy import pi
from numpy import exp, dot
from numpy.linalg import norm
from scipy.optimize import check_grad
from ase.atoms import Atoms
from taps.utils.shortcut import isdct, isstr


class Model:
    implemented_properties = {'potential'}
    model_parameters = {
        'real_model': {'default': "None", 'assert': 'True'},
        'results': {'default': "dict()", 'assert': isdct},
        'directory': {'default': 'None', 'assert': isstr},
        'prefix': {'default': 'None', 'assert': isstr},
        'pbc': {'default': 'None', 'assert': 'True'},
        'potential_unit': {'default': '"eV"', 'assert': isstr},
        'data_ids': {'default': 'None', 'assert': isdct}
    }
    potential_unit = 'eV'
    name = 'Model'

    def __init__(self, results={}, label=None, prj=None,
                 prjf=None, **model_kwargs):
        OrderedDict
        # silence!
        self.results = results
        self.label = label
        self.prj = prj
        self.prjf = prjf
        self._cache = {}

        for key, value in model_kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key == 'real_model':
            if self.__class__.__name__ != 'Gaussian':
                super().__setattr__(key, self)
            else:
                super().__setattr__(key, value)
            #     from_ = 'taps.model'
            #     module = __import__(from_, {}, None, [value])
            #     value = getattr(module, value)()
        elif key == 'prj':
            if value is None:
                def value(x):
                    return x
            super().__setattr__(key, value)
        elif key == 'prjf':
            if value is None:
                def value(f, x):
                    return f
            super().__setattr__(key, value)
        elif key in self.model_parameters:
            attribute = self.model_parameters[key]
            default = attribute['default']
            assertion = attribute['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            if attribute.get('class', False) and type(value) == str:
                from_ = attribute['from']
                module = __import__(from_, {}, None, [value])
                value = getattr(module, value)()
            super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        elif isinstance(getattr(type(self), key, None), property):
            super().__setattr__(key, value)
        else:
            raise AttributeError('No key name %s allowed' % key)

    def __getattr__(self, key):
        if key == 'real_model':
            return self
        else:
            super().__getattribute__(key)

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

    def get_properties(self, paths, properties=['potential'],
                       index=np.s_[1:-1], coords=None, caching=False,
                       real_model=False, **kwargs):
        if type(properties) == str:
            properties = [properties]
        # For machine learning purpose we put real_model in model
        if real_model:
            model = self.real_model
        else:
            model = self
        for property in properties:
            if property not in model.implemented_properties:
                raise NotImplementedError('Can not calaculate %s' % property)
        if coords is None:
            idx = np.arange(paths.N)[index]
            coords = model.prj(paths.coords(idx))
        new_coords = None
        new_properties = []
        results = {}
        if caching:
            model._cache = getattr(model, '_cache', dict())
            cached_results = model._cache.get(coords.tobytes())
            if cached_results is not None:
                cached_properties = cached_results.keys()
                for property in properties:
                    if property in cached_properties:
                        results[property] = cached_results[property]
                    else:
                        new_properties.append(property)
            else:
                new_properties = list(properties)
                new_coords = coords
        else:
            new_properties = properties
            new_coords = coords

        if new_coords is not None:
            model.calculate(paths, new_coords, properties=new_properties,
                            **kwargs)

        for new_property in new_properties:
            new_result = model.results[new_property]
            if new_property == 'potential' and len(new_result) == 1:
                results[new_property] = new_result[0]
            elif new_result.shape[-1] == 1:
                results[new_property] = new_result[..., 0]
            else:
                results[new_property] = new_result

            if new_property == 'forces':
                positionss = model.results.get('positions', None)
                results['gradients'] = model.prjf(new_result, positionss)

        if caching:
            model._cache[coords.tobytes()] = copy.deepcopy(results)
        if len(properties) == 1:
            property = list(results.keys())[0]
            return results[property]
        return results

    def get_kinetics(self, paths, properties=['kinetic_energy'],
                     index=np.s_[1:-1], coords=None, caching=False,
                     real_model=False, **kwargs):
        if type(properties) == str:
            properties = [properties]
        # For machine learning purpose we put real_model in model
        if real_model:
            model = self.real_model
        else:
            model = self

        if coords is None:
            idx = np.arange(paths.N)[index]
            coords = model.prj(paths.coords(idx))
        required = {}
        for property in properties:
            if property in ['displacements']:
                required[property] = True
            elif property in ['velocity', 'kinetic_energy', 'momentum']:
                required['velocity'] = True
                if property in ['kinetic_energy', 'momentum']:
                    required['mass'] = True
            elif property in ['acceleration', 'kinetic_grad']:
                required['acceleration'] = True
                if property in ['kinetic_grad']:
                    required['mass'] = True
        if required.get('mass'):
            m = model.get_effective_mass(paths, coords=coords)
        if required.get('velocity'):
            v = coords.get_velocity()
        if required.get('acceleration'):
            a = coords.get_acceleration()
        if required.get('displacements'):
            d = coords.get_displacements()

        results = {}
        if 'velocity' in properties:
            results['velocity'] = v
        if 'acceleration' in properties:
            results['acceleration'] = a
        if 'displacement' in properties:
            results['displacements'] = d
        if 'kinetic_energy' in properties:
            axis = tuple(np.arange(len(v.shape) - 1))         # 0     or (0, 1)
            results['kinetic_energy'] = np.sum(0.5 * m * v * v, axis=axis)
        if 'momentum' in properties:
            results['momentum'] = m * v
        if 'kinetic_grad' in properties:
            results['kinetic_energy_gradient'] = m * a
        return results

    def get_potential(self, paths, **kwargs):
        return self.get_properties(paths, properties='potential', **kwargs)

    def get_potentials(self, paths, **kwargs):
        return self.get_properties(paths, properties='potentials', **kwargs)

    def get_potential_energy(self, paths, **kwargs):
        return self.get_properties(paths, properties='potential', **kwargs)

    def get_potential_energies(self, paths, **kwargs):
        return self.get_properties(paths, properties='energies', **kwargs)

    def get_forces(self, paths, **kwargs):
        return self.get_properties(paths, properties='forces', **kwargs)

    def get_gradient(self, paths, **kwargs):
        return self.get_properties(paths, properties='gradients', **kwargs)

    def get_gradients(self, paths, **kwargs):
        return self.get_properties(paths, properties='gradients', **kwargs)

    def get_hessian(self, paths, **kwargs):
        return self.get_properties(paths, properties='hessian', **kwargs)

    def get_velocity(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='velocity', **kwargs)

    def get_acceleration(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='acceleration', **kwargs)

    def get_displacements(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='displacements', **kwargs)

    def get_momentum(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='momentum', **kwargs)

    def get_kinetic_energy(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='kinetic_energy', **kwargs)

    def get_kinetic_energy_gradient(self, paths, **kwargs):
        return self.get_kinetics(paths, properties='kinetic_grad', **kwargs)

    def generate_unique_hash(self, positions):
        """return string that explains current calculation
        """
        dk = hashlib.pbkdf2_hmac('sha512', b"Heyy!C@reful!ThisisMyP@ssword!",
                                 positions.tobytes(), 1234)
        return str(binascii.hexlify(dk))[2:-1]

    def get_labels(self, coords=None):
        labels = []
        directory = self.directory or ''
        prefix = self.prefix or ''

        for positions in coords.T:
            unique_hash = self.generate_unique_hash(positions)
            labels.append(directory + '/' + unique_hash + '/' + prefix)
        return labels

    def check_grad(self, paths=None, index=np.s_[:], epsilon=1e-4,
                   debug=False):
        def prind(*args):
            if debug:
                print(*args)

        def V(paths, index=None):
            shape = paths.coords[..., index].shape

            def f(p):
                paths.coords[..., index] = p.reshape(shape)
                E = self.get_potential_energy(paths, index=index)
                prind('E', p, E.sum())
                return E.sum()
            return f

        def dV(paths, index=None):
            shape = paths.coords[..., index].shape

            def f(p):
                paths.coords[..., index] = p.reshape(shape)
                F = self.get_forces(paths, index=index).flatten()
                prind('F', p, F)
                return -F
            return f

        paths0 = paths.coords[..., index].copy()
        print(check_grad(V(paths, index), dV(paths, index),
                         paths0.flatten(), epsilon=epsilon))
        paths.coords[..., index] = paths0.copy()

    def check_hess(self, paths=None, index=None, epsilon=1e-4,
                   debug=False):
        if index is None:
            index = paths.P // 2
        assert isinstance(index, (int, np.int64)), 'Index should be integer'

        def prind(*args):
            if debug:
                print(*args)

        def dV(paths, index=None, d=None, m=None):
            shape = paths.coords[..., index].shape

            def f(p):
                paths.coords[..., index] = p.reshape(shape)
                F = self.get_forces(paths, index=index)
                prind('F', F.flatten())
                return -F[d, m].sum()
            return f

        def ddV(paths, index=None, d=None, m=None):
            H = self.get_hessian(paths, index=index)
            prind('H', H.flatten())

            def f(p):
                return H[:, :, d, m].flatten()
            return f

        paths0 = paths.coords[..., index].copy()
        err = 0
        for m in range(paths.M):
            for d in range(paths.D):
                err += check_grad(dV(paths, index, d, m),
                                  ddV(paths, index, d, m),
                                  paths0.flatten(), epsilon=epsilon)
            paths.coords[..., index] = paths0.copy()
        prind('Total hessian error :', err)

    def add_data_ids(self, ids, overlap_handler=True):
        """
        ids : dict of list
        """
        if getattr(self, 'data_ids', None) is None:
            self.data_ids = dict()
        for table_name, id in ids.items():
            if self.data_ids.get(table_name) is None:
                self.data_ids[table_name] = []
            if overlap_handler:
                for i in id:
                    if i not in self.data_ids[table_name]:
                        self.data_ids[table_name].append(i)
        self.optimized = False


class PeriodicModel(Model):
    name2tag = {
        'HC1': 11, 'H': 12, 'H11': 13, 'HC2': 14, 'H12': 15, 'O': 8, 'CT1': 61,
        'C': 62, 'CT2': 63, 'CT3': 64, 'CT4': 65, 'N': 7
    }
    tag2name = {v: k for k, v in name2tag.items()}
    name = [
        'HC1', 'CT1', 'HC1', 'HC1', 'C', 'O', 'N', 'H', 'CT2', 'H11', 'CT3',
        'HC2', 'HC2', 'HC2', 'C', 'O', 'N', 'H', 'CT4', 'H12', 'H12', 'H12'
    ]
    name2c = {
        "HC1": 0.1437, "CT1": -0.5188, "C": 0.6731, "HC2": 0.0425, "O": -0.5854,
        "H": 0.3018, "CT2": 0.0448, "H11": 0.0228, "CT3": -0.0909, "N": -0.4937,
        "CT4": -0.0076, "H12": 0.0665}
    bonds_idx = [
        (2, 3), (2, 4), (1, 2), (11, 12), (11, 13), (11, 14), (9, 10), (7, 8),
        (19, 20), (19, 21), (19, 22), (17, 18), (5, 6), (5, 7), (2, 5), (7, 9),
        (15, 16), (15, 17), (9, 11), (9, 15), (17, 19)
    ]
    theta_idx = [
        (5, 7, 8), (4, 2, 5), (3, 2, 4), (3, 2, 5), (1, 2, 3), (1, 2, 4),
        (1, 2, 5), (15, 17, 18), (13, 11, 14), (12, 11, 13), (12, 11, 14),
        (10, 9, 11), (10, 9, 15), (9, 11, 12), (9, 11, 13), (9, 11, 14),
        (8, 7, 9), (7, 9, 10), (21, 19, 22), (20, 19, 21), (20, 19, 22),
        (18, 17, 19), (17, 19, 20), (17, 19, 21), (17, 19, 22), (6, 5, 7),
        (5, 7, 9), (2, 5, 6), (2, 5, 7), (16, 15, 17), (15, 17, 19),
        (11, 9, 15), (9, 15, 16), (9, 15, 17), (7, 9, 11), (7, 9, 15)
    ]
    dihedral_idx = [
        (6, 5, 7, 8), (2, 7, 5, 6), (4, 2, 5, 6), (4, 2, 5, 7), (3, 2, 5, 6),
        (3, 2, 5, 7), (2, 5, 7, 8), (1, 2, 5, 6), (1, 2, 5, 7), (5, 7, 9, 10),
        (15, 17, 19, 20), (15, 17, 19, 21), (15, 17, 19, 22), (14, 11, 9, 15),
        (13, 11, 9, 15), (12, 11, 9, 15), (16, 15, 17, 18), (10, 9, 11, 12),
        (10, 9, 11, 13), (10, 9, 11, 14), (10, 9, 15, 16), (10, 9, 15, 17),
        (9, 15, 17, 18), (8, 7, 9, 10), (8, 7, 9, 11), (8, 7, 9, 15),
        (7, 9, 11, 12), (7, 9, 11, 13), (7, 9, 11, 14), (18, 17, 19, 20),
        (18, 17, 19, 21), (18, 17, 19, 22), (5, 9, 7, 8), (15, 19, 17, 18),
        (6, 5, 7, 9), (5, 7, 9, 11), (5, 7, 9, 15), (2, 5, 7, 9),
        (16, 15, 17, 19), (11, 9, 15, 16), (11, 9, 15, 17), (9, 15, 17, 19),
        (7, 9, 15, 16), (7, 9, 15, 17), (9, 17, 15, 16)
    ]
    bond_param = {
        ('CT1', 'HC1'): (340.0, 1.09), ('CT3', 'HC2'): (340.0, 1.09),
        ('CT2', 'H11'): (340.0, 1.09), ('H', 'N'): (434.0, 1.01),
        ('CT4', 'H12'): (340.0, 1.09), ('C', 'O'): (570.0, 1.22),
        ('C', 'N'): (490.0, 1.33), ('C', 'CT1'): (317.0, 1.52),
        ('CT2', 'CT3'): (310.0, 1.52), ('C', 'CT2'): (317.0, 1.52),
        ('CT2', 'N'): (337.0, 1.44), ('CT4', 'N'): (337.0, 1.44)
    }
    angle_param = {
        ('HC2', 'CT3', 'HC2'): (35.0, 109.5), ('CT2', 'C', 'N'): (70.0, 116.6),
        ('HC1', 'CT1', 'HC1'): (35.0, 109.5), ('C', 'N', 'H'): (50.0, 120.0),
        ('CT3', 'CT2', 'H11'): (50.0, 109.5), ('N', 'C', 'O'): (80.0, 122.9),
        ('CT2', 'CT3', 'HC2'): (50.0, 109.5), ('CT2', 'N', 'H'): (50.0, 118.0),
        ('H11', 'CT2', 'N'): (50.0, 109.5), ('C', 'N', 'CT2'): (50.0, 121.9),
        ('CT4', 'N', 'H'): (50.0, 118.0), ('H12', 'CT4', 'N'): (50.0, 109.5),
        ('CT1', 'C', 'O'): (80.0, 120.4), ('C', 'CT2', 'H11'): (50.0, 109.5),
        ('H12', 'CT4', 'H12'): (35.0, 109.5), ('CT1', 'C', 'N'): (70.0, 116.6),
        ('C', 'N', 'CT4'): (50.0, 121.9), ('C', 'CT2', 'CT3'): (63.0, 111.1),
        ('CT2', 'C', 'O'): (80.0, 120.4), ('C', 'CT1', 'HC1'): (50.0, 109.5),
        ('CT3', 'CT2', 'N'): (80.0, 109.7), ('C', 'CT2', 'N'): (63.0, 110.1)
    }
    dihedral_param = {
        ('O', 'C', 'N', 'H'): (1.25, 2.0, 180.000077144),
        ('C', 'N', 'CT2', 'H11'): (0.0, 2.0, 0.0),
        ('HC1', 'CT1', 'C', 'O'): (0.0, 2.0, 0.0),
        ('HC1', 'CT1', 'C', 'N'): (0.0, 2.0, 0.0),
        ('CT1', 'C', 'N', 'H'): (1.25, 2.0, 180.000077144),
        ('C', 'N', 'CT4', 'H12'): (0.0, 2.0, 0.0),
        ('HC2', 'CT3', 'CT2', 'C'): (0.0777778, 3.0, 0.0),
        ('H11', 'CT2', 'CT3', 'HC2'): (0.0777778, 3.0, 0.0),
        ('H11', 'CT2', 'C', 'O'): (0.04, 3.0, 180.000077144),
        ('H11', 'CT2', 'C', 'N'): (0.0, 2.0, 0.0),
        ('CT2', 'C', 'N', 'H'): (1.25, 2.0, 180.000077144),
        ('H', 'N', 'CT2', 'H11'): (0.0, 2.0, 0.0),
        ('H', 'N', 'CT2', 'CT3'): (0.0, 2.0, 0.0),
        ('H', 'N', 'CT2', 'C'): (0.0, 2.0, 0.0),
        ('N', 'CT2', 'CT3', 'HC2'): (0.0777778, 3.0, 0.0),
        ('H', 'N', 'CT4', 'H12'): (0.0, 2.0, 0.0),
        ('O', 'C', 'N', 'CT2'): (1.25, 2.0, 180.000077144),
        ('C', 'N', 'CT2', 'CT3'): (0.25, 4.0, 180.000077144),
        ('C', 'N', 'CT2', 'C'): (0.425, 2.0, 180.000077144),
        ('CT1', 'C', 'N', 'CT2'): (1.25, 2.0, 180.000077144),
        ('O', 'C', 'N', 'CT4'): (1.25, 2.0, 180.000077144),
        ('CT3', 'CT2', 'C', 'O'): (0.0, 2.0, 0.0),
        ('CT3', 'CT2', 'C', 'N'): (0.05, 4.0, 0.0),
        ('CT2', 'C', 'N', 'CT4'): (1.25, 2.0, 180.000077144),
        ('N', 'CT2', 'C', 'O'): (0.0, 2.0, 0.0),
        ('N', 'CT2', 'C', 'N'): (1.0, 2.0, 180.000077144),
        ('C', 'CT2', 'N', 'H'): (0.55, 2.0, 180.000077144),
        ('C', 'CT4', 'N', 'H'): (0.55, 2.0, 180.000077144),
        ('CT1', 'N', 'C', 'O'): (5.25, 2.0, 180.0),
        ('CT2', 'N', 'C', 'O'): (5.25, 2.0, 180.000077144)
    }
    LJ_param = {
        'HC1': (0.0157, 1.487), 'CT1': (0.1094, 1.908), 'C': (0.086, 1.908),
        'O': (0.21, 1.6612), 'N': (0.17, 1.824), 'H': (0.0157, 0.6),
        'CT2': (0.1094, 1.908), 'H11': (0.0157, 1.387), 'CT3': (0.1094, 1.908),
        'HC2': (0.0157, 1.487), 'CT4': (0.1094, 1.908), 'H12': (0.0157, 1.387)
    }

    def V(self, atoms):
        atoms.set_initial_charges([self.name2c[n] for n in self.name])
        atoms.set_tags([self.name2tag[n] for n in self.name])

        bond_param = self.bond_param
        tag2name = self.tag2name
        bonds_idx = self.bonds_idx
        theta_idx = self.theta_idx
        dihedral_idx = self.dihedral_idx
        angle_param = self.angle_param
        dihedral_param = self.dihedral_param
        LJ_param = self.LJ_param

        def Vb(atoms):
            Vb = 0
            for i, j in bonds_idx:
                i -= 1
                j -= 1
                b = norm(atoms[i].position - atoms[j].position)
                name_pair = [tag2name[atoms[i].tag], tag2name[atoms[j].tag]]
                name_pair.sort()
                Kb, b0 = bond_param[tuple(name_pair)]
                Vb += Kb * (b - b0) ** 2
            return Vb

        def Vangle(atoms):
            Vt = 0
            for i, j, k in theta_idx:
                i -= 1
                j -= 1
                k -= 1
                ba = atoms[i].position - atoms[j].position
                bc = atoms[k].position - atoms[j].position
                costheta = dot(ba, bc) / (norm(ba) * norm(bc))
                theta = np.arccos(costheta)
                name_triple = [tag2name[atoms[i].tag], tag2name[atoms[k].tag]]
                name_triple.sort()
                ordered_triple = (name_triple[0],
                                  tag2name[atoms[j].tag],
                                  name_triple[1])
                Kt, theta0 = angle_param[ordered_triple]
                Vt += Kt * (theta - theta0 * pi / 180) ** 2
            return Vt

        def Vdihedral(atoms):
            Vd = 0
            for i, j, k, l in dihedral_idx:
                i -= 1
                j -= 1
                k -= 1
                l -= 1
                # Calculate dihedral angle phi
                b0 = -1.0 * (atoms[j].position - atoms[i].position)
                b1 = atoms[k].position - atoms[j].position
                b2 = atoms[l].position - atoms[k].position
                b1 /= norm(b1)
                v = b0 - dot(b0, b1) * b1
                w = b2 - dot(b2, b1) * b1
                x = dot(v, w)
                y = dot(np.cross(b1, v), w)
                phi = np.arctan2(y, x)
                Kphi, n, delta = dihedral_param[(tag2name[atoms[i].tag],
                                                 tag2name[atoms[j].tag],
                                                 tag2name[atoms[k].tag],
                                                 tag2name[atoms[l].tag])]
                Vd += Kphi * (np.cos(n * phi - delta * np.pi / 180) + 1)
            return Vd

        def VcoulombLJ(atoms):
            eps = 7.  # Water
            N = len(atoms)
            VLJ = 0
            Vc = 0
            for i in range(N):
                for j in range(i + 1, N):
                    pair = [tag2name[atoms[i].tag], tag2name[atoms[j].tag]]
                    pair.sort()
                    if tuple(pair) in bond_param.keys():
                        continue
                    elif i == 5 and j == N - 5:
                        continue
                    rij = np.linalg.norm(atoms[i].position - atoms[j].position)
                    epsi, rimin = LJ_param[tag2name[atoms[i].tag]]
                    epsj, rjmin = LJ_param[tag2name[atoms[j].tag]]
                    # Lennard-Johns
                    epsij = np.sqrt(epsi * epsj)
                    rijmin = (rimin + rjmin) / 2
                    VLJ += epsij * ((rijmin / rij)**12 - 2 * (rijmin / rij)**6)
                    # Coulomb
                    qiqj = atoms[i].charge * atoms[j].charge
                    Vc += eps * qiqj / rij
            return Vc + VLJ
        V = Vb(atoms) + Vangle(atoms) + Vdihedral(atoms) + VcoulombLJ(atoms)
        return -V

    def F2(self, atoms, dX=[0.01, 0, 0], dY=[0, 0.01, 0], dZ=[0, 0, 0.01]):
        V = self.V(atoms)
        _atoms = atoms.copy()
        F = []
        for atom in _atoms:
            dVdR = []
            for dR in [dX, dY, dX]:
                atom.position += np.array(dR)
                dVdR.append((V - self.V(_atoms)) / np.linalg.norm(dR))
                atom.position -= np.array(dR)
            F.append(dVdR)
        return F

    def F(self, atoms, dphi=0.001, dpsi=0.001):
        V = self.V(atoms)
        if V == -12.5:
            return np.zeros((1, 2))
        phi = atoms.get_dihedral(4, 6, 8, 14)
        psi = atoms.get_dihedral(6, 8, 14, 16)
        phi_atoms = atoms.copy()
        psi_atoms = atoms.copy()
        phimask = [True] * 8 + [False] * 14
        psimask = [False] * 14 + [True] * 8
        phi_atoms.set_dihedral(14, 8, 6, 4, angle=phi + dphi, mask=phimask)
        F_phi = (self.V(phi_atoms) - V) / dphi * 180 / np.pi
        psi_atoms.set_dihedral(6, 8, 14, 16, angle=psi + dpsi, mask=psimask)
        F_psi = (self.V(psi_atoms) - V) / dpsi * 180 / np.pi
        return -np.array([[F_phi, F_psi]])

    def get_potential_energy(self, paths, coords=None, index=np.s_[1:-1]):
        trj = paths.images[index]
        if type(trj) == Atoms:
            trj = [trj]
        if coords is not None:
            init = paths._images[0]
            positions = paths.prj.inv(coords).T
            trj = [Atoms(init, positions=pos) for pos in positions]
        E = np.array([self.V(atoms) for atoms in trj])
        return E - 100

    def get_forces(self, paths, coords=None, index=np.s_[1:-1], reduced=None):
        trj = paths.images[index]
        if type(trj) == Atoms:
            trj = [trj]
        if coords is not None:
            init = paths._images[0]
            positions = paths.prj.inv(coords).T
            trj = [Atoms(init, positions=pos) for pos in positions]
        F = np.array([self.F(atoms).T for atoms in trj])
        return F.T
        # return np.zeros((1, 2, len(trj)))

    def get_chi_psi(self, p):
        # Tuple of length P, P
        phi = self.get_dihedral(p, 4, 6, 8, 14)
        psi = self.get_dihedral(p, 6, 8, 14, 16)
        phi[phi > np.pi] -= 2 * np.pi
        psi[psi > np.pi] -= 2 * np.pi
        return phi, psi

    def get_dihedral(self, p, a, b, c, d):
        # vector 1->2, 2->3, 3->4 and their normalized cross products:
        # p : 3 x N x P
        v_a = p[:, b, :] - p[:, a, :]  # 3 x P
        v_b = p[:, c, :] - p[:, b, :]
        v_c = p[:, d, :] - p[:, c, :]
        bxa = np.cross(v_b.T, v_a.T).T  # 3 x P
        cxb = np.cross(v_c.T, v_b.T).T
        bxanorm = np.linalg.norm(bxa, axis=0)  # P
        cxbnorm = np.linalg.norm(cxb, axis=0)
        if np.any(bxanorm == 0) or np.any(cxbnorm == 0):
            raise ZeroDivisionError('Undefined dihedral angle')
        bxa /= bxanorm  # 3 x P
        cxb /= cxbnorm
        angle = np.sum(bxa * cxb, axis=0)  # P
        # check for numerical trouble due to finite precision:
        angle[angle < -1] = -1
        angle[angle > 1] = 1
        angle = np.arccos(angle)
        reverse = np.sum(bxa * v_c, axis=0) > 0
        angle[reverse] = 2 * np.pi - angle[reverse]
        return angle.flatten()


class PeriodicModel2(Model):
    implemented_properties = {'potential', 'gradients', 'hessian'}

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        V = -129.7 + 0.1 * np.sin(4 * coords).sum(axis=(0, 1))
        dV = 0.1 * 4 * np.cos(4 * coords)
        coords = np.atleast_3d(coords)
        D, M, P = coords.shape
        _coords = coords.reshape(D * M, P)
        H = np.zeros((D * M, P))  # DM x P
        H[0] = -(0.1 * 16) * np.sin(4 * _coords[0])
        H[1] = -(0.1 * 16) * np.sin(4 * _coords[1])
        H = np.einsum('i..., ij->ij...', H, np.identity(D * M))
        H = H.reshape((D, M, D, M, P))

        if np.isscalar(V):
            V = np.array([V])
        self.results['potential'] = V
        self.results['gradients'] = dV
        self.results['hessian'] = H


class PeriodicModel3(Model):
    implemented_properties = {'potential', 'gradients', 'forces', 'hessian'}

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        if 'potential' in properties:
            V = np.cos(4 * coords).sum(axis=(0, 1))
            self.results['potential'] = V
        if 'gradients' in properties or 'forces' in properties:
            F = 4 * np.sin(4 * coords)
            if 'gradients' in properties:
                self.results['gradients'] = -F
            if 'forces' in properties:
                self.results['forces'] = F

        if 'hessian' in properties:
            coords = np.atleast_3d(coords)
            D, M, P = coords.shape
            _coords = coords.reshape(D * M, P)
            H = np.zeros((D * M, P))  # DM x P
            H = -16 * np.cos(4 * _coords)
            H = np.einsum('i..., ij->ij...', H, np.identity(D * M))
            H = H.reshape((D, M, D, M, P))
            self.results['hessian'] = H


class FlatModel(Model):
    implemented_properties = {'potential', 'forces', 'hessian'}

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        d, A, N = coords.shape
        V = -129.5 + np.zeros(N)
        F = (np.random.normal(0, 0.1, d * A * N)).reshape(coords.shape)
        H = np.zeros((d, A, d, A, N))

        self.results['potential'] = V
        self.results['gradients'] = F
        self.results['hessian'] = H
