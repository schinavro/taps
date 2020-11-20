import numpy as np
from numpy import cos, sin, sum, cross, repeat, concatenate, newaxis
from numpy import newaxis as nax
from numpy import atleast_3d
from numpy.linalg import norm
from ase.data import atomic_masses
# from ase.wyckoff.wyckoff import Wyckoff
# from ase.wyckoff.xtal2 import parse_wyckoff_site


class Projector:
    prj_parameters = {
        'dt': {
            'default': '0.1',
            'assert': '{name:s} > 0'
        },
        'mass_type': {
            'default': '"invariant"',
            'assert': 'True'
        },
        'moving_average': {
            'default': '2',
            'assert': 'isinstance({name:s}, (int, np.int64))'
        },
        'time_unit': {
            'default': '"ang / sqrt(u / eV)"',
            'assert': 'isinstance({name:s}, str)'
        },
        'mass_unit': {
            'default': '"u"',
            'assert': 'isinstance({name:s}, str)'
        },
        'distance_unit': {
            'default': '"Ang"',
            'assert': 'isinstance({name:s}, str)'
        },
        'kinetic_unit': {
            'default': '"eV"',
            'assert': 'isinstance({name:s}, str)'
        }
    }
    dt = 0.1
    moving_average = 2
    mass_type = "invariant"
    kinetic_unit = 'eV'

    def __init__(self, X=None, F=None, X_inv=None, F_inv=None,
                 **kwargs):
        self.X = X
        self.F = F
        self.X_inv = X_inv
        self.F_inv = F_inv
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, positions=None, forces=None, coords=None):
        if positions is not None and forces is not None:
            return self.F(positions, forces)
        elif positions is not None:
            return self.X(positions)
        elif coords is not None and forces is not None:
            return self.F_inv(coords, forces)
        elif coords is not None:
            return self.X_inv(coords)
        else:
            raise AttributeError('please sepcify `positions` or `coords`')

    def __setattr__(self, key, value):
        if key in ['X', 'F', 'X_inv', 'F_inv']:
            if value is None and 'X' in key:
                value = self.default_X
            elif value is None and 'F' in key:
                value = self.default_F
            super().__setattr__(key, value)
        elif key[0] == '_':
            super().__setattr__(key, value)
        elif key in self.prj_parameters.keys():
            default = self.prj_parameters[key]['default']
            assertion = self.prj_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), assertion
            if key == 'moving_average':
                if value == 2:
                    velocity = self.two_points_average_velocity
                    gradient = self.two_points_velocity_gradient
                    acceleration = self.three_points_acceleration
                elif value == 3:
                    velocity = self.three_points_average_veolicty
                    gradient = self.three_points_velocity_gradient
                    acceleration = self.three_pointse_acceleration
                elif value == 4:
                    velocity = self.four_points_average_veolicty
                    gradient = self.four_points_velocity_gradient
                    acceleration = self.four_points_acceleration
                else:
                    NotImplementedError('Select `2` or `3` with %s' % key)
                super().__setattr__('_get_velocity', velocity)
                super().__setattr__('_get_velocity_gradient', gradient)
                super().__setattr__('_get_acceleration', acceleration)
            super().__setattr__(key, value)
        else:
            raise AttributeError('No such key can be justified, %s' % key)

    def inv(self, coords=None, forces=None):
        if coords is not None and forces is not None:
            return self.F_inv(coords, forces)
        elif coords is not None:
            return self.X_inv(coords)
        else:
            raise NotImplementedError('please sepcify `coords`')

    def get_displacements(self, paths, index=np.s_[:], coords=None):
        p = paths.coords
        if coords is not None:
            p = coords
        d = norm(np.diff(p), axis=0).sum(axis=0)
        d = concatenate([[0], d])
        return np.add.accumulate(d)[index]

    @property
    def get_velocity(self):
        if self.__dict__.get('_get_velocity') is None:
            return self.two_points_average_velocity
        return self._get_velocity

    @property
    def get_acceleration(self):
        if self.__dict__.get('_get_acceleration') is None:
            return self.three_points_acceleration
        return self._get_acceleration

    @property
    def get_velocity_gradient(self):
        if self.__dict__.get('_get_velocity_gradient') is None:
            return self.two_points_velocity_gradient
        return self._get_velocity_gradient

    def two_points_average_velocity(self, paths, index=np.s_[:], coords=None):
        """
        Returns Dim x M x P array
        """
        p = paths.coords.copy()
        dt = self.dt
        if coords is not None:
            p = coords
        if index == np.s_[:]:
            p = concatenate([p, p[..., -1, nax]], axis=2)
            return (p[..., 1:] - p[..., :-1]) / dt
        elif index == np.s_[1:-1]:
            return (p[..., 2:] - p[..., 1:-1]) / dt
        i = np.arange(paths.P)[index]
        if i[-1] == paths.P - 1:
            p = concatenate([p, p[..., -1, nax]], axis=2)
        return (p[..., i] - p[..., i - 1]) / dt

    def three_points_average_velocity(self, paths, index=np.s_[:], coords=None):
        """
        Returns Dim x M x P - 2 array
        """
        dt = self.dt
        p = paths.coords.copy()
        if coords is not None:
            p = coords
        if index == np.s_[:]:
            D, M = paths.DM
            p = concatenate([p[..., 0, nax], p, p[..., -1, nax]], axis=2)
            return (p[..., 2:] - p[..., :-2]) / (2 * dt)
        elif index == np.s_[1:-1]:
            return (p[..., 2:] - p[..., :-2]) / (2 * dt)
        P = paths.P
        i = np.arange(P)[index]
        if i[0] == 0:
            p = concatenate([p[..., 0, nax], p], axis=2)
            i += 1
        if i[-1] == P - 1:
            p = concatenate([p, p[..., -1, nax]], axis=2)
        return (p[..., i + 1] - p[..., i - 1]) / dt

    def four_points_average_velocity(self, paths, index=np.s_[1:-1],
                                     coords=None):
        dt = self.dt
        p = paths.coords
        if index == np.s_[1:-1]:
            return (p[..., 2:] - p[..., :-2]) / (2 * dt)
        elif index == np.s_[:]:
            D, M = paths.DM
            v = (p[..., 2:] - p[..., :-2]) / (2 * dt)
            v0 = (p[..., 1] - p[..., 0])[..., newaxis] / (2 * dt)
            vf = (p[..., -1] - p[..., -2])[..., newaxis] / (2 * dt)
            return np.concatenate([v0, v, vf], axis=2)
        attatch_initial = False
        attatch_final = False
        P = paths.P
        i = np.arange(P)[index]
        if i[0] == 0:
            attatch_initial = True
            i = i[1:]
        if i[-1] == P - 1:
            attatch_final = True
            i = i[:-1]
        if coords is not None:
            p = coords
        v = (p[..., i + 1] - p[..., i - 1]) / dt
        if attatch_initial:
            v = np.concatenate([v[:, :, 0, newaxis], v], axis=2)
        if attatch_final:
            v = np.concatenate([v, v[..., -1, newaxis]], axis=2)
        return v

    def two_points_velocity_gradient(self, paths, index=np.s_[:],
                                     coords=None):
        p = paths.coords
        ddt = self.dt * self.dt
        if coords is not None:
            p = coords
        if index == np.s_[:]:
            p = concatenate([p[..., 0, nax], p, p[..., -1, nax]], axis=2)
            return (2 * p[:, :, 1:-1] - p[:, :, :-2] - p[:, :, 2:]) / ddt
        elif index == np.s_[1:-1]:
            return (2 * p[:, :, 1:-1] - p[:, :, :-2] - p[:, :, 2:]) / ddt
        P = paths.P
        i = np.arange(P)[index]
        if i[0] == 0:
            p = concatenate([p[..., 0, nax], p], axis=2)
            i += 1
        if i[-1] == P - 1:
            p = concatenate([p, p[..., -1, nax]], axis=2)
        return (2 * p[..., i] - p[..., i - 1] - p[..., i + 1]) / ddt

    def three_points_velocity_gradient(self):
        return None

    def four_points_velocity_gradient(self):
        return None

    def two_points_acceleration(self):
        return None

    def three_points_acceleration(self, paths, index=np.s_[:], coords=None):
        """
        Get D x N x P-2 ndarray
        Returns 3 x N x P - 1 array
        """
        p = paths.coords
        ddt = self.dt * self.dt
        if coords is not None:
            p = coords
        if index == np.s_[:]:
            p = concatenate([p[..., 0, nax], p, p[..., -1, nax]], axis=2)
            return (2 * p[:, :, 1:-1] - p[:, :, :-2] - p[:, :, 2:]) / ddt
        elif index == np.s_[1:-1]:
            return (2 * p[:, :, 1:-1] - p[:, :, :-2] - p[:, :, 2:]) / ddt
        P = paths.P
        i = np.arange(P)[index]
        if i[0] == 0:
            p = concatenate([p[..., 0, nax], p], axis=2)
            i += 1
        if i[-1] == P - 1:
            p = concatenate([p, p[..., -1, nax]], axis=2)
        return (2 * p[..., i] - p[..., i - 1] - p[..., i + 1]) / ddt

    def four_points_acceleration(self):
        return None

    def get_effective_mass(self, paths, index=None):
        return atomic_masses[paths._numbers][:, np.newaxis]

    def get_momentum(self, paths, index=None):
        m = self.get_effective_mass(paths, index=index)
        v = self.get_velocity(paths, index=index)
        return m * v

    def get_kinetic_energy(self, paths, index=None):
        v = self.get_velocity(paths, index=index)
        m = self.get_effective_mass(paths, index=index)
        return np.sum(0.5 * m * v * v, axis=(0, 1))

    def get_kinetic_energy_gradient(self, paths, index=None, reduced=None):
        a = self.get_velocity_gradient(paths, index=index)
        m = self.get_effective_mass(paths, index=index)
        return m * a

    def default_X(self, value):
        return value

    def default_F(self, paths=None, forces=None):
        return forces


class Mask(Projector):
    prj_parameters = {
        'reference': {
            'default': 'None',
            'assert': 'True'
        },
        'mask': {
            'default': 'None',
            'assert': 'True'
        },
        'dimension': {
            'default': '3',
            'assert': '{name:s} in [1, 2, 3]'
        },
    }

    def __init__(self, reference=None, mask=None, dimension=None,
                 **kwargs):
        prj_parameters = super().prj_parameters
        prj_parameters.update(self.prj_parameters)
        self.prj_parameters.update(prj_parameters)

        self.reference = reference
        self.mask = mask
        self.dimension = dimension

        self._cache = {'count': 0, 'I_phi': [], 'I_psi': []}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def X(self, coords):
        return coords[:self.dimension, self.mask]

    def X_inv(self, coords_inv):
        P = coords_inv.shape[-1]
        reference = self.reference.T[..., np.newaxis] * np.ones((1, 1, P))
        reference[:self.dimension, self.mask] = coords_inv
        return reference

    def F(self, coords, forces):
        return forces[:self.dimension, self.mask]

    def F_inv(self, coords, forces):
        P = forces.shape[-1]
        N, D = self.reference.shape
        zero_pad_forces = np.zeros((D, N, P))
        zero_pad_forces[:self.dimension, self.mask] = forces
        return zero_pad_forces

    def get_effective_mass2(self, paths, index=None):
        # return np.array([[100]], [[75]])
        return atomic_masses[paths._numbers][self.mask]


class WyckoffProjector(Projector):
    """
    key_idx : list of tuples; mapping letters i'th operator to atom position j
        letters = ['C', 'H']
        key_idx = [(1, 0), (2, 1)]
    pos2wyc : list of tuples; letters and operator idx pair corresponding
              each atoms
        pos2wyc = [('a', 0), ('a', 1), ('b', 0) ...]
        pos2wyc = [(0, 0), (0, 1), (1, 0) ...]
    """
    prj_parameters = {
        'spacegroup': {'default': 'None', 'assert': 'True'},
        'wyckoff_symbols': {'default': 'None', 'assert': 'True'},
        'letters': {'default': 'None', 'assert': 'True'},
        'cell': {'default': 'None', 'assert': '{name:s}.shape == (3, 3)'},
        'key_idx': {'default': 'None', 'assert': 'True'},
        'pos2wyc': {'default': 'None', 'assert': 'True'}
    }

    def __init__(self, **kwargs):
        super().prj_parameters.update(self.prj_parameters)
        self.prj_parameters.update(super().prj_parameters)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def X(self, positions):
        """
        positions        : 3 x N x P
        cell_inv         : 3 x 3
        rotation[letter] : Multiplicity[letter] x 3 x 3
        trans[letter]    : 3 x Multiplicity[letter] x 1
        return           : 3 x M x P
        """
        rotation_inv, translation = self.rotation_inv, self.translation
        key_idx = self.key_idx
        sp = np.einsum('ij, j... -> i...', self.cell_inv, positions)
        wyck_coords = np.zeros((3, len(self.letters), len(sp.shape[-1])))
        for l in range(len(self.letters)):
            i, j = key_idx[l]
            r_inv, t = rotation_inv[l][i], translation[l][:, i]
            # 3 x 3 @ 3 x P -> 3 x P
            wyck_coords[:, l] = np.einsum('ij, jk -> ik', r_inv, sp[:, j] - t)
        return wyck_coords

    def X_inv(self, wyck_coords):
        """
        wyck_coords       : 3 x M(Letter) x P
        rotation[letter] : Multiplicity[letter] x 3 x 3
        trans[letter]    : 3 x Multiplicity[letter] x 1
        return           : 3 x N x P; positions
        """
        P = wyck_coords.shape[-1]
        multiplicity, N = self.multiplicity, self.N
        rotation, translation = self.rotation, self.translation
        scaled_positions = np.zeros((3, N, P))
        sp = scaled_positions
        i = 0
        for l in range(len(self.letters)):
            rot, trans, mul = rotation[l], translation[l], multiplicity[l]
            # 3 x 3 x M @ 3 x P  -> 3 x M x P + 3 x M x P -> 3 x M x P
            sp[:, i:i + mul] = np.einsum('mij, jp -> imp',
                                         rot, wyck_coords[:, l]) + trans
            i += mul
        # 3 x 3 @ 3 x N x P -> 3 x N x P
        return self.get_cartesian_positions(sp)

    def F(self, positions, forces, scaled=False):
        """
        positions         : 3 x N x P
        forces            : 3 x N x P
        direction[letter] : Multiplicity[letter] x 3
        return            : 3 x M(letters) x P
        """
        if not scaled:
            positions = self.get_scaled_positions(positions)
            forces = self.get_scaled_positions(forces)
        p, f = positions, forces
        rotation_inv = self.rotation_inv
        wyck_forces = np.zeros((3, len(self.letters), p.shape[-1]))
        i = 0
        for l, j in self.pos2wyc:
            r_inv = rotation_inv[l][j]
            # 3 x 3 @ 3 x P -> 3 x P
            wyck_forces[:, l] += np.einsum('ij, jk -> ik', r_inv, f[:, i])
            i += 1
        return wyck_forces / self.multiplicity[..., np.newaxis]

    def F_inv(self, coords, forces):
        raise NotImplementedError()

    def get_effective_mass(self, paths, index=None):
        if self.__dict__.get('_effective_mass') is not None:
            return self._effective_mass
        m = atomic_masses[self.wyckoff_symbols]
        self._effective_mass = (m * self.multiplicity)[..., np.newaxis]
        return self._effective_mass

    def get_scaled_positions(self, positions):
        """
        positions : 3 x N x P
        cell      : 3 x 3
        """
        return np.einsum('ij, jnp -> inp', self.cell_inv, positions)

    def get_cartesian_positions(self, scaled_positions):
        """
        scaled_positions : 3 x N x P
        cell             : 3 x 3
        """
        return np.einsum('ij, jnp -> inp', self.cell, scaled_positions)

    @property
    def wyckoff(self):
        if self.__dict__.get('_wyckoff') is None:
            self._wyckoff = Wyckoff(self.spacegroup)
        return self._wyckoff

    @property
    def N(self):
        if self.__dict__.get('_N') is None:
            self._N = self.multiplicity.sum()
        return self._N

    @property
    def multiplicity(self):
        if self.__dict__.get('_multiplicity') is None:
            wyckoff = self.wyckoff
            self._multiplicity = np.zeros(len(self.letters), dtype=int)
            for l, letter in enumerate(self.letters):
                self._multiplicity[l] = wyckoff[letter]['multiplicity']
        return self._multiplicity

    @property
    def rotation(self):
        if self.__dict__.get('_rotation') is None:
            self._rotation, self._translation = self.get_operators()
        return self._rotation

    @property
    def rotation_inv(self):
        if self.__dict__.get('_rotation_inv') is None:
            rotation = self.rotation
            self._rotation_inv = []
            for rot in rotation:
                _rot_inv = np.zeros(rot.shape)
                for i, r in enumerate(rot):
                    _rot_inv[i] = np.linalg.inv(r)
                self._rotation_inv.append(_rot_inv)
        return self._rotation_inv

    @property
    def translation(self):
        if self.__dict__.get('_translation') is None:
            self._rotation, self._translation = self.get_operators()
        return self._translation

    @property
    def cell_inv(self):
        if self.__dict__.get('_cell_inv') is None:
            self._cell_inv = np.linalg.inv(self.cell)
        return self._cell_inv

    def get_operators(self, letters=None):
        """
        rotation[letter] : Multiplicity[letter] x 3 x 3
        trans[letter]    : 3 x Multiplicity[letter] x 1
        """
        if letters is None:
            letters = self.letters
        wyckoff = self.wyckoff
        rotation, translation = [], []
        for letter, multiplicity in zip(letters, self.multiplicity):
            rot = np.zeros((wyckoff[letter]['multiplicity'], 3, 3))
            trans = np.zeros((3, wyckoff[letter]['multiplicity'], 1))
            r, t = parse_wyckoff_site(wyckoff[letter]['coordinates'])
            for i in range(multiplicity):
                rot[i] = r[i]
                trans[:, i] = t[i][..., nax]
            rotation.append(rot)
            translation.append(trans)
        return rotation, translation


class AlanineProjector(Projector):
    prj_parameters = {
        'reference': {'default': 'None', 'assert': 'True'}}
    mass_type = "variable"
    reference = np.array([[3.13042320, 8.69636925, 6.86034480],
                          [3.62171100, 7.75287090, 7.13119080],
                          [3.06158460, 6.94361235, 6.64450215],
                          [3.56832480, 7.62030150, 8.21956230],
                          [5.03124195, 7.76696535, 6.56486415],
                          [5.19044280, 7.93294500, 5.34189060],
                          [6.01497510, 7.58529855, 7.48734615],
                          [5.65167885, 7.45428060, 8.42656485],
                          [7.50000000, 7.50000000, 7.50000000],
                          [7.84286445, 8.35064955, 8.11409370],
                          [7.87916790, 6.20253030, 8.23173825],
                          [8.96770980, 6.15547755, 8.34399540],
                          [7.53870225, 5.32801485, 7.65999495],
                          [7.41872085, 6.16658700, 9.23056245],
                          [8.38912575, 7.62345720, 6.23286150],
                          [9.61384530, 7.55572530, 6.43116195],
                          [7.83695265, 7.81965300, 5.02677705],
                          [6.80236320, 7.87813665, 4.96104660],
                          [8.67108645, 7.97151630, 3.84856545],
                          [9.48211245, 8.68811745, 4.03978560],
                          [9.13567845, 7.01691960, 3.55615545],
                          [8.04737280, 8.33316525, 3.02266680]])

    def __init__(self, reference=None, **kwargs):
        super().prj_parameters.update(self.prj_parameters)
        self.prj_parameters.update(super().prj_parameters)

        self._cache = {'count': 0, 'I_phi': [], 'I_psi': [], 'Xn_T': []}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def X(self, positions):
        """
         positions : 3 x N x P
         returns : coords;  D x M x P, in this case 1 x 2 x P
        """

        phi = self.get_dihedral(positions, 4, 6, 8, 14)  # 1 x P
        psi = self.get_dihedral(positions, 6, 8, 14, 16)  # 1 x P
        return np.vstack([phi, psi])[np.newaxis, :]

    def X_inv(self, coords):
        """
         coords : D x M x P, in this case 1 x 2 x P
         returns : coords;  3 x N x P
        """
        coords = atleast_3d(coords)
        phi, psi = coords[0]
        p = self.reference.T[:, :, np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x N x P
        # Phi, Psi
        positions[:, :8] = self.rotate(p, phi, v=(6, 8), mask=np.s_[:8])
        positions[:, 14:] = self.rotate(p, psi, v=(14, 8), mask=np.s_[14:])
        return self.overlap_handler(positions)

    def F(self, positions, forces):
        """
         positions : 3 x N x P
         forces : 3 x N x P
         return D x M x P
        """
        mask, n = [4, 6, 14, 16], 2
        # mask, n = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21], 2
        c, phi_ax_idx, psi_ax_idx = (8, 6, 14)
        x = positions.copy()
        center = x[:, c]
        _p = x[:, :] - center[:, newaxis]
        phi_ax = _p[:, phi_ax_idx]
        psi_ax = _p[:, psi_ax_idx]

        p = _p[:, mask]
        lpl = norm(p, axis=0)
        e_p = p / lpl
        hax = repeat(phi_ax[:, newaxis], 2, axis=1)
        sax = repeat(psi_ax[:, newaxis], 2, axis=1)
        ax = concatenate((hax, sax), axis=1)
        laxl = norm(ax, axis=0)
        e_ax = ax / laxl

        r = lpl * (e_p - sum((e_p * e_ax), axis=0) * e_ax)
        f = forces[:, mask].copy()
        torque = cross(r, f, axis=0)
        thi = sum(torque[:, :n].sum(axis=1) * e_ax[:, 0], axis=0)
        tsi = sum(torque[:, n:].sum(axis=1) * e_ax[:, -1], axis=0)
        return np.array([thi, tsi])[np.newaxis, :]

    def F_inv(self, coords, forces):
        """
        coords : D x M x P; 2 x 1 x P
        forces : D x M x P; 2 x 1 x P
        reurn : force of 3 x 22 x P
        """
        self.reference
        raise NotImplementedError()

    def get_effective_mass(self, paths, index=np.s_[1:-1]):
        model_name = paths.model.__class__.__name__
        masses = atomic_masses[paths._numbers]
        if not model_name == 'Gaussian':
            return np.array([[110], [70]])
        if self.mass_type == 'invariant':
            return np.array([[110], [70]])
        #   return masses
        k, imgdata = paths.model.kernel, paths.imgdata
        hyperparameters = {'sigma_f': 0.1,
                           'sigma_n^f': 1e-3,
                           'sigma_n^e': 1e-4,
                           'l^2': 0.01}
        if self.found_new_data(imgdata):

            for row in imgdata._c.select('id>%d' % self._cache['count']):
                I_phi, I_psi = self.get_inertia(row.positions, masses)
                self._cache['I_phi'].append(I_phi)
                self._cache['I_psi'].append(I_psi)
                self._cache['Xn_T'].append(row.data.X.T)
                self._cache['count'] += 1
            Xn = np.array(self._cache['Xn_T']).T
            self._cache['K_inv'] = np.linalg.inv(
                k(Xn, Xn, orig=True, hyperparameters=hyperparameters))
            self._cache['m_phi'] = np.average(self._cache['I_phi'])
            self._cache['m_psi'] = np.average(self._cache['I_psi'])

        Xm = paths.coords[..., index].copy()
        Xn = np.array(self._cache['Xn_T']).T
        K_s = k(Xn, Xm, orig=True, hyperparameters=hyperparameters)
        K_inv = self._cache['K_inv']
        m_phi = self._cache['m_phi']
        m_psi = self._cache['m_psi']
        Y_phi = np.array(self._cache['I_phi'])
        Y_psi = np.array(self._cache['I_psi'])
        I_phi = m_phi + K_s.T @ K_inv @ (Y_phi - m_phi)
        I_psi = m_psi + K_s.T @ K_inv @ (Y_psi - m_psi)
        return np.array([[I_phi, I_psi]])   # inertia; I_theta, phi
        # return 100

    def found_new_data(self, imgdata):
        count = imgdata._c.count()
        if count == 0:
            raise NotImplementedError('No data found')
        is_count_changed = self._cache['count'] != count
        return is_count_changed

    def get_inertia(self, positions, masses):
        pos = positions.copy()
        pos -= positions[8]
        ephi = pos[6] / norm(pos[6])
        epsi = pos[14] / norm(pos[14])
        phi = pos[:8]   # 8 x 3
        psi = pos[14:]  # 8 x 3
        I_phi = masses[:8] * norm(np.cross(phi, ephi), axis=1) ** 2   # 8
        I_psi = masses[14:] * norm(np.cross(psi, epsi), axis=1) ** 2  # 8
        return I_phi.sum(), I_psi.sum()

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
        angle = np.arccos(angle) * 180 / np.pi
        reverse = np.sum(bxa * v_c, axis=0) > 0
        angle[reverse] = 360 - angle[reverse]
        return angle.reshape(1, -1)

    def rotate(self, ref, a=None, v=None, i=8, mask=None):
        """Rotate atoms based on a vector and an angle, or two vectors.
        Parameters:
         ref  : reference; C70; 3 x N x P  or  3 x N
         a    : angles   2 x P
         v    : tuple, rotation axis index
         i    : center index ; 8
        Return:
         c : 3 x N x P  or  3 x N
        """
        c = ref.copy()  # 3 x N x P
        v_i, v_f = v
        v = (c[:, v_i] - c[:, v_f])    # 3 x P or 3
        normv = norm(v, axis=0)  # P or 1

        if np.any(normv == 0.0):
            raise ZeroDivisionError('Cannot rotate: norm(v) == 0')

        # a *= pi / 180
        v /= normv                     # 3 x P  or  3
        co = cos(a)
        si = sin(a)                     # P  or  1
        center = c[:, i]               # 3 x P     or  3
        c = c.copy() - center[:, np.newaxis]  # 3 x N x P or 3 x N

        # 3 x N x P @ 3 x 1 x P  ->  N x P
        c0v = sum(c * v[:, np.newaxis], axis=0)
        r = c * co - cross(c, (v * si), axis=0)       # 3 x N x P
        # 3 x 1 x P (x) 1 x N x P -> 3 x N x P
        r = r + ((1.0 - c) * v)[:, np.newaxis] * c0v[np.newaxis, :]
        r = r + center[:, np.newaxis]
        return r[:, mask]

    def overlap_handler(self, coords):
        """
        Handles the too close atom # 17 - 5 distance issue:
        Data from ase.data covalent radii
        """
        c = coords.copy()
        cov_radii = (0.31 + 0.66)
        Hyd = c[:, 5, :]         # 3 x P
        Oxy = c[:, 17, :]        # 3 x P
        vec = Hyd - Oxy                     # 3 x P
        d = norm(vec, axis=0)               # P
        e_v = vec / d                       # 3 x P
        too_short = d < cov_radii * 0.7
        push = e_v[:, too_short] * cov_radii * 0.7 / 2
        c[:, 5, too_short] = Hyd[:, too_short] - push
        c[:, 17, too_short] = Oxy[:, too_short] + push
        return c


class MalonaldehydeProjector(Projector):
    prj_parameters = {
        'reference': {'default': 'None', 'assert': 'True'}}
    mass_type = "variable"
    reference = np.array([[-0.0001, -1.1149, 0.0069],
                          [1.2852, -0.3328, -0.0057],
                          [-1.2852, -0.3328, -0.0056],
                          [0.0000, 1.6434, 0.0000],
                          [-0.0001, -2.2187, 0.0000],
                          [2.2008, -0.9494, -0.0178],
                          [-2.2008, -0.9493, -0.0176],
                          [1.3685, 0.8902, 0.0022],
                          [-1.3685, 0.8902, 0.0022]])

    def __init__(self, reference=None, **kwargs):
        super().prj_parameters.update(self.prj_parameters)
        self.prj_parameters.update(super().prj_parameters)

        self._cache = {'count': 0}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def X(self, positions):
        """
         positions : 3 x N x P
         returns : coords;  D x M x P, in this case 2 x 1 x P
        """
        return positions[:2, 3, np.newaxis, :]

    def X_inv(self, coords):
        """
         coords : D x M x P, in this case 2 x 1 x P
         returns : coords;  3 x N x P
        """
        coords = atleast_3d(coords)
        p = self.reference.T[:, :, np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x N x P
        # Phi, Psi
        positions[:2, 3, np.newaxis] = coords
        return positions

    def F(self, positions, forces):
        """
         positions : 3 x N x P
         forces : 3 x N x P
         return D x M x P; in this case 2 x 1 x P
        """
        return forces[:2, 3, np.newaxis, :]

    def get_effective_mass(self, paths, index=np.s_[1:-1]):
        return np.array([[1]])
