from taps.prorjectors import Projector
from ase.wyckoff.wyckoff import Wyckoff
from ase.wyckoff.xtal2 import parse_wyckoff_site


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

    @Projector.pipeline
    def x(self, coords):
        """
        positions        : 3 x N x P
        cell_inv         : 3 x 3
        rotation[letter] : Multiplicity[letter] x 3 x 3
        trans[letter]    : 3 x Multiplicity[letter] x 1
        return           : 3 x M x P
        """
        rotation_inv, translation = self.rotation_inv, self.translation
        key_idx = self.key_idx
        sp = np.einsum('ij, j... -> i...', self.cell_inv, coords)
        wyck_coords = np.zeros((3, len(self.letters), len(sp.shape[-1])))
        for l in range(len(self.letters)):
            i, j = key_idx[l]
            r_inv, t = rotation_inv[l][i], translation[l][:, i]
            # 3 x 3 @ 3 x P -> 3 x P
            wyck_coords[:, l] = np.einsum('ij, jk -> ik', r_inv, sp[:, j] - t)
        return wyck_coords

    @Projector.pipeline
    def x_inv(self, wyck_coords):
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
