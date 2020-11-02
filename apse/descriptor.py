from sbdesc.sb_desc import Sb_desc
import numpy as np


class Descriptor:
    def __init__(self):
        pass


atomic_weights = {'H': 0.2, 'C': 0.5, 'N': 0.7, 'O': 0.9, 'Ni': 1.0}


class SphericalHarmonicDescriptor(Descriptor):
    allowed_parameters = ['weight', 'n_atom', 'Q']

    def __init__(self, cutoff_rad=3.7755, n_max=4, libname='libsbdesc.so',
                 libdir='/group/schinavro/libCalc/sbdesc/lib/', **kwargs):
        """
        https://github.com/harharkh/sb_desc
        """
        self.libname = libname
        self.libdir = libdir
        self.sb_desc = Sb_desc(libname, libdir)
        self.cutoff_rad = cutoff_rad
        self.n_max = n_max
        for key, value in kwargs.items():
            if key in self.allowed_parameters:
                setattr(self, key, value)
            else:
                raise TypeError('__init__() got unexpected kwargs %s' % key)

    def __call__(self, paths, coords):
        """
        # Displacements to the surrounding atoms in Angstroms
        # [x_1, y_1, z_1, ...]
        return : M x A x Q
        """
        self._paths = getattr(self, '_paths', None) or paths
        w, cr, na, nm = self.weight, self.cutoff_rad, self.n_atom, self.n_max
        positions_arr = coords.T
        M, A = positions_arr.shape[:2]
        Dm = np.zeros((M, A, self.Q))
        for i, positions in enumerate(positions_arr):
            displacements = self.get_displacements(positions)
            for j, disp in enumerate(displacements):
                Dm[i, j] = self.sb_desc(disp, w[j], cr, na, nm)
        return Dm

    def get_displacements(self, positions):
        """
        positions : A x 3
        disp : A x 3(A-1)
        """
        A = len(positions)
        disp = np.zeros((A, 3 * (A - 1)))
        for i, position in enumerate(positions):
            disp[i] = (np.delete(positions - position, i, 0)).flatten()
        return disp

    @property
    def Q(self):
        if getattr(self, '_Q', None) is not None:
            return self._Q
        self._Q = ((self.n_max + 1) * (self.n_max + 2)) // 2
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    @property
    def weight(self):
        if getattr(self, '_weight', None) is not None:
            return self._weight
        A, symbols = self._paths.A, self._paths.symbols

        weight = np.array([atomic_weights[sym] for sym in symbols])
        self._weight = np.zeros((A, A - 1))
        for i in range(A):
            self._weight[i] = np.roll(weight, i)[1:]
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @property
    def n_atom(self):
        if getattr(self, '_n_atom', None) is not None:
            return self._n_atom
        self._n_atom = self._paths.A - 1
        return self._n_atom

    @n_atom.setter
    def n_atoms(self, n_atom):
        self._n_atom = n_atom
