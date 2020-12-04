import numpy as np
from numpy import concatenate
from numpy import newaxis as nax
from scipy.fftpack import dst, idst
from taps.utils.arraywrapper import arraylike


@arraylike
class Coords:
    """
    Default - descretized coordinate representation.
    """
    def __init__(self, coords=None, epoch=3, _Nk=None, Nk=None, unit=None):
        coords = np.asarray(coords, dtype=float)
        self.coords = coords
        self.epoch = epoch
        self._Nk = _Nk or Nk
        self.unit = unit

    def __call__(self, index=np.s_[:], coords=None):
        if coords is not None:
            kwargs = self.__dict__.copy()
            del kwargs['coords']
            return self.__class__(coords, **kwargs)
        if index.__class__.__name__ == 'slice' and index == np.s_[:]:
            return self
        kwargs = self.__dict__.copy()
        del kwargs['coords']
        idx = np.arange(self.N)[index].reshape(-1)
        coords = self.coords[..., idx]
        return self.__class__(coords, **kwargs)

    @classmethod
    def ascoords(cls, coords):
        """Return argument as a Cell object.  See :meth:`ase.cell.Cell.new`.

        A new Cell object is created if necessary."""
        if isinstance(coords, cls):
            return coords
        return cls.new(coords)

    @classmethod
    def new(cls, coords=None):
        """Create new coords from any parameters.

        If cell is three numbers, assume three lengths with right angles.

        If cell is six numbers, assume three lengths, then three angles.

        If cell is 3x3, assume three cell vectors."""

        if coords is None:
            coords = np.zeros((3, 3))

        coords = np.array(coords, float)

        coordsobj = cls(coords)
        return coordsobj

    def __array__(self, dtype=float):
        if dtype != float:
            raise ValueError('Cannot convert cell to array of type {}'
                             .format(dtype))
        return self.coords

    def __bool__(self):
        return bool(self.any())  # need to convert from np.bool_

    def __repr__(self):
        return 'Coords{}'.format(self.coords.shape)

    @property
    def N(self):
        """
        Number of steps
        """
        return self.coords.shape[-1]

    @property
    def Nk(self):
        return self._Nk or self.N - 2

    @Nk.setter
    def Nk(self, Nk=None):
        self._Nk = Nk

    @property
    def D(self):
        """
        Total dimension
        """
        return len(self.coords.shape)

    @property
    def A(self):
        """
        Number of individual components
        """
        return 1

    def get_displacements(self, coords=None, epoch=None, index=np.s_[:]):
        p = coords or self.coords
        shape = p.shape
        if len(shape) == 2:
            d = np.linalg.norm(np.diff(p), axis=0)
            d = concatenate([[0], d], axis=-1)
            return np.add.accumulate(d)[index]
        d = np.linalg.norm(np.diff(p), axis=0)
        d = concatenate([np.zeros((shape[1], 1)), d])
        return np.add.accumulate(d, axis=-1)[..., index]

    def get_velocity(self, coords=None, epoch=None, index=np.s_[:]):
        """
        Returns Dim x M x P array, two point
        """
        p = coords or self.coords.copy()
        epoch, N = epoch or self.epoch, self.N
        dt = epoch / N
        if index == np.s_[:]:
            p = np.concatenate([p, p[..., -1, nax]], axis=-1)
            return (p[..., 1:] - p[..., :-1]) / dt
        elif index == np.s_[1:-1]:
            return (p[..., 2:] - p[..., 1:-1]) / dt
        i = np.arange(N)[index]
        if i[-1] == N - 1:
            p = concatenate([p, p[..., -1, nax]], axis=-1)
        return (p[..., i] - p[..., i - 1]) / dt

    def get_acceleration(self, coords=None, epoch=None, index=np.s_[:]):
        """
        Get D x N x P-2 ndarray
        Returns 3 x N x P - 1 array
        """
        p = coords or self.coords.copy()
        epoch, N = epoch or self.epoch, self.N
        dt = epoch / N
        ddt = dt * dt
        if index == np.s_[:]:
            p = concatenate([p[..., 0, nax], p, p[..., -1, nax]], axis=-1)
            return (2 * p[..., 1:-1] - p[..., :-2] - p[..., 2:]) / ddt
        elif index == np.s_[1:-1]:
            return (2 * p[..., 1:-1] - p[..., :-2] - p[..., 2:]) / ddt
        i = np.arange(N)[index]
        if i[0] == 0:
            p = concatenate([p[..., 0, nax], p], axis=-1)
            i += 1
        if i[-1] == N - 1:
            p = concatenate([p, p[..., -1, nax]], axis=-1)
        return (2 * p[..., i] - p[..., i - 1] - p[..., i + 1]) / ddt

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

    def flat(self):
        N = self.N
        return self.coords.reshape((-1, N))


class AlanineDipeptideCoords(Coords):
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

    def __init__(self, *args, reference=None, **kwargs):
        self.reference = reference or self.reference
        super().__init__(*args, **kwargs)

    def get_real_model_representation(self, index=np.s_[:]):
        """
         coords : D x P, in this case 2 x P
         returns : coords;  3 x N x P
        """
        idx = np.arange(self.N)[index]
        coords = self.coords[..., idx]
        phi, psi = coords
        p = self.reference.T[..., np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x N x P
        # Phi, Psi
        positions[:, :8] = self.rotate(p, phi, v=(6, 8), mask=np.s_[:8])
        positions[:, 14:] = self.rotate(p, psi, v=(14, 8), mask=np.s_[14:])
        return self.overlap_handler(positions)

    def projector(self, positions):
        """
         positions : 3 x N x P
         returns : coords;  D x M x P, in this case 1 x 2 x P
        """

        phi = self.get_dihedral(positions, 4, 6, 8, 14)  # 1 x P
        psi = self.get_dihedral(positions, 6, 8, 14, 16)  # 1 x P
        return np.vstack([phi, psi])[np.newaxis, :]
