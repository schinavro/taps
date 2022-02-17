import copy
import numpy as np
from numpy import concatenate
from numpy import newaxis as nax
from taps.coords import Coordinate
from taps.utils.arraywrapper import arraylike


# @arraylike
class Cartesian(Coordinate):

    """
    Discretized coordinate representaion of a system. In default, system is
    considered as Cartesian.

    Parameters
    ----------

    coords: numpy array shape of DxN or 3xAxN

    epoch: float
       Total transition time of a system
    Nk: int
       Number of sine component representation of a system
    unit: string
      length unit of a system. Currently it is only for a display purpose.
      (TODO: automatic unit matching with Model class)
    """
    def __init__(self, mass=1., **kwargs):
        self.mass = mass
        super().__init__(**kwargs)

    def masses(self, paths, **kwargs):
        if isinstance(self.mass, np.ndarray):
            return self.mass[..., nax]
        return self.mass

    def displacements(self, paths, **kwargs):
        """
        Return vector
        """
        init = self.coords[..., 0, nax]
        # p = coords or self.coords
        return self.coords - init

    def velocities(self, paths, coords=None, epoch=None, index=np.s_[:]):
        """ Return velocity at each step
        Get coords and return DxN or 3xAxN array, two point moving average.

        :math:`v[i] = (x[i+1] - x[i]) / dt`

        Parameters
        ----------
        coords : array
            size of DxN or 3xAxN
        epoch : float
            total time step.
        index : slice obj; Default `np.s_[:]`
            Choose the steps want it to be returned. Default is all steps.
        """
        p = coords or self.coords.copy()
        # epoch, N = epoch or self.epoch, self.N
        dt = self.dt
        if index == np.s_[:]:
            p = np.concatenate([p, p[..., -1, nax]], axis=-1)
            return (p[..., 1:] - p[..., :-1]) / dt
            # vel = np.diff(p, n=1, prepend=p[..., 0, nax]) / dt
            # return vel
        elif index == np.s_[1:-1]:
            return (p[..., 2:] - p[..., 1:-1]) / dt
        i = np.arange(N)[index]
        if i[-1] == N - 1:
            p = concatenate([p, p[..., -1, nax]], axis=-1)
        return (p[..., i] - p[..., i - 1]) / dt

    def accelerations(self, paths, coords=None, index=np.s_[:]):
        """ Return acceleration at each step
        Get Dx N ndarray, Returns 3xNxP - 1 array, use three point to get
        acceleration

        :math:`a[i] = (2x[i] - x[i+1] - x[i-1]) / dtdt`

        Parameters
        ----------
        coords : array
            size of DxN or 3xAxN
        epoch : float
            total time step.
        index : slice obj; Default `np.s_[:]`
            Choose the steps want it to be returned. Default is all steps.
        """
        p = coords or self.coords.copy()
        dt = self.dt
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


class AlanineDipeptideCartesian(Cartesian):
    """
    C70 : phi = -8.537736562175269e-07 psi = 8.537736462515939e-07
    C7eq : phi = -60.00000014729346 psi = 100.25000021100078
    C7ax : phi = 54.99999985270659 psi = -59.999999788999276
    """
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
