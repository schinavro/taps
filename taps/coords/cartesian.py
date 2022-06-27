import numpy as np
from numpy import concatenate
from numpy import newaxis as nax
from taps.coords import Coordinate


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

    def __getitem__(self, idx):
        return self.coords[..., idx]

    def __call__(self, index=np.s_[:], coords=None):
        if coords is not None:
            kwargs = self.__dict__.copy()
            del kwargs['coords']
            return self.__class__(coords=coords, **kwargs)
        if index.__class__.__name__ == 'slice' and index == np.s_[:]:
            return self
        kwargs = self.__dict__.copy()
        del kwargs['coords']
        idx = np.arange(self.N)[index].reshape(-1)
        coords = self.coords[..., idx]
        return self.__class__(coords=coords, **kwargs)

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

    def velocities(self, paths, coords=None, index=np.s_[:]):
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
        N = self.N
        dt = self.dt
        if index == np.s_[:]:
            p = np.concatenate([p, p[..., -1, nax]], axis=-1)
            # return (p[..., 1:] - p[..., :-1]) / dt
            vel = (p[..., 1:] - p[..., :-1]) / dt
            vel[..., -1] = vel[..., -2]
            return vel
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
        N = self.N
        dt = self.dt
        ddt = dt * dt
        if index == np.s_[:]:
            p = concatenate([p[..., 0, nax], p, p[..., -1, nax]], axis=-1)
            # return (2 * p[..., 1:-1] - p[..., :-2] - p[..., 2:]) / ddt
            acc = (2 * p[..., 1:-1] - p[..., :-2] - p[..., 2:]) / ddt
            acc[..., -1] = acc[..., -2]
            return acc
        elif index == np.s_[1:-1]:
            return (2 * p[..., 1:-1] - p[..., :-2] - p[..., 2:]) / ddt
        i = np.arange(N)[index]
        if i[0] == 0:
            p = concatenate([p[..., 0, nax], p], axis=-1)
            i += 1
        if i[-1] == N - 1:
            p = concatenate([p, p[..., -1, nax]], axis=-1)
        return (2 * p[..., i] - p[..., i - 1] - p[..., i + 1]) / ddt
